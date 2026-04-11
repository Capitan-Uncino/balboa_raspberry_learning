use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use rppal::gpio::{Gpio, Trigger};
use rppal::i2c::I2c;
use std::convert::TryInto;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const ONLINE: bool = false;
const NEW_BATCH: bool = true;

#[derive(Debug, Clone, Copy)]
pub struct StateAction {
    pub phi: f32,
    pub phi_dot: f32,
    pub theta: f32,
    pub theta_dot: f32,
    pub u: f32,
}

// --- System Dimensions ---
const DIM_X: usize = 4; // [theta, theta_dot]
const DIM_U: usize = 1; // [torque]
const DIM_X_AND_U: usize = DIM_X + DIM_U;
const DIM_PARAMS: usize = (DIM_X_AND_U * (DIM_X_AND_U + 1)) / 2;

// --- LSPI Hyperparameters ---
const GAMMA: f64 = 0.99; // Discount factor
const SAMPLES_PER_ITER: usize = 10000; // Samples per policy evaluation
const LAMBDA_REG: f64 = 1e-5; // L2 Regularization
const ETA: f64 = 1.068;

fn get_quadratic_features(
    x: &SVector<f64, DIM_X>,
    u: &SVector<f64, DIM_U>,
) -> SVector<f64, DIM_PARAMS> {
    let mut feat = SVector::<f64, DIM_PARAMS>::zeros();
    let mut z = SVector::<f64, DIM_X_AND_U>::zeros();
    z.fixed_view_mut::<DIM_X, 1>(0, 0).copy_from(x);
    z.fixed_view_mut::<DIM_U, 1>(DIM_X, 0).copy_from(u);
    let mut idx = 0;
    for i in 0..DIM_X_AND_U {
        for j in i..DIM_X_AND_U {
            feat[idx] = z[i] * z[j];
            idx += 1;
        }
    }
    feat
}

fn theta_to_h(theta: &SVector<f64, DIM_PARAMS>) -> SMatrix<f64, DIM_X_AND_U, DIM_X_AND_U> {
    let mut h_mat = SMatrix::<f64, DIM_X_AND_U, DIM_X_AND_U>::zeros();
    let mut idx = 0;
    for i in 0..DIM_X_AND_U {
        for j in i..DIM_X_AND_U {
            let val = theta[idx];
            if i == j {
                h_mat[(i, j)] = val;
            } else {
                h_mat[(i, j)] = val * 0.5;
                h_mat[(j, i)] = val * 0.5;
            }
            idx += 1;
        }
    }
    h_mat
}

fn compute_k_from_h(h_mat: &SMatrix<f64, DIM_X_AND_U, DIM_X_AND_U>) -> SMatrix<f64, DIM_U, DIM_X> {
    let q_uu = h_mat.fixed_view::<DIM_U, DIM_U>(DIM_X, DIM_X);
    let q_ux = h_mat.fixed_view::<DIM_U, DIM_X>(DIM_X, 0);
    match q_uu.try_inverse() {
        Some(inv) => inv * q_ux,
        None => SMatrix::<f64, DIM_U, DIM_X>::identity(),
    }
}

fn spectral_radius(
    a_mat: &SMatrix<f64, DIM_X, DIM_X>,
    b_mat: &SMatrix<f64, DIM_X, DIM_U>,
    k: &SMatrix<f64, DIM_U, DIM_X>,
) -> f64 {
    let a_cl = a_mat - b_mat * k;
    let eig = a_cl.complex_eigenvalues();
    eig.iter()
        .map(|c| c.norm())
        .fold(0.0, |a, b| f64::max(a, b))
}

pub fn run_lstdq(
    batch: Vec<StateAction>,
    k: &SMatrix<f64, DIM_U, DIM_X>,
) -> SVector<f64, DIM_PARAMS> {
    let mut a_mat = DMatrix::<f64>::zeros(DIM_PARAMS, DIM_PARAMS);
    let mut b_vec = DVector::<f64>::zeros(DIM_PARAMS);

    // 1. Define your Q and R penalty matrices here since 'sys' is removed.
    // Tune these to penalize chassis tilt vs wheel movement vs battery usage.
    let q_cost = SMatrix::<f64, DIM_X, DIM_X>::from_diagonal(&SVector::from([
        10.0, // phi penalty
        1.0,  // phi_dot penalty
        1.0,  // theta penalty
        0.1,  // theta_dot penalty
    ]));
    let r_cost = SMatrix::<f64, DIM_U, DIM_U>::from_diagonal(&SVector::from([1.0]));

    // Construct matrix M = [I; K]
    let mut m_mat = SMatrix::<f64, DIM_X_AND_U, DIM_X>::zeros();
    m_mat
        .fixed_view_mut::<DIM_X, DIM_X>(0, 0)
        .copy_from(&SMatrix::<f64, DIM_X, DIM_X>::identity());
    m_mat.fixed_view_mut::<DIM_U, DIM_X>(DIM_X, 0).copy_from(k);

    // Compute the outer product: P = M * M^T
    let p_mat = m_mat * m_mat.transpose();

    // Vectorize P to get the bias vector: bias = eta * svec(P)
    let mut bias_vec = SVector::<f64, DIM_PARAMS>::zeros();
    let mut idx = 0;
    for i in 0..DIM_X_AND_U {
        for j in i..DIM_X_AND_U {
            let val = p_mat[(i, j)];
            bias_vec[idx] = val * ETA;
            idx += 1;
        }
    }

    // 2. Iterate directly over the raw chronological batch
    // We stop at batch.len() - 1 because we need a "next" state for every "current" state
    for i in 0..(batch.len() - 1) {
        let current = &batch[i];
        let next = &batch[i + 1];

        // Construct mathematical vectors directly from the I2C structs
        let x = SVector::<f64, DIM_X>::from_column_slice(&[
            current.phi as f64,
            current.phi_dot as f64,
            current.theta as f64,
            current.theta_dot as f64,
        ]);
        let u = SVector::<f64, DIM_U>::from_column_slice(&[current.u as f64]);

        let x_next = SVector::<f64, DIM_X>::from_column_slice(&[
            next.phi as f64,
            next.phi_dot as f64,
            next.theta as f64,
            next.theta_dot as f64,
        ]);

        // Cost calculation using our local Q and R
        let cost = x.dot(&(q_cost * x)) + u.dot(&(r_cost * u));

        // Feature computation with BIAS using the NOISY action applied
        let phi = get_quadratic_features(&x, &u) + bias_vec;

        // Target policy uses the current gain matrix K (Greedy action)
        let u_next_greedy = -k * x_next;
        let phi_next = get_quadratic_features(&x_next, &u_next_greedy) + bias_vec;

        // LSTDQ Update
        let temporal_diff = phi - (GAMMA * phi_next);

        for r in 0..DIM_PARAMS {
            let phi_r = phi[r];
            b_vec[r] += phi_r * cost;
            for c in 0..DIM_PARAMS {
                a_mat[(r, c)] += phi_r * temporal_diff[c];
            }
        }
    }

    // Apply L2 Regularization
    for i in 0..DIM_PARAMS {
        a_mat[(i, i)] += LAMBDA_REG;
    }

    // Solve for theta
    let theta_dyn = a_mat
        .lu()
        .solve(&b_vec)
        .unwrap_or(DVector::zeros(DIM_PARAMS));

    let mut theta = SVector::<f64, DIM_PARAMS>::zeros();
    theta.copy_from_slice(theta_dyn.as_slice());

    theta
} // --- Helpers ---
  //

fn calculate_k(
    batch: Vec<StateAction>,
    current_k: &SMatrix<f64, DIM_U, DIM_X>,
) -> SMatrix<f64, DIM_U, DIM_X> {
    let theta = run_lstdq(batch, current_k);
    let h_mat = theta_to_h(&theta);
    compute_k_from_h(&h_mat)
}

fn write_gains(i2c_bus: &Arc<Mutex<I2c>>, gains: [f32; 4]) {
    let mut write_buf = Vec::with_capacity(17);
    write_buf.push(0u8); // Offset 0
    for k in &gains {
        write_buf.extend_from_slice(&k.to_le_bytes());
    }

    if let Ok(mut i2c) = i2c_bus.lock() {
        println!(
            "---> Sending Gains: [k1: {:.3}, k2: {:.3}, k3: {:.3}, k4: {:.3}]",
            gains[0], gains[1], gains[2], gains[3]
        );

        if let Err(e) = i2c.write(&write_buf) {
            eprintln!("I2C Write Error: {:?}", e);
        }
    } else {
        eprintln!("Failed to acquire I2C lock for writing.");
    }
}

fn log_progress(current: usize, total: usize, completed: usize, mode: &str) {
    let step = (total / 10).max(1);
    if current > 0 && current % step == 0 {
        let percent = (current as f32 / total as f32) * 100.0;
        let bars = (percent / 10.0) as usize;
        let bar_str = format!("[{}{}]", "=".repeat(bars), " ".repeat(10 - bars));

        println!(
            "[INFO] {} {} {:>3.0}% ({:>4}/{:>4}) | Total Completed: {}",
            mode, bar_str, percent, current, total, completed
        );
    }
}

/// Scans the directory for batch_X.csv files and returns the next available index.
fn get_next_file_index() -> usize {
    let mut highest = 0;
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("batch_") && name.ends_with(".csv") {
                    // Extract X from "batch_X.csv"
                    let parts: Vec<&str> = name.split('_').collect();
                    if let Some(num_part) = parts.get(1) {
                        if let Some(num_str) = num_part.split('.').next() {
                            if let Ok(num) = num_str.parse::<usize>() {
                                if num >= highest {
                                    highest = num + 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    highest
}

fn collect_full_batch(
    i2c_bus: &Arc<Mutex<I2c>>,
    log_label: &str,
    batch_index: usize,
    was_balancing: &mut bool,
    pending_gains: &Arc<Mutex<Option<[f32; 4]>>>,
) -> Vec<StateAction> {
    let mut state_batch = Vec::with_capacity(SAMPLES_PER_ITER);
    let mut loop_counter = 0;

    // --- HARDWARE SYNC SETUP ---
    let gpio = Gpio::new().expect("Failed to initialize GPIO");
    let mut sync_pin = gpio.get(22).expect("GPIO 22 busy").into_input();

    sync_pin
        .set_interrupt(Trigger::RisingEdge, None)
        .expect("Failed to set interrupt");

    let mut stability_counter = 0;
    let stability_threshold = 10;
    let stop_angle_rad = 60.0_f32.to_radians();

    println!(">>> Syncing with Balboa Hardware Clock on GPIO 22...");

    while state_batch.len() < SAMPLES_PER_ITER {
        // 1. WAIT FOR ARDUINO SIGNAL (Blocks until Window Opens)
        // 15ms timeout ensures we don't hang forever if the Arduino is off
        let _ = sync_pin.poll_interrupt(true, Some(Duration::from_millis(15)));

        // 2. STRICT ENFORCEMENT CHECK
        // If the pin is no longer high, Linux stalled our thread. Skip this cycle!
        if sync_pin.is_low() {
            continue;
        }

        // 3. READ DATA OVER I2C
        let mut buf = [0u8; 20];
        let read_result = {
            let mut bus = i2c_bus.lock().unwrap();
            bus.read(&mut buf)
        };

        if let Ok(20) = read_result {
            // =======================================================
            // 4. WRITE NEW GAINS (IF READY)
            // Done immediately after reading while window is still open
            // =======================================================
            if let Some(new_gains) = pending_gains.lock().unwrap().take() {
                write_gains(i2c_bus, new_gains);
            }

            // 5. PARSE BYTES
            let data = StateAction {
                phi: f32::from_le_bytes(buf[0..4].try_into().unwrap()),
                phi_dot: f32::from_le_bytes(buf[4..8].try_into().unwrap()),
                theta: f32::from_le_bytes(buf[8..12].try_into().unwrap()),
                theta_dot: f32::from_le_bytes(buf[12..16].try_into().unwrap()),
                u: f32::from_le_bytes(buf[16..20].try_into().unwrap()),
            };

            // 6. DATA SANITY CHECKS
            let is_sane = data.theta.is_finite() && data.theta_dot.abs() < 100.0;
            let is_upright = data.theta < 900.0 && data.theta.abs() < stop_angle_rad;

            // 7. STABILITY LOGIC
            if is_sane && is_upright {
                stability_counter += 1;
                if stability_counter >= stability_threshold {
                    if !*was_balancing {
                        println!(">>> ROBOT STANDING: Resuming {}...", log_label);
                        *was_balancing = true;
                    }

                    state_batch.push(data);
                    loop_counter += 1;

                    // Progress logging
                    if loop_counter >= 100 {
                        log_progress(state_batch.len(), SAMPLES_PER_ITER, batch_index, log_label);
                        loop_counter = 0;
                    }
                }
            } else {
                // If data is corrupted or robot is down, reset the stability window
                stability_counter = 0;
                if *was_balancing {
                    println!(
                        "<<< ROBOT FELL: Pausing {} (Collected: {}/{})",
                        log_label,
                        state_batch.len(),
                        SAMPLES_PER_ITER
                    );
                    *was_balancing = false;
                }
            }
        } else {
            // I2C read failed, reset stability
            stability_counter = 0;
        }
    }

    // Clean up interrupt before exiting so it doesn't leak or conflict later
    let _ = sync_pin.clear_interrupt();

    state_batch
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("========================================");
    println!("    BALBOA BRAIN v2.0 - I2C ENABLED     ");
    println!("========================================");
    println!("Mode flags: ONLINE={}, NEW_BATCH={}", ONLINE, NEW_BATCH);

    if ONLINE {
        run_online_mode()?;
    } else if NEW_BATCH {
        run_data_collection_mode()?;
    } else {
        run_offline_computation_mode()?;
    }

    Ok(())
}

fn run_online_mode() -> Result<(), Box<dyn Error>> {
    let mut i2c = I2c::new().map_err(|e| format!("Failed to init I2C: {}", e))?;
    i2c.set_slave_address(0x08)
        .map_err(|e| format!("Failed to set I2C address: {}", e))?;
    let i2c_bus = Arc::new(Mutex::new(i2c));

    let initial_k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(&[100.0, 5.0, 2.0, 0.5]);
    let current_k = Arc::new(Mutex::new(initial_k_mat));

    // NEW: Shared state to hold gains computed by the background thread
    let pending_gains: Arc<Mutex<Option<[f32; 4]>>> = Arc::new(Mutex::new(None));

    let mut computations_completed = 0;
    let mut was_balancing = false;

    println!("Starting 100Hz I2C control loop on Address 0x08...");

    loop {
        let batch_to_process = collect_full_batch(
            &i2c_bus,
            "LSTDQ Batch",
            computations_completed,
            &mut was_balancing,
            &pending_gains, // Pass the shared state
        );

        computations_completed += 1;
        let k_clone = Arc::clone(&current_k);
        let pending_clone = Arc::clone(&pending_gains);

        // Spawn math thread
        thread::spawn(move || {
            let k_to_use = { *k_clone.lock().unwrap() };
            let new_k_mat = calculate_k(batch_to_process, &k_to_use);

            {
                *k_clone.lock().unwrap() = new_k_mat;
            }

            let new_k_array = [
                new_k_mat[(0, 0)] as f32,
                new_k_mat[(0, 1)] as f32,
                new_k_mat[(0, 2)] as f32,
                new_k_mat[(0, 3)] as f32,
            ];

            // Store the calculated gains so the main thread can grab them
            // during its next 5ms hardware window
            *pending_clone.lock().unwrap() = Some(new_k_array);
            println!(">>> LSTDQ Update: New K vector queued for next I2C window.");
        });
    }
}

fn run_data_collection_mode() -> Result<(), Box<dyn Error>> {
    let mut i2c = I2c::new().map_err(|e| format!("Failed to init I2C: {}", e))?;
    i2c.set_slave_address(0x08)
        .map_err(|e| format!("Failed to set I2C address: {}", e))?;
    let i2c_bus = Arc::new(Mutex::new(i2c));

    // SMART INDEX: Start from the highest existing file + 1
    let mut file_index = get_next_file_index();
    let mut was_balancing = false;

    // NEW: Dummy container to satisfy the collect_full_batch signature
    let dummy_pending_gains: Arc<Mutex<Option<[f32; 4]>>> = Arc::new(Mutex::new(None));

    println!(
        "Started data collection mode. Will start at index: {}",
        file_index
    );

    loop {
        // Pass the dummy_pending_gains as the 5th argument
        let batch_to_process = collect_full_batch(
            &i2c_bus,
            "CSV Collection",
            file_index,
            &mut was_balancing,
            &dummy_pending_gains,
        );

        let filename = format!("batch_{}.csv", file_index);
        let mut file = File::create(&filename)?;

        writeln!(file, "phi,phi_dot,theta,theta_dot,u")?;
        for s in &batch_to_process {
            writeln!(
                file,
                "{},{},{},{},{}",
                s.phi, s.phi_dot, s.theta, s.theta_dot, s.u
            )?;
        }

        println!(
            ">>> Saved batch of size {} to {} (Next: {})",
            SAMPLES_PER_ITER,
            filename,
            file_index + 1
        );
        file_index += 1;
    }
}

/// Mode 3: ONLINE = false, NEW_BATCH = false
/// Prompts for a filename, loads the CSV into a batch, computes K, and prints it.
fn run_offline_computation_mode() -> Result<(), Box<dyn Error>> {
    print!("Enter the CSV file name to process (e.g., batch_0.csv): ");
    io::stdout().flush()?;

    let mut filename = String::new();
    io::stdin().read_line(&mut filename)?;
    let filename = filename.trim();

    println!("Reading data from {}...", filename);
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut batch: Vec<StateAction> = Vec::new();

    // Skip the header row
    for line in contents.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 5 {
            batch.push(StateAction {
                phi: parts[0].parse()?,
                phi_dot: parts[1].parse()?,
                theta: parts[2].parse()?,
                theta_dot: parts[3].parse()?,
                u: parts[4].parse()?,
            });
        }
    }

    if batch.is_empty() {
        println!("No valid data found in file.");
        return Ok(());
    }

    println!("Loaded {} records. Computing K...", batch.len());

    // Provide the initial K matrix required for computation
    let initial_k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(&[100.0, 5.0, 2.0, 0.5]);

    // Call the computation algorithm
    let new_k_mat = calculate_k(batch, &initial_k_mat);

    println!("========================================");
    println!(">>> COMPUTED K MATRIX RESULT <<<");
    println!("{}", new_k_mat);
    println!("========================================");

    Ok(())
}
