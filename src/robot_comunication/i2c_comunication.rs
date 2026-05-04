use crate::file_utils::get_next_file_index;
use crate::learning::lstdq::{calculate_k, StateAction, ANALYTIC_LQR_POLICY, SAMPLES_PER_ITER};
use crate::logging_utils::log_progress;
use rppal::gpio::{Gpio, Trigger};
use rppal::i2c::I2c;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

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
    let stop_angle_rad = 60.0_f64.to_radians();

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
                phi: f32::from_le_bytes(buf[0..4].try_into().unwrap()) as f64,
                theta: f32::from_le_bytes(buf[4..8].try_into().unwrap()) as f64,
                phi_dot: f32::from_le_bytes(buf[8..12].try_into().unwrap()) as f64,
                theta_dot: f32::from_le_bytes(buf[12..16].try_into().unwrap()) as f64,
                u: f32::from_le_bytes(buf[16..20].try_into().unwrap()) as f64,
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

pub fn run_online_mode() -> Result<(), Box<dyn Error>> {
    let mut i2c = I2c::new().map_err(|e| format!("Failed to init I2C: {}", e))?;
    i2c.set_slave_address(0x08)
        .map_err(|e| format!("Failed to set I2C address: {}", e))?;
    let i2c_bus = Arc::new(Mutex::new(i2c));

    let initial_k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(&ANALYTIC_LQR_POLICY);
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

pub fn run_data_collection_mode() -> Result<(), Box<dyn Error>> {
    let mut i2c = I2c::new().map_err(|e| format!("Failed to init I2C: {}", e))?;
    i2c.set_slave_address(0x08)
        .map_err(|e| format!("Failed to set I2C address: {}", e))?;
    let i2c_bus = Arc::new(Mutex::new(i2c));

    // SMART INDEX: Start from the highest existing file + 1
    let mut file_index = get_next_file_index();
    let mut was_balancing = false;
    let data_dir = "collected_data";
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

        let filename = format!("{}/batch_{}.csv", data_dir, file_index);
        let mut file = File::create(&filename)?;

        writeln!(file, "phi,phi_dot,theta,theta_dot,u")?;
        for s in &batch_to_process {
            writeln!(
                file,
                "{},{},{},{},{}",
                s.phi, s.theta, s.phi_dot, s.theta_dot, s.u
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
