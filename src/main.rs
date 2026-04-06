use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use serialport::SerialPort;
use std::convert::TryInto;
use std::io::{Read, Write};

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const BALBOA_ADDR: u16 = 20;

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
const ETA: f64 = 0.1; // Parameter eta from Algorithm 2 (Tune this!)

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

fn write_gains(serial_bus: &Arc<Mutex<Box<dyn SerialPort>>>, gains: [f32; 4]) {
    let mut write_buf = Vec::with_capacity(16);

    // Pack the 4 floats into little-endian bytes
    for k in gains {
        write_buf.extend_from_slice(&k.to_le_bytes());
    }

    if let Ok(mut port) = serial_bus.lock() {
        // Console feedback so you know exactly what LSTDQ is deciding
        println!(
            "---> Sending Gains: [k1: {:.3}, k2: {:.3}, k3: {:.3}, k4: {:.3}]",
            gains[0], gains[1], gains[2], gains[3]
        );

        if let Err(e) = port.write_all(&write_buf) {
            eprintln!("UART Write Error: {:?}", e);
        } else {
            // Force the OS to push the bytes out the TX pin right now
            let _ = port.flush();
        }
    } else {
        eprintln!("Failed to acquire UART lock for writing.");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("    BALBOA BRAIN v2.0 - UART ENABLED    ");
    println!("========================================");

    let port_name = "/dev/ttyS0";
    let baud_rate = 115200;

    let port = serialport::new(port_name, baud_rate)
        .timeout(Duration::from_millis(100))
        .open()?;

    let serial_bus = Arc::new(Mutex::new(port));

    let initial_k_mat = SMatrix::<f64, 1, 4>::from_row_slice(&[100.0, 5.0, 2.0, 0.5]);
    let current_k = Arc::new(Mutex::new(initial_k_mat));

    write_gains(&serial_bus, [100.0f32, 5.0, 2.0, 0.5]);

    let mut state_batch: Vec<StateAction> = Vec::with_capacity(SAMPLES_PER_ITER);

    let mut loop_counter = 0;
    let mut is_connected = true;
    let wake_up_time = Instant::now();

    println!("Starting 100Hz control loop on {}...", port_name);

    loop {
        let start_time = Instant::now();
        let mut buf = [0u8; 20];

        let read_result = {
            let mut bus = serial_bus.lock().unwrap();
            bus.read_exact(&mut buf)
        };

        if read_result.is_ok() {
            if !is_connected {
                println!(
                    "[+] CONNECTION RESTORED: Resuming control loop at {}.",
                    wake_up_time.elapsed().as_secs_f32()
                );
                is_connected = true;
            }
            let data = StateAction {
                phi: f32::from_le_bytes(buf[0..4].try_into().unwrap()),
                phi_dot: f32::from_le_bytes(buf[4..8].try_into().unwrap()),
                theta: f32::from_le_bytes(buf[8..12].try_into().unwrap()),
                theta_dot: f32::from_le_bytes(buf[12..16].try_into().unwrap()),
                u: f32::from_le_bytes(buf[16..20].try_into().unwrap()),
            };

            // --- Once per second feedback ---
            loop_counter += 1;
            if loop_counter >= 100 {
                println!(
                    "[Live] Tilt: {:>6.2}° | Wheel: {:>6.2} | Motor: {:>6.2}",
                    data.phi.to_degrees(),
                    data.theta,
                    data.u
                );
                loop_counter = 0;
            }

            state_batch.push(data);
        } else {
            if is_connected {
                eprintln!(
                    "[!] CONNECTION LOST: Watchdog timeout (100ms) at {}.",
                    wake_up_time.elapsed().as_secs_f32()
                );
                is_connected = false;
            }
        }

        // 3. Check if the batch is full
        if state_batch.len() >= SAMPLES_PER_ITER {
            let batch_to_process =
                std::mem::replace(&mut state_batch, Vec::with_capacity(SAMPLES_PER_ITER));

            let serial_clone = Arc::clone(&serial_bus);
            let k_clone = Arc::clone(&current_k);

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

                write_gains(&serial_clone, new_k_array);
                println!(">>> LSTDQ Update: Pushed new K vector to Arduino.");
            });
        }
    }
}
