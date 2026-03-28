use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use rppal::i2c::I2c;
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

/// Helper function to pack 4 floats and write them to I2C offset 20
fn write_gains(i2c_bus: &Arc<Mutex<I2c>>, gains: [f32; 4]) {
    let mut write_buf = Vec::with_capacity(16);

    for k in gains {
        write_buf.extend_from_slice(&k.to_le_bytes());
    }

    if let Ok(mut bus) = i2c_bus.lock() {
        // CRITICAL UPDATE: Write to offset 20, because 0-19 is taken by the 5 read floats
        if let Err(e) = bus.block_write(20, &write_buf) {
            eprintln!("I2C Write Error: {:?}", e);
        }
    } else {
        eprintln!("Failed to acquire I2C lock for writing.");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize the I2C bus and set the slave address
    let mut i2c = I2c::with_bus(1)?;
    i2c.set_slave_address(BALBOA_ADDR)?;

    let i2c_bus = Arc::new(Mutex::new(i2c));

    let initial_k_mat = SMatrix::<f64, 1, 4>::from_row_slice(&[100.0, 5.0, 2.0, 0.5]);
    let current_k = Arc::new(Mutex::new(initial_k_mat));

    write_gains(&i2c_bus, [100.0f32, 5.0, 2.0, 0.5]);

    let mut state_batch: Vec<StateAction> = Vec::with_capacity(SAMPLES_PER_ITER);

    println!("Starting 100Hz control loop...");

    loop {
        let start_time = Instant::now();

        let mut buf = [0u8; 20];
        let read_result = {
            let mut bus = i2c_bus.lock().unwrap();
            bus.block_read(0, &mut buf)
        };

        if read_result.is_ok() {
            let data = StateAction {
                phi: f32::from_le_bytes(buf[0..4].try_into().unwrap()),
                phi_dot: f32::from_le_bytes(buf[4..8].try_into().unwrap()),
                theta: f32::from_le_bytes(buf[8..12].try_into().unwrap()),
                theta_dot: f32::from_le_bytes(buf[12..16].try_into().unwrap()),
                u: f32::from_le_bytes(buf[16..20].try_into().unwrap()),
            };

            state_batch.push(data);
        } else {
            eprintln!("I2C Read Error");
        }

        // 3. Check if the batch is full
        if state_batch.len() >= SAMPLES_PER_ITER {
            // Efficiently swap out the full batch for an empty one
            let batch_to_process =
                std::mem::replace(&mut state_batch, Vec::with_capacity(SAMPLES_PER_ITER));

            // Clone the Arcs for the background thread
            let i2c_clone = Arc::clone(&i2c_bus);
            let k_clone = Arc::clone(&current_k);

            // Spawn a detached thread to handle the math and writing
            thread::spawn(move || {
                // a. Safely read the current K matrix
                // We copy it into a local variable so we don't hold the lock during heavy math
                let k_to_use = { *k_clone.lock().unwrap() };

                let new_k_mat = calculate_k(batch_to_process, &k_to_use);

                {
                    *k_clone.lock().unwrap() = new_k_mat;
                }

                // d. Convert the new SMatrix down to an f32 array for the Arduino
                let new_k_array = [
                    new_k_mat[(0, 0)] as f32,
                    new_k_mat[(0, 1)] as f32,
                    new_k_mat[(0, 2)] as f32,
                    new_k_mat[(0, 3)] as f32,
                ];

                // e. Push to Arduino
                write_gains(&i2c_clone, new_k_array);
                println!("Pushed new K vector to Arduino.");
            });
        }

        // 4. Maintain a strict 100Hz (10ms) polling rate
        let elapsed = start_time.elapsed();
        let target_duration = Duration::from_millis(10);

        if elapsed < target_duration {
            thread::sleep(target_duration - elapsed);
        }
    }
}
