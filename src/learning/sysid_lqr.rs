use crate::learning::policy::Policy;
use nalgebra::{DMatrix, SMatrix, SVector};

//pub const ANALYTIC_LQR_POLICY: [f64; 4] = [1.3665, 15.4366, 0.4062, 1.3743];
pub const ANALYTIC_LQR_POLICY: [f64; 4] = [0.5196, 8.3716, 0.3161, 0.5893];
pub const DT: f64 = 0.01;

pub const Q_COST: [f64; 4] = [10.0, 100.0, 0.0, 0.1];
pub const R_COST: [f64; 1] = [3.0];

// --- System Dimensions ---
pub const DIM_X: usize = 4;
pub const DIM_U: usize = 1;
const DIM_X_AND_U: usize = DIM_X + DIM_U;
pub const DEBUG: bool = true;
pub const SAMPLES_PER_ITER: usize = 10000;

fn spectral_radius(
    a_mat: &SMatrix<f64, DIM_X, DIM_X>,
    b_mat: &SMatrix<f64, DIM_X, DIM_U>,
    k: &SMatrix<f64, DIM_U, DIM_X>,
) -> f64 {
    // Under u = Kx convention, closed loop is A + BK
    let a_cl = a_mat + b_mat * k;
    let eig = a_cl.complex_eigenvalues();
    eig.iter()
        .map(|c| c.norm())
        .fold(0.0, |a, b| f64::max(a, b))
}

#[derive(Debug, Clone, Copy)]
pub struct StateAction {
    pub phi: f64,
    pub theta: f64,
    pub phi_dot: f64,
    pub theta_dot: f64,
    pub u: f64,
}

// =====================================================================
// Helper 1: Closed-Loop System Identification
// =====================================================================
pub fn estimate_system_dynamics_closed_loop(
    batch: &[StateAction],
    current_k: &SMatrix<f64, DIM_U, DIM_X>,
) -> Option<(SMatrix<f64, DIM_X, DIM_X>, SMatrix<f64, DIM_X, DIM_U>)> {
    let n = batch.len();
    if n < 2 {
        return None;
    }

    let mut x_curr = DMatrix::<f64>::zeros(DIM_X_AND_U, n - 1);
    let mut x_next = DMatrix::<f64>::zeros(DIM_X, n - 1);

    for i in 0..(n - 1) {
        let t_curr = &batch[i];
        let t_next = &batch[i + 1];

        // 1. Extract the state vector
        let state_vec =
            SVector::<f64, DIM_X>::new(t_curr.phi, t_curr.theta, t_curr.phi_dot, t_curr.theta_dot);

        // 2. Reconstruct pure exploration noise (u = Kx convention)
        // u_total = K * x + noise  =>  noise = u_total - K * x
        let u_control = (current_k * state_vec)[0];
        let pure_noise = t_curr.u - u_control;

        // 3. Populate X_curr using [x_curr^T, pure_noise^T]^T
        x_curr[(0, i)] = t_curr.phi;
        x_curr[(1, i)] = t_curr.theta;
        x_curr[(2, i)] = t_curr.phi_dot;
        x_curr[(3, i)] = t_curr.theta_dot;
        x_curr[(4, i)] = pure_noise;

        // X_next = x(t+1)
        x_next[(0, i)] = t_next.phi;
        x_next[(1, i)] = t_next.theta;
        x_next[(2, i)] = t_next.phi_dot;
        x_next[(3, i)] = t_next.theta_dot;
    }

    // 4. Solve for Theta = [A_cl  B] using Normal Equations instead of SVD
    // This avoids a 50,000 x 50,000 memory allocation!
    let x_curr_t = x_curr.transpose(); // Size: N x 5
    let x_next_t = x_next.transpose(); // Size: N x 4

    // (X^T * X) is a tiny 5x5 matrix. (X^T * Y) is a 5x4 matrix.
    let xt_x = &x_curr * &x_curr_t;
    let xt_y = &x_curr * &x_next_t;

    let theta_t = xt_x
        .try_inverse()
        .expect("System ID failed: Data matrix singular (Add more exploration noise)")
        * xt_y;

    let theta = theta_t.transpose();

    // 5. Extract A_cl and B
    let mut a_cl = SMatrix::<f64, DIM_X, DIM_X>::zeros();
    let mut b_mat = SMatrix::<f64, DIM_X, DIM_U>::zeros();

    for r in 0..DIM_X {
        for c in 0..DIM_X {
            a_cl[(r, c)] = theta[(r, c)];
        }
        for c in 0..DIM_U {
            b_mat[(r, c)] = theta[(r, DIM_X + c)];
        }
    }

    // 6. Recover the true open-loop physical A matrix
    // Since u = Kx convention, A_cl = A + BK  =>  A = A_cl - BK
    let a_mat = a_cl - b_mat * current_k;

    if DEBUG {
        println!("=== Identified True Open-Loop Dynamics ===");
        println!("A Matrix:\n{:.4}", a_mat);
        println!("B Matrix:\n{:.4}", b_mat);
        println!("==========================================");
    }

    Some((a_mat, b_mat))
}

// =====================================================================
// Helper 2: LQR Calculation
// =====================================================================
pub fn compute_lqr_gain(
    a_mat: &SMatrix<f64, DIM_X, DIM_X>,
    b_mat: &SMatrix<f64, DIM_X, DIM_U>,
) -> SMatrix<f64, DIM_U, DIM_X> {
    let q_mat = SMatrix::<f64, DIM_X, DIM_X>::from_diagonal(
        &SVector::<f64, DIM_X>::from_row_slice(&Q_COST),
    );
    let r_mat = SMatrix::<f64, DIM_U, DIM_U>::from_diagonal(
        &SVector::<f64, DIM_U>::from_row_slice(&R_COST),
    );

    let mut p_mat = q_mat;
    let max_iter = 1000;
    let tolerance = 1e-9; // Tightened for high-precision inverted pendulum stability

    for _ in 0..max_iter {
        let b_t_p = b_mat.transpose() * p_mat;
        let inv_term = (r_mat + b_t_p * b_mat)
            .try_inverse()
            .expect("LQR: Matrix inversion failed (Check R cost)");

        let next_p = q_mat + a_mat.transpose() * p_mat * a_mat
            - a_mat.transpose() * p_mat * b_mat * inv_term * b_mat.transpose() * p_mat * a_mat;

        let diff = (next_p - p_mat).norm();
        p_mat = next_p;

        if diff < tolerance {
            break;
        }
    }

    // Compute standard optimal K: K_opt = (R + B^T P B)^-1 B^T P A
    let inv_term = (r_mat + b_mat.transpose() * p_mat * b_mat)
        .try_inverse()
        .unwrap();

    let k_opt = inv_term * b_mat.transpose() * p_mat * a_mat;

    // Because your system specifically applies u = Kx (positive feedback),
    // we must return -K_opt so the resulting control law is actively stabilizing.
    -k_opt
}

// =====================================================================
// Main Policy Update Function
// =====================================================================

pub fn get_policy(batch: &[StateAction], current_policy: &Policy) -> Policy {
    // 1. Extract the explicit gains array, panicking if it's a non-linear policy
    let gains_array = current_policy
        .get_gains()
        .expect("Input policy must be linear (explicit gains must be Some)");

    // Reconstruct the SMatrix for mathematical operations
    let current_k = nalgebra::SMatrix::<f64, DIM_U, DIM_X>::from_row_slice(&gains_array);

    // 2. Determine the final K matrix
    let k_final = match estimate_system_dynamics_closed_loop(batch, &current_k) {
        Some((a_mat, b_mat)) => {
            // Compute Greedy LQR Policy
            let k_greedy = compute_lqr_gain(&a_mat, &b_mat);

            // Apply Polyak Averaging (Policy-Space Trust Region)
            let alpha = 1.0;
            let k_trust = current_k * (1.0 - alpha) + k_greedy * alpha;

            println!(">>> Trust Region Applied: alpha = {}", alpha);
            println!(
                ">>> Spectral Radius: {:.4}",
                spectral_radius(&a_mat, &b_mat, &k_trust)
            );

            k_trust
        }
        None => {
            eprintln!("Batch too small for System ID. Returning current K.");
            current_k // Copy the current K to be used as k_final
        }
    };

    // 3. Extract the resulting matrix back into an array to store in the Policy struct
    let explicit_array = [
        k_final[(0, 0)],
        k_final[(0, 1)],
        k_final[(0, 2)],
        k_final[(0, 3)],
    ];

    // 4. Return the encapsulated Policy
    Policy::new(move |x| (k_final * x)[0], Some(explicit_array))
}
