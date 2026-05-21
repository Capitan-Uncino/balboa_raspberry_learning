use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use std::f64::consts::PI;

// --- PUBLIC CONSTANTS (Fully 5-State API) ---
// Added 0.0 for the 5th state initialization
pub const ANALYTIC_LQR_POLICY: [f64; 5] = [0.18257419, 4.41295298, 0.098522314, 0.44153694, 0.0];
pub const DT: f64 = 0.01;

// --- Backlash Constants ---
pub const TOTAL_BACKLASH: f64 = 3.0 * PI / 180.0;
pub const DEADZONE_EPSILON: f64 = TOTAL_BACKLASH / 2.0;

// --- System Dimensions ---
pub const DIM_X: usize = 5; // Policy is fully 5D
pub const DIM_U: usize = 1;
const DIM_X_AND_U: usize = DIM_X + DIM_U; // 6
const DIM_PARAMS: usize = (DIM_X_AND_U * (DIM_X_AND_U + 1)) / 2; // 21

// --- LSPI Hyperparameters ---
const GAMMA: f64 = 1.00;
pub const SAMPLES_PER_ITER: usize = 100000;
const LAMBDA_REG: f64 = 1e-5;

pub fn spectral_radius(
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

// --- Data Struct: Strictly 4 Physical States ---
// The batch relies entirely on raw sensor data.
#[derive(Debug, Clone, Copy)]
pub struct StateAction {
    pub phi: f64,
    pub theta: f64,
    pub phi_dot: f64,
    pub theta_dot: f64,
    pub u: f64,
}

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
        Some(inv) => -inv * q_ux,
        None => SMatrix::<f64, DIM_U, DIM_X>::identity(),
    }
}

pub fn run_lstdq(
    batch: &[StateAction],
    k: &SMatrix<f64, DIM_U, DIM_X>,
) -> SVector<f64, DIM_PARAMS> {
    let mut a_mat = DMatrix::<f64>::zeros(DIM_PARAMS, DIM_PARAMS);
    let mut b_vec = DVector::<f64>::zeros(DIM_PARAMS);

    let q_cost = SMatrix::<f64, DIM_X, DIM_X>::from_diagonal(&SVector::from([
        10.0,  // phi penalty
        100.0, // theta penalty
        1.0,   // phi_dot penalty
        10.0,  // theta_dot penalty
        0.0,   // Backlash tracking penalty must be strictly 0
    ]));
    let r_cost = SMatrix::<f64, DIM_U, DIM_U>::from_diagonal(&SVector::from([300.0]));

    let mut skipped_couples = 0;
    let state_jump_threshold = 2.0;

    let mut current_b = 0.0;

    for i in 0..(batch.len() - 1) {
        let current = &batch[i];
        let next = &batch[i + 1];

        // Validate continuity against original 4 physical states
        let phys_current = SVector::<f64, 4>::from_column_slice(&[
            current.phi,
            current.theta,
            current.phi_dot,
            current.theta_dot,
        ]);
        let phys_next = SVector::<f64, 4>::from_column_slice(&[
            next.phi,
            next.theta,
            next.phi_dot,
            next.theta_dot,
        ]);

        if (phys_current - phys_next).norm() > state_jump_threshold {
            skipped_couples += 1;
            current_b = 0.0; // Reset estimator if discontinuity occurs
            continue;
        }

        // Hysteresis tracking: estimate the 5th state dynamically
        let delta_abs_motor = (next.phi - current.phi) + (next.theta - current.theta);
        let next_b = (current_b + delta_abs_motor).clamp(-DEADZONE_EPSILON, DEADZONE_EPSILON);

        // Construct full 5D state vectors
        let x = SVector::<f64, DIM_X>::from_column_slice(&[
            current.phi,
            current.theta,
            current.phi_dot,
            current.theta_dot,
            current_b,
        ]);

        let x_next = SVector::<f64, DIM_X>::from_column_slice(&[
            next.phi,
            next.theta,
            next.phi_dot,
            next.theta_dot,
            next_b,
        ]);

        let u = SVector::<f64, DIM_U>::from_column_slice(&[current.u]);
        let cost = x.dot(&(q_cost * x)) + u.dot(&(r_cost * u));
        let phi_t = get_quadratic_features(&x, &u);

        let u_next_greedy = k * x_next;
        let psi_t_plus_1 = get_quadratic_features(&x_next, &u_next_greedy);

        let temporal_diff = phi_t - (GAMMA * psi_t_plus_1);

        for r in 0..DIM_PARAMS {
            let phi_r = phi_t[r];
            b_vec[r] += phi_r * cost;
            for c in 0..DIM_PARAMS {
                a_mat[(r, c)] += phi_r * temporal_diff[c];
            }
        }

        current_b = next_b;
    }

    let total_couples = batch.len().saturating_sub(1);
    println!(
        "LSTDQ Batch Processing: Skipped {} / {} transitions due to discontinuity.",
        skipped_couples, total_couples
    );

    for i in 0..DIM_PARAMS {
        a_mat[(i, i)] += LAMBDA_REG;
    }

    let q_dyn = a_mat
        .lu()
        .solve(&b_vec)
        .unwrap_or(DVector::zeros(DIM_PARAMS));

    let mut q_params = SVector::<f64, DIM_PARAMS>::zeros();
    q_params.copy_from_slice(q_dyn.as_slice());

    q_params
}

pub fn calculate_k(
    batch: &[StateAction],
    current_k: &SMatrix<f64, DIM_U, DIM_X>,
) -> SMatrix<f64, DIM_U, DIM_X> {
    let theta = run_lstdq(batch, current_k);
    let h_mat = theta_to_h(&theta);
    compute_k_from_h(&h_mat)
}
