use nalgebra::{DMatrix, DVector, SMatrix, SVector};
pub const ANALYTIC_LQR_POLICY: [f64; 4] = [0.18257419, 4.41295298, 0.098522314, 0.44153694];

pub const DT: f64 = 0.01;

// --- System Dimensions ---
pub const DIM_X: usize = 4;
pub const DIM_U: usize = 1;
const DIM_X_AND_U: usize = DIM_X + DIM_U;
const DIM_PARAMS: usize = (DIM_X_AND_U * (DIM_X_AND_U + 1)) / 2;

// --- LSPI Hyperparameters ---
const GAMMA: f64 = 1.00; // Discount factor 0.99
pub const SAMPLES_PER_ITER: usize = 100000; // Samples per policy evaluation
const LAMBDA_REG: f64 = 1e-5; // L2 Regularization

//
//
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

fn run_lstdq(batch: Vec<StateAction>, k: &SMatrix<f64, DIM_U, DIM_X>) -> SVector<f64, DIM_PARAMS> {
    let mut a_mat = DMatrix::<f64>::zeros(DIM_PARAMS, DIM_PARAMS);
    let mut b_vec = DVector::<f64>::zeros(DIM_PARAMS);

    let q_cost = SMatrix::<f64, DIM_X, DIM_X>::from_diagonal(&SVector::from([
        10.0,  // phi penalty
        100.0, // theta penalty
        1.0,   // phi_dot penalty
        10.0,  // theta_dot penalty
    ]));
    let r_cost = SMatrix::<f64, DIM_U, DIM_U>::from_diagonal(&SVector::from([300.0]));

    let mut skipped_couples = 0;
    let state_jump_threshold = 2.0;

    // 2. Iterate directly over the raw chronological batch
    for i in 0..(batch.len() - 1) {
        let current = &batch[i];
        let next = &batch[i + 1];

        // Construct mathematical vectors directly from the I2C structs FIRST
        let x = SVector::<f64, DIM_X>::from_column_slice(&[
            current.phi,
            current.theta,
            current.phi_dot,
            current.theta_dot,
        ]);

        let x_next = SVector::<f64, DIM_X>::from_column_slice(&[
            next.phi,
            next.theta,
            next.phi_dot,
            next.theta_dot,
        ]);

        // --- NEW: Discontinuity check ---
        // If the state changes too drastically in one timestep, it implies a reset/fall.
        let state_diff_norm = (x - x_next).norm();
        if state_diff_norm > state_jump_threshold {
            skipped_couples += 1;
            continue; // Skip this transition entirely
        }

        let u = SVector::<f64, DIM_U>::from_column_slice(&[current.u]);

        // ct := xt^T S xt + ut^T R ut
        let cost = x.dot(&(q_cost * x)) + u.dot(&(r_cost * u));

        // phi_t := phi(xt, ut)
        let phi_t = get_quadratic_features(&x, &u);

        // psi_{t+1} := phi(x_{next}, Keval * x_{next})
        let u_next_greedy = k * x_next;
        let psi_t_plus_1 = get_quadratic_features(&x_next, &u_next_greedy);

        // LSTDQ Update mapping to: phi_t * (phi_t - psi_{t+1} + f)^T
        //
        let temporal_diff = phi_t - (GAMMA * psi_t_plus_1);

        for r in 0..DIM_PARAMS {
            let phi_r = phi_t[r];
            b_vec[r] += phi_r * cost;
            for c in 0..DIM_PARAMS {
                a_mat[(r, c)] += phi_r * temporal_diff[c];
            }
        }
    }

    // --- NEW: Print the results ---
    let total_couples = batch.len().saturating_sub(1);
    println!(
        "LSTDQ Batch Processing: Skipped {} / {} transitions due to discontinuity.",
        skipped_couples, total_couples
    );

    // Apply L2 Regularization
    for i in 0..DIM_PARAMS {
        a_mat[(i, i)] += LAMBDA_REG;
    }

    // Solve for q-function parameters
    let q_dyn = a_mat
        .lu()
        .solve(&b_vec)
        .unwrap_or(DVector::zeros(DIM_PARAMS));

    let mut q_params = SVector::<f64, DIM_PARAMS>::zeros();
    q_params.copy_from_slice(q_dyn.as_slice());

    q_params
}

pub fn calculate_k(
    batch: Vec<StateAction>,
    current_k: &SMatrix<f64, DIM_U, DIM_X>,
) -> SMatrix<f64, DIM_U, DIM_X> {
    let theta = run_lstdq(batch, current_k);
    let h_mat = theta_to_h(&theta);
    compute_k_from_h(&h_mat)
}
