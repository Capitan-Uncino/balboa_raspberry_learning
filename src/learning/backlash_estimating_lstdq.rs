use nalgebra::{DMatrix, DVector, SMatrix, SVector};

// Added a 0.0 at the end for the new backlash state
pub const ANALYTIC_LQR_POLICY: [f64; 5] = [0.18257419, 4.41295298, 0.098522314, 0.44153694, 0.0];

pub const DT: f64 = 0.01;

// --- System Dimensions ---
pub const DIM_X: usize = 5; // Updated to 5 to include backlash
pub const DIM_U: usize = 1;
const DIM_X_AND_U: usize = DIM_X + DIM_U; // 6
const DIM_PARAMS: usize = (DIM_X_AND_U * (DIM_X_AND_U + 1)) / 2; // (6 * 7) / 2 = 21

// --- LSPI Hyperparameters ---
const GAMMA: f64 = 0.99; // Discount factor
pub const SAMPLES_PER_ITER: usize = 100000; // Samples per policy evaluation
const LAMBDA_REG: f64 = 1e-5; // L2 Regularization

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

// 5-State struct for LSPI
#[derive(Debug, Clone, Copy)]
pub struct StateAction {
    pub phi: f64,
    pub theta: f64,
    pub phi_dot: f64,
    pub theta_dot: f64,
    pub backlash: f64,
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

// Updated signature to take slice &[StateAction] instead of Vec
fn run_lstdq(batch: &[StateAction], k: &SMatrix<f64, DIM_U, DIM_X>) -> SVector<f64, DIM_PARAMS> {
    let mut a_mat = DMatrix::<f64>::zeros(DIM_PARAMS, DIM_PARAMS);
    let mut b_vec = DVector::<f64>::zeros(DIM_PARAMS);

    // 5x5 Q-Matrix. Backlash penalty is strictly 0.0
    let q_cost = SMatrix::<f64, DIM_X, DIM_X>::from_diagonal(&SVector::from([
        10.0,  // phi penalty
        100.0, // theta penalty
        1.0,   // phi_dot penalty
        10.0,  // theta_dot penalty
        0.0,   // backlash penalty MUST remain 0
    ]));
    let r_cost = SMatrix::<f64, DIM_U, DIM_U>::from_diagonal(&SVector::from([30.0]));

    let mut skipped_couples = 0;
    let state_jump_threshold = 2.0;

    for i in 0..(batch.len() - 1) {
        let current = &batch[i];
        let next = &batch[i + 1];

        let x = SVector::<f64, DIM_X>::from_column_slice(&[
            current.phi,
            current.theta,
            current.phi_dot,
            current.theta_dot,
            current.backlash,
        ]);

        let x_next = SVector::<f64, DIM_X>::from_column_slice(&[
            next.phi,
            next.theta,
            next.phi_dot,
            next.theta_dot,
            next.backlash,
        ]);

        let state_diff_norm = (x - x_next).norm();
        if state_diff_norm > state_jump_threshold {
            skipped_couples += 1;
            continue;
        }

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
    }

    let total_couples = batch.len().saturating_sub(1);
    println!(
        "LSTDQ Batch Processing: Skipped {} / {} transitions due to discontinuity.",
        skipped_couples, total_couples
    );

    // Apply L2 Regularization
    for i in 0..DIM_PARAMS {
        a_mat[(i, i)] += LAMBDA_REG;
    }

    // Print Matrix A Diagnostics
    let eigvals = a_mat.complex_eigenvalues();
    let mut max_eig = 0.0_f64;
    let mut min_eig = f64::MAX;

    for c in eigvals.iter() {
        let norm = c.norm();
        if norm > max_eig {
            max_eig = norm;
        }
        if norm < min_eig {
            min_eig = norm;
        }
    }

    let svd = a_mat.clone().svd(false, false);
    let cond_num = if svd.singular_values.len() > 0 {
        let max_sv = svd.singular_values[0];
        let min_sv = svd.singular_values[svd.singular_values.len() - 1];
        max_sv / min_sv
    } else {
        f64::NAN
    };

    println!(
        "Matrix A Diagnostics | Cond Num: {:.4e} | Min Eig: {:.4e} | Max Eig: {:.4e}",
        cond_num, min_eig, max_eig
    );

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
    // 1.5 degrees in radians is ~0.026. Multiplying by 40 brings the
    // numerical range to ~1.0, matching the scale of phi and theta.
    const BACKLASH_SCALE: f64 = 4.0;

    // 1. Scale the dataset for the LSPI solver
    let mut scaled_batch = batch.to_vec();
    for state in scaled_batch.iter_mut() {
        state.backlash *= BACKLASH_SCALE;
    }

    // 2. Scale the input K matrix (K_scaled = K_phys * S^-1)
    let mut scaled_k = *current_k;
    scaled_k[(0, 4)] /= BACKLASH_SCALE;

    // 3. Run LSPI in the well-conditioned scaled space
    let theta = run_lstdq(&scaled_batch, &scaled_k);
    let h_mat = theta_to_h(&theta);
    let new_scaled_k = compute_k_from_h(&h_mat);

    // 4. Un-scale the resulting K matrix back to physical units (K_phys = K_scaled * S)
    // This allows your robot to use the matrix directly with raw sensor data!
    let mut new_k = new_scaled_k;
    new_k[(0, 4)] *= BACKLASH_SCALE;

    // Print the physical policy
    println!("Calculated Policy K (Physical Units): {}", new_k);

    new_k
}
