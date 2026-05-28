use nalgebra::{DMatrix, DVector, SMatrix, SVector};

pub const ANALYTIC_LQR_POLICY: [f64; 4] = [1.3665, 15.4366, 0.4062, 1.3743];
pub const DT: f64 = 0.01;

// --- OPTIONAL FEATURE CONSTANTS ---
pub const STANDARDIZATION: bool = true;
pub const OUTLIERS_REMOVAL: bool = true;
pub const LS_REGULARIZATION: bool = true;

// --- System Dimensions ---
pub const DIM_X: usize = 4;
pub const DIM_U: usize = 1;
const DIM_X_AND_U: usize = DIM_X + DIM_U;
const DIM_PARAMS: usize = (DIM_X_AND_U * (DIM_X_AND_U + 1)) / 2;

// --- LSPI Hyperparameters ---
const GAMMA: f64 = 1.00; // Discount factor
pub const SAMPLES_PER_ITER: usize = 100000; // Samples per policy evaluation
const LAMBDA_REG: f64 = 1e-5; // Regularization
const LAMBDA_TD: f64 = 0.40; // Trace decay factor

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

/// Helper function to compute standardization and outlier statistics efficiently
fn compute_feature_stats(
    batch: &[StateAction],
    exclude_outliers: bool,
    raw_mean: Option<&SVector<f64, DIM_PARAMS>>,
    raw_std: Option<&SVector<f64, DIM_PARAMS>>,
) -> (SVector<f64, DIM_PARAMS>, SVector<f64, DIM_PARAMS>) {
    let mut mean = SVector::<f64, DIM_PARAMS>::zeros();
    let mut sq_mean = SVector::<f64, DIM_PARAMS>::zeros();
    let mut count = 0.0;

    for current in batch.iter() {
        let x = SVector::<f64, DIM_X>::from_column_slice(&[
            current.phi,
            current.theta,
            current.phi_dot,
            current.theta_dot,
        ]);
        let u = SVector::<f64, DIM_U>::from_column_slice(&[current.u]);
        let phi_t = get_quadratic_features(&x, &u);

        if exclude_outliers {
            if let (Some(rm), Some(rs)) = (raw_mean, raw_std) {
                let mut is_outlier = false;
                for i in 0..DIM_PARAMS {
                    if (phi_t[i] - rm[i]).abs() > 3.0 * rs[i] {
                        is_outlier = true;
                        break;
                    }
                }
                if is_outlier {
                    continue;
                }
            }
        }

        mean += phi_t;
        for i in 0..DIM_PARAMS {
            sq_mean[i] += phi_t[i] * phi_t[i];
        }
        count += 1.0;
    }

    if count > 0.0 {
        mean /= count;
        sq_mean /= count;
    }

    let mut std = SVector::<f64, DIM_PARAMS>::zeros();
    for i in 0..DIM_PARAMS {
        let var = sq_mean[i] - mean[i] * mean[i];
        std[i] = if var > 1e-8 { var.sqrt() } else { 1.0 };
    }

    (mean, std)
}

fn run_lstdq(batch: &[StateAction], k: &SMatrix<f64, DIM_U, DIM_X>) -> SVector<f64, DIM_PARAMS> {
    let mut a_mat = DMatrix::<f64>::zeros(DIM_PARAMS, DIM_PARAMS);
    let mut b_vec = DVector::<f64>::zeros(DIM_PARAMS);

    let q_cost =
        SMatrix::<f64, DIM_X, DIM_X>::from_diagonal(&SVector::from([10.0, 100.0, 0.0, 0.1]));
    let r_cost = SMatrix::<f64, DIM_U, DIM_U>::from_diagonal(&SVector::from([3.0]));

    let mut skipped_couples = 0;
    let state_jump_threshold = 2.0;

    // --- STEP 1: Compute Feature Statistics ---
    let (raw_mean, raw_std) = if OUTLIERS_REMOVAL || STANDARDIZATION {
        compute_feature_stats(batch, false, None, None)
    } else {
        (SVector::zeros(), SVector::from_element(1.0))
    };

    // To be strictly correct, our standardization factors must NOT be skewed by outliers.
    let (_clean_mean, clean_std) = if STANDARDIZATION && OUTLIERS_REMOVAL {
        compute_feature_stats(batch, true, Some(&raw_mean), Some(&raw_std))
    } else {
        (raw_mean.clone(), raw_std.clone())
    };

    let mut z_trace = SVector::<f64, DIM_PARAMS>::zeros();

    for i in 0..(batch.len() - 1) {
        let current = &batch[i];
        let next = &batch[i + 1];

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

        let state_diff_norm = (x - x_next).norm();
        if state_diff_norm > state_jump_threshold {
            skipped_couples += 1;
            z_trace.fill(0.0);
            continue;
        }

        let u = SVector::<f64, DIM_U>::from_column_slice(&[current.u]);
        let phi_t = get_quadratic_features(&x, &u);

        // --- STEP 2: Outlier Removal ---
        if OUTLIERS_REMOVAL {
            let mut is_outlier = false;
            for r in 0..DIM_PARAMS {
                // 3-sigma rule based on raw batch distributions
                if (phi_t[r] - raw_mean[r]).abs() > 3.0 * raw_std[r] {
                    is_outlier = true;
                    break;
                }
            }
            if is_outlier {
                skipped_couples += 1;
                z_trace.fill(0.0); // Crucial: clear memory so chronological chain resets
                continue;
            }
        }

        let cost = x.dot(&(q_cost * x)) + u.dot(&(r_cost * u));

        let u_next_greedy = k * x_next;
        let psi_t_plus_1 = get_quadratic_features(&x_next, &u_next_greedy);

        let mut phi_t_model = phi_t;
        let mut psi_t_plus_1_model = psi_t_plus_1;

        // --- STEP 3: Standardization ---
        if STANDARDIZATION {
            for r in 0..DIM_PARAMS {
                // We divide by std deviations but DO NOT mean shift.
                // A pure quadratic form requires the origin (0,0) to evaluate to 0.
                phi_t_model[r] /= clean_std[r];
                psi_t_plus_1_model[r] /= clean_std[r];
            }
        }

        let temporal_diff = phi_t_model - (GAMMA * psi_t_plus_1_model);
        z_trace = z_trace * (GAMMA * LAMBDA_TD) + phi_t_model;

        for r in 0..DIM_PARAMS {
            let z_r = z_trace[r];
            b_vec[r] += z_r * cost;
            for c in 0..DIM_PARAMS {
                a_mat[(r, c)] += z_r * temporal_diff[c];
            }
        }
    }

    let total_couples = batch.len().saturating_sub(1);
    println!(
        "LSTDQ(lambda) Batch Processing: Skipped {} / {} transitions.",
        skipped_couples, total_couples
    );

    // --- STEP 4: Least Squares Regularization ---
    if LS_REGULARIZATION {
        for i in 0..DIM_PARAMS {
            a_mat[(i, i)] += LAMBDA_REG;
        }
    }

    let q_dyn = a_mat
        .lu()
        .solve(&b_vec)
        .unwrap_or(DVector::zeros(DIM_PARAMS));

    let mut q_params = SVector::<f64, DIM_PARAMS>::zeros();

    // --- STEP 5: Reverse Standardization (Un-scaling) ---
    for i in 0..DIM_PARAMS {
        if STANDARDIZATION {
            // Because we solved for weights using scaled inputs, we must divide
            // the resulting weight by the scaling factor to return it to physical coordinates
            q_params[i] = q_dyn[i] / clean_std[i];
        } else {
            q_params[i] = q_dyn[i];
        }
    }

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
