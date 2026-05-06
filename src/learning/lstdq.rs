use nalgebra::{DMatrix, DVector, SMatrix, SVector};

pub const ANALYTIC_LQR_POLICY: [f64; 4] = [0.182574, 4.412952, 0.098523, 0.441536];

pub const DT: f64 = 0.01;

// --- System Dimensions ---
pub const DIM_X: usize = 4; // [theta, theta_dot]
pub const DIM_U: usize = 1; // [torque]
const DIM_X_AND_U: usize = DIM_X + DIM_U;
const DIM_PARAMS: usize = (DIM_X_AND_U * (DIM_X_AND_U + 1)) / 2;

// --- LSPI Hyperparameters ---
const GAMMA: f64 = 0.99; // Discount factor
pub const SAMPLES_PER_ITER: usize = 10000; // Samples per policy evaluation
const LAMBDA_REG: f64 = 1e-5; // L2 Regularization
const ETA: f64 = 0.0f64;

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
            current.phi,
            current.theta,
            current.phi_dot,
            current.theta_dot,
        ]);
        let u = SVector::<f64, DIM_U>::from_column_slice(&[current.u]);

        let x_next = SVector::<f64, DIM_X>::from_column_slice(&[
            next.phi,
            next.theta,
            next.phi_dot,
            next.theta_dot,
        ]);

        // Cost calculation using our local Q and R
        let cost = x.dot(&(q_cost * x)) + u.dot(&(r_cost * u));

        // Feature computation with BIAS using the NOISY action applied
        let phi = get_quadratic_features(&x, &u) + bias_vec;

        // Target policy uses the current gain matrix K (Greedy action)
        let u_next_greedy = k * x_next;
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
pub fn calculate_k(
    batch: Vec<StateAction>,
    current_k: &SMatrix<f64, DIM_U, DIM_X>,
) -> SMatrix<f64, DIM_U, DIM_X> {
    let theta = run_lstdq(batch, current_k);
    let h_mat = theta_to_h(&theta);
    compute_k_from_h(&h_mat)
}
