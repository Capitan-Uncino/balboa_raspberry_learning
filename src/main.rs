use nalgebra::{ComplexField, DMatrix, DVector, SMatrix, SVector};
use plotters::prelude::*;
use rand::prelude::*;
use rand::rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::error::Error;

// --- System Dimensions ---
const DIM_X: usize = 2; // [theta, theta_dot]
const DIM_U: usize = 1; // [torque]
const DIM_X_AND_U: usize = DIM_X + DIM_U;
const DIM_PARAMS: usize = (DIM_X_AND_U * (DIM_X_AND_U + 1)) / 2;

// --- LSPI Hyperparameters ---
const GAMMA: f64 = 0.99; // Discount factor
const SAMPLES_PER_ITER: usize = 10000; // Samples per policy evaluation
const MAX_ITERATIONS: usize = 15;
const LAMBDA_REG: f64 = 1e-5; // L2 Regularization
const DT: f64 = 0.05; // Time step for discretization

// --- Physics Constants (Pendulum) ---
const G: f64 = 9.81;
const L: f64 = 1.0;
const M: f64 = 1.0;

// --- System Definition ---
struct PendulumSystem {
    A: SMatrix<f64, DIM_X, DIM_X>,
    B: SMatrix<f64, DIM_X, DIM_U>,
    Q: SMatrix<f64, DIM_X, DIM_X>,
    R: SMatrix<f64, DIM_U, DIM_U>,
    resettable: bool,
}

impl PendulumSystem {
    fn new(resettable: bool) -> Self {
        // Continuous Time Dynamics (Linearized at theta=0)
        // x_dot = [0, 1; g/l, 0] * x + [0; 1/(ml^2)] * u
        // Discretized: x_{k+1} = (I + A*dt) * x_k + (B*dt) * u_k

        let a_cont = SMatrix::<f64, 2, 2>::new(0.0, 1.0, G / L, 0.0);
        let b_cont = SMatrix::<f64, 2, 1>::new(0.0, 1.0 / (M * L * L));

        // Simple Euler Discretization
        let A = SMatrix::<f64, 2, 2>::identity() + a_cont * DT;
        let B = b_cont * DT;

        // LQR Costs
        let Q = SMatrix::<f64, 2, 2>::from_diagonal(&SVector::from([1.0, 0.1])); // Penalty on angle
        let R = SMatrix::<f64, 1, 1>::from_diagonal(&SVector::from([0.1])); // Penalty on torque

        Self {
            A,
            B,
            Q,
            R,
            resettable,
        }
    }

    fn step(&self, x: &SVector<f64, DIM_X>, u: &SVector<f64, DIM_U>) -> SVector<f64, DIM_X> {
        self.A * x + self.B * u
    }
}

// --- Simulation State ---
struct Simulation {
    id: usize,
    label: String,
    K: SMatrix<f64, DIM_U, DIM_X>,
    diag_iterations: Vec<usize>,
    diag_policy_error: Vec<f64>,
    diag_spectral_radius: Vec<f64>,
}

impl Simulation {
    fn new(id: usize, label: String, K_init: SMatrix<f64, DIM_U, DIM_X>) -> Self {
        Self {
            id,
            label,
            K: K_init,
            diag_iterations: Vec::new(),
            diag_policy_error: Vec::new(),
            diag_spectral_radius: Vec::new(),
        }
    }
}

// --- Main Algorithm ---
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== LSPI Inverted Pendulum Experiment ===");
    let system = PendulumSystem::new(false);

    //Compute Analytic Ground Truth (LQR)
    let K_star = solve_dare_analytic(&system);
    let rho_star = spectral_radius(&system.A, &system.B, &K_star);
    println!("Optimal K: {:.4}", K_star);
    println!("Optimal Spectral Radius: {:.4}", rho_star);

    // Initialize Simulations
    let mut sims = Vec::new();
    let mut rng = rng();

    // STABLE Starts (PD Controller: K ~ [10, 3])
    // A rough PD controller stabilizes the pendulum
    for i in 0..5 {
        let noise_dist = Uniform::new(-5.0, 5.0).unwrap();
        let noise_vec = SMatrix::<f64, DIM_U, DIM_X>::from_fn(|_, _| noise_dist.sample(&mut rng));

        // Base Stable Controller (PD): K = [15.0, 5.0]
        let K_stable = SMatrix::<f64, 1, 2>::new(15.0, 5.0) + noise_vec;
        sims.push(Simulation::new(i, format!("Stable PD {}", i + 1), K_stable));
    }

    //  UNSTABLE Starts (Zero or Random small gains)
    for i in 5..20 {
        let noise_dist = Uniform::new(-20.0, 20.0).unwrap();
        let K_unstable = SMatrix::<f64, DIM_U, DIM_X>::from_fn(|_, _| noise_dist.sample(&mut rng));
        sims.push(Simulation::new(
            i,
            format!("Unstable Rand {}", i - 4),
            K_unstable,
        ));
    }

    //  Run LSPI Loop
    // Exploration noise needs to be significant to excite the dynamics
    let exploration_dist = Normal::new(0.0, 5.0).unwrap();

    for iter in 0..MAX_ITERATIONS {
        println!("--- Iteration {}/{} ---", iter + 1, MAX_ITERATIONS);

        for sim in &mut sims {
            // A. Run LSTDQ
            let (theta, _) = run_lstdq(&system, &sim.K, &exploration_dist, &mut rng);

            // B. Update Policy
            let H = theta_to_H(&theta);
            let K_new = compute_K_from_H(&H);

            // C. Diagnostics
            let error = (K_new - K_star).norm();
            let rho = spectral_radius(&system.A, &system.B, &K_new);

            sim.diag_iterations.push(iter);
            sim.diag_policy_error.push(error);
            sim.diag_spectral_radius.push(rho);
            sim.K = K_new;
        }
    }

    // 4. Plot Results
    println!("-> Generating Plots...");
    plot_combined_results(&sims)?;
    println!("-> Done.");

    Ok(())
}

// --- Plotting Logic ---
// --- Plotting Logic (Both Graphs) ---
fn plot_combined_results(sims: &Vec<Simulation>) -> Result<(), Box<dyn Error>> {
    let x_max = MAX_ITERATIONS as f64;

    // 1. Policy Error Plot (Clamped)
    {
        let root =
            BitMapBackend::new("01_pendulum_policy_error.png", (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        // Cap the error view at 10.0 because unstable gains might be huge
        let max_err = 10.0;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Policy Error (Green=Stable Start, Red=Unstable Start)",
                ("sans-serif", 30),
            )
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0f64..x_max, 0f64..max_err)?;

        chart
            .configure_mesh()
            .y_desc("Error ||K - K*||")
            .x_desc("Iterations")
            .draw()?;

        for (i, sim) in sims.iter().enumerate() {
            // COLOR LOGIC: Green for Stable (0-4), Red for Unstable (5-9)
            let color = if i < 5 {
                let intensity = 100 + (i as u8 * 30);
                RGBColor(0, intensity, 0).mix(0.8)
            } else {
                let intensity = 100 + ((i - 5) as u8 * 10);
                RGBColor(intensity, 0, 0).mix(0.8)
            };

            chart
                .draw_series(LineSeries::new(
                    sim.diag_iterations
                        .iter()
                        .zip(sim.diag_policy_error.iter())
                        .map(|(x, y)| (*x as f64, *y)),
                    Into::<ShapeStyle>::into(&color).stroke_width(2),
                ))?
                .label(&sim.label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }

    // 2. Spectral Radius Plot (ZOOMED IN)
    {
        let root =
            BitMapBackend::new("02_pendulum_spectral_radius.png", (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        // Zoomed in Y-axis
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Spectral Radius (Green=Converging, Red=Diverging)",
                ("sans-serif", 30),
            )
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0f64..x_max, 0f64..1.5f64)?;

        chart
            .configure_mesh()
            .y_desc("Spectral Radius (Rho)")
            .x_desc("Iterations")
            .draw()?;

        // Draw Stability Threshold Line (At exactly 1.0)
        chart
            .draw_series(LineSeries::new(
                (0..MAX_ITERATIONS).map(|x| (x as f64, 1.0)),
                BLACK.filled().stroke_width(3),
            ))?
            .label("Stability Limit (1.0)");

        for (i, sim) in sims.iter().enumerate() {
            // SAME COLOR LOGIC
            let color = if i < 5 {
                let intensity = 100 + (i as u8 * 30);
                RGBColor(0, intensity, 0).mix(0.8)
            } else {
                let intensity = 100 + ((i - 5) as u8 * 10);
                RGBColor(intensity, 0, 0).mix(0.8)
            };

            chart
                .draw_series(LineSeries::new(
                    sim.diag_iterations
                        .iter()
                        .zip(sim.diag_spectral_radius.iter())
                        .map(|(x, y)| (*x as f64, *y)),
                    Into::<ShapeStyle>::into(&color).stroke_width(2),
                ))?
                .label(&sim.label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(())
}
// --- Constants Update ---
const ETA: f64 = 0.1; // Parameter eta from Algorithm 2 (Tune this!)

// --- Updated LSTDQ Implementation (Algorithm 2) ---
fn run_lstdq(
    sys: &PendulumSystem,
    K: &SMatrix<f64, DIM_U, DIM_X>,
    dist: &impl Distribution<f64>,
    rng: &mut ThreadRng,
) -> (SVector<f64, DIM_PARAMS>, f64) {
    let mut A_mat = DMatrix::<f64>::zeros(DIM_PARAMS, DIM_PARAMS);
    let mut b_vec = DVector::<f64>::zeros(DIM_PARAMS);

    // Construct matrix M = [I; K]
    let mut M_mat = SMatrix::<f64, DIM_X_AND_U, DIM_X>::zeros();

    // Top block is Identity (Dimension X)
    M_mat
        .fixed_view_mut::<DIM_X, DIM_X>(0, 0)
        .copy_from(&SMatrix::<f64, DIM_X, DIM_X>::identity());

    // Bottom block is K (Dimension U x X)
    M_mat.fixed_view_mut::<DIM_U, DIM_X>(DIM_X, 0).copy_from(K);

    // Compute the outer product: P = M * M^T
    let P = M_mat * M_mat.transpose();

    // Vectorize P to get the bias vector: bias = eta * svec(P)
    let mut bias_vec = SVector::<f64, DIM_PARAMS>::zeros();
    let mut idx = 0;
    // Note: We must use the same svec ordering as get_quadratic_features
    for i in 0..DIM_X_AND_U {
        for j in i..DIM_X_AND_U {
            let val = P[(i, j)];
            bias_vec[idx] = val * ETA;
            idx += 1;
        }
    }

    let mut x = SVector::<f64, DIM_X>::from_fn(|_, _| dist.sample(rng) * 0.1);

    for _ in 0..SAMPLES_PER_ITER {
        let noise = SVector::<f64, DIM_U>::from_fn(|_, _| dist.sample(rng));
        let u = (-K * x) + noise; // Exploration Policy

        // Cost calculation
        let cost = x.dot(&(sys.Q * x)) + u.dot(&(sys.R * u));

        let x_next = sys.step(&x, &u);

        // Feature computation with BIAS
        // phi(x,u) = svec([x;u][x;u]^T) + bias_vec
        let phi = get_quadratic_features(&x, &u) + bias_vec;

        let u_next_greedy = -K * x_next;
        let phi_next = get_quadratic_features(&x_next, &u_next_greedy) + bias_vec;

        // LSTDQ Update
        let temporal_diff = &phi - (GAMMA * &phi_next);

        for r in 0..DIM_PARAMS {
            let phi_r = phi[r];
            b_vec[r] += phi_r * cost;
            for c in 0..DIM_PARAMS {
                A_mat[(r, c)] += phi_r * temporal_diff[c];
            }
        }

        x = x_next;
        if sys.resettable && x.norm() > 2.0 {
            // Pendulum reset
            x = SVector::<f64, DIM_X>::from_fn(|_, _| dist.sample(rng) * 0.1);
        }
    }

    for i in 0..DIM_PARAMS {
        A_mat[(i, i)] += LAMBDA_REG;
    }

    let theta_dyn = A_mat
        .lu()
        .solve(&b_vec)
        .unwrap_or(DVector::zeros(DIM_PARAMS));
    let mut theta = SVector::<f64, DIM_PARAMS>::zeros();
    theta.copy_from_slice(theta_dyn.as_slice());

    (theta, 0.0)
}
// --- Helpers ---
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

fn theta_to_H(theta: &SVector<f64, DIM_PARAMS>) -> SMatrix<f64, DIM_X_AND_U, DIM_X_AND_U> {
    let mut H = SMatrix::<f64, DIM_X_AND_U, DIM_X_AND_U>::zeros();
    let mut idx = 0;
    for i in 0..DIM_X_AND_U {
        for j in i..DIM_X_AND_U {
            let val = theta[idx];
            if i == j {
                H[(i, j)] = val;
            } else {
                H[(i, j)] = val * 0.5;
                H[(j, i)] = val * 0.5;
            }
            idx += 1;
        }
    }
    H
}

fn compute_K_from_H(H: &SMatrix<f64, DIM_X_AND_U, DIM_X_AND_U>) -> SMatrix<f64, DIM_U, DIM_X> {
    let Q_uu = H.fixed_view::<DIM_U, DIM_U>(DIM_X, DIM_X);
    let Q_ux = H.fixed_view::<DIM_U, DIM_X>(DIM_X, 0);
    match Q_uu.try_inverse() {
        Some(inv) => inv * Q_ux,
        None => SMatrix::<f64, DIM_U, DIM_X>::identity(),
    }
}

fn spectral_radius(
    A: &SMatrix<f64, DIM_X, DIM_X>,
    B: &SMatrix<f64, DIM_X, DIM_U>,
    K: &SMatrix<f64, DIM_U, DIM_X>,
) -> f64 {
    let A_cl = A - B * K;
    let eig = A_cl.complex_eigenvalues();
    eig.iter()
        .map(|c| c.norm())
        .fold(0.0, |a, b| f64::max(a, b))
}

fn solve_dare_analytic(sys: &PendulumSystem) -> SMatrix<f64, DIM_U, DIM_X> {
    let mut P = sys.Q.clone();
    for _ in 0..5000 {
        let term1 = sys.R + GAMMA * sys.B.transpose() * P * sys.B;
        let term1_inv = term1.try_inverse().unwrap();
        let term2 = GAMMA * sys.B.transpose() * P * sys.A;
        let P_next = sys.Q + GAMMA * sys.A.transpose() * P * sys.A
            - sys.A.transpose() * term2.transpose() * term1_inv * term2;
        if (P_next - P).norm() < 1e-9 {
            P = P_next;
            break;
        }
        P = P_next;
    }
    let term1 = sys.R + GAMMA * sys.B.transpose() * P * sys.B;
    let term1_inv = term1.try_inverse().unwrap();
    term1_inv * (GAMMA * sys.B.transpose() * P * sys.A)
}
