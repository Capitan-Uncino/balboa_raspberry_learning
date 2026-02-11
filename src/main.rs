use nalgebra::{ComplexField, DMatrix, DVector, SMatrix, SVector};
use plotters::prelude::*;
use rand::prelude::*;
use rand::rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::error::Error;

// --- Constants ---
const DIM_X: usize = 3;
const DIM_U: usize = 3;
const DIM_X_AND_U: usize = DIM_X + DIM_U;
const DIM_PARAMS: usize = (DIM_X_AND_U * (DIM_X_AND_U + 1)) / 2;

// Paper settings (exact)
const GAMMA: f64 = 0.98;
const SAMPLES_PER_ITER: usize = 50000;
const MAX_ITERATIONS: usize = 15;
const LAMBDA_REG: f64 = 1e-6;

// --- System Definition ---
struct LQRSystem {
    A: SMatrix<f64, DIM_X, DIM_X>,
    B: SMatrix<f64, DIM_X, DIM_U>,
    Q: SMatrix<f64, DIM_X, DIM_X>,
    R: SMatrix<f64, DIM_U, DIM_U>,
}

impl LQRSystem {
    fn new_paper_instance() -> Self {
        // A from Eq 5.1 in paper (Spectral radius ~1.02)
        let A = SMatrix::<f64, 3, 3>::new(1.01, 0.01, 0.00, 0.01, 1.01, 0.01, 0.00, 0.01, 1.01);
        let B = SMatrix::<f64, 3, 3>::identity();
        let Q = SMatrix::<f64, 3, 3>::identity() * 1e-3;
        let R = SMatrix::<f64, 3, 3>::identity();

        Self { A, B, Q, R }
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
    diag_theta_norm: Vec<f64>,
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
            diag_theta_norm: Vec::new(),
        }
    }
}

// --- Main Algorithm ---
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== LSPI Multi-Start Experiment ===");
    let system = LQRSystem::new_paper_instance();

    // 1. Compute Analytic Ground Truth
    let K_star = solve_dare_analytic(&system);
    let rho_star = spectral_radius(&system.A, &system.B, &K_star);
    println!("Optimal Spectral Radius: {:.4}", rho_star);

    // 2. Initialize 10 Simulations
    let mut sims = Vec::new();
    let mut rng = rng();

    // --> 5 STABLE Identifications (K ~ 0.2 * I + noise)
    // A ~ 1.01. K=0.2 makes A-BK ~ 0.8. Stable.
    for i in 0..5 {
        // FIX 1: Added .unwrap() because Uniform::new returns a Result now
        let noise_dist = Uniform::new(-0.1, 0.1).unwrap();
        let noise = SMatrix::<f64, DIM_U, DIM_X>::from_fn(|_, _| noise_dist.sample(&mut rng));
        let K_init = (SMatrix::<f64, DIM_U, DIM_X>::identity() * 0.2) + noise;
        sims.push(Simulation::new(i, format!("Stable {}", i + 1), K_init));
    }

    // --> 5 UNSTABLE Identifications (K ~ 0.0 + noise)
    // A ~ 1.01. K=0.0 makes A-BK ~ 1.01. Unstable.
    for i in 5..10 {
        // FIX 1: Added .unwrap()
        let noise_dist = Uniform::new(-0.1, 0.1).unwrap();
        let K_init = SMatrix::<f64, DIM_U, DIM_X>::zeros()
            + SMatrix::from_fn(|_, _| noise_dist.sample(&mut rng));
        sims.push(Simulation::new(i, format!("Unstable {}", i - 4), K_init));
    }

    // 3. Run Loop
    let exploration_dist = Normal::new(0.0, 1.0).unwrap();

    for iter in 0..MAX_ITERATIONS {
        println!("--- Iteration {}/{} ---", iter + 1, MAX_ITERATIONS);

        // Update every simulation
        for sim in &mut sims {
            // A. Run LSTDQ
            let (theta, _cond) = run_lstdq(&system, &sim.K, &exploration_dist, &mut rng);

            // B. Update Policy
            let H = theta_to_H(&theta);
            let K_new = compute_K_from_H(&H);

            // C. Record Diagnostics
            let error = (K_new - K_star).norm();
            let rho = spectral_radius(&system.A, &system.B, &K_new);
            let theta_n = theta.norm();

            sim.diag_iterations.push(iter);
            sim.diag_policy_error.push(error);
            sim.diag_spectral_radius.push(rho);
            sim.diag_theta_norm.push(theta_n);

            sim.K = K_new;
        }
    }

    // 4. Plot All
    println!("-> Generating Combined Plots...");
    plot_combined_results(&sims)?;
    println!("-> Done.");

    Ok(())
}

// --- Plotting Logic (Greens vs Reds) ---
fn plot_combined_results(sims: &Vec<Simulation>) -> Result<(), Box<dyn Error>> {
    let x_max = MAX_ITERATIONS as f64;

    // 1. Policy Error Plot (Clamped)
    {
        let root = BitMapBackend::new("01_multi_policy_error.png", (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        // Cap the error view at 2.0 so huge errors don't squash the chart
        let max_err = 2.0;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Policy Error (Green=Stable Start, Red=Unstable Start)",
                ("sans-serif", 30),
            )
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0f64..x_max, 0f64..max_err)?;

        chart.configure_mesh().draw()?;

        for (i, sim) in sims.iter().enumerate() {
            // COLOR LOGIC:
            // - Stable (0-4): Green gradients
            // - Unstable (5-9): Red gradients
            let color = if i < 5 {
                // Variation of Green: RGB(0, 100..255, 0)
                let intensity = 100 + (i as u8 * 30);
                RGBColor(0, intensity, 0).mix(0.8)
            } else {
                // Variation of Red: RGB(100..255, 0, 0)
                let intensity = 100 + ((i - 5) as u8 * 30);
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
            BitMapBackend::new("02_multi_spectral_radius.png", (1024, 768)).into_drawing_area();
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
                let intensity = 100 + ((i - 5) as u8 * 30);
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
} // --- LSTDQ Implementation ---
fn run_lstdq(
    sys: &LQRSystem,
    K: &SMatrix<f64, DIM_U, DIM_X>,
    dist: &impl Distribution<f64>,
    rng: &mut ThreadRng,
) -> (SVector<f64, DIM_PARAMS>, f64) {
    let mut A_mat = DMatrix::<f64>::zeros(DIM_PARAMS, DIM_PARAMS);
    let mut b_vec = DVector::<f64>::zeros(DIM_PARAMS);
    let mut x = SVector::<f64, DIM_X>::from_fn(|_, _| dist.sample(rng));

    for _ in 0..SAMPLES_PER_ITER {
        let noise = SVector::<f64, DIM_U>::from_fn(|_, _| dist.sample(rng));
        let u = (-K * x) + noise;
        let cost = x.dot(&(sys.Q * x)) + u.dot(&(sys.R * u)); // Discounted Cost
        let x_next = sys.step(&x, &u);

        let phi = get_quadratic_features(&x, &u);
        let u_next_greedy = -K * x_next;
        let phi_next = get_quadratic_features(&x_next, &u_next_greedy);

        let temporal_diff = &phi - (GAMMA * &phi_next);

        // Manual outer product
        for r in 0..DIM_PARAMS {
            let phi_r = phi[r];
            let cost_term = phi_r * cost;
            b_vec[r] += cost_term;
            for c in 0..DIM_PARAMS {
                A_mat[(r, c)] += phi_r * temporal_diff[c];
            }
        }

        x = x_next;
        // Reset if unstable to keep data valid
        if x.norm() > 50.0 {
            x = SVector::<f64, DIM_X>::from_fn(|_, _| dist.sample(rng));
        }
    }

    for i in 0..DIM_PARAMS {
        A_mat[(i, i)] += LAMBDA_REG;
    }

    let svd = A_mat.clone().svd(false, false);
    let cond = svd.singular_values[0] / svd.singular_values[svd.singular_values.len() - 1];

    let theta_dyn = A_mat
        .lu()
        .solve(&b_vec)
        .unwrap_or(DVector::zeros(DIM_PARAMS));
    let mut theta = SVector::<f64, DIM_PARAMS>::zeros();
    theta.copy_from_slice(theta_dyn.as_slice());

    (theta, cond)
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
    eig.iter().map(|c| c.norm()).fold(0.0, f64::max)
}

fn solve_dare_analytic(sys: &LQRSystem) -> SMatrix<f64, DIM_U, DIM_X> {
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
