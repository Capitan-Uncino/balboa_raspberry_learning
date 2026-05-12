use crate::file_utils::get_next_file_index;
use crate::graphic_utils::plot_cost_evolution;
use crate::learning::lstdq_no_bias_correction::{
    calculate_k, StateAction, ANALYTIC_LQR_POLICY, DIM_U, DIM_X, SAMPLES_PER_ITER,
};
use crate::logging_utils::log_progress;
use mujoco_rs::prelude::*;
use mujoco_rs::viewer::MjViewer;
use nalgebra::{SMatrix, SVector};
use rand::rng;
use rand::RngExt;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const THETA_OU: f64 = 0.30;
const SIGMA_OU: f64 = 0.10;
const SEED: u64 = 42;
const BACKLASH_JOINTS: bool = true;

pub struct LinearRobotSim {
    pub phi: f64,
    pub theta: f64,
    pub phi_dot: f64,
    pub theta_dot: f64,
}

impl LinearRobotSim {
    pub fn new() -> Self {
        Self {
            phi: 0.0,
            theta: 0.05, // Initial angle (roughly ~2.8 degrees) to kickstart
            phi_dot: 0.0,
            theta_dot: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.phi = 0.0;
        self.theta = 0.05;
        self.phi_dot = 0.0;
        self.theta_dot = 0.0;
    }

    /// Steps the physics forward using Euler integration on the linearized state-space model
    pub fn step(&mut self, tau_0: f64, dt: f64) {
        // Physical Constants (Table 1)
        const M_W: f64 = 0.0042;
        const M_P: f64 = 0.316;
        const R: f64 = 0.040;
        const L: f64 = 0.023;
        const I_P: f64 = 444.43e-6;
        const I_W: f64 = 26.89e-6;
        const G: f64 = 9.81;

        // E Matrix Elements
        let e11 = I_W + (R * R) * (M_W + M_P);
        let e12 = M_P * R * L;
        let e21 = M_P * R * L;
        let e22 = I_P + M_P * (L * L);

        // Inverse of E Matrix
        let det_e = (e11 * e22) - (e12 * e21);
        let inv_e11 = e22 / det_e;
        let inv_e12 = -e12 / det_e;
        let inv_e21 = -e21 / det_e;
        let inv_e22 = e11 / det_e;

        // G Matrix * theta
        // G = [0, -m_p * g * l]^T
        let g_theta_1 = 0.0;
        let g_theta_2 = -M_P * G * L * self.theta;

        // E^-1 * G * theta
        let e_inv_g_1 = (inv_e11 * g_theta_1) + (inv_e12 * g_theta_2);
        let e_inv_g_2 = (inv_e21 * g_theta_1) + (inv_e22 * g_theta_2);

        // E^-1 * F * tau_0
        // F = [1, 0]^T
        let e_inv_f_1 = inv_e11 * 1.0 + inv_e12 * 0.0;
        let e_inv_f_2 = inv_e21 * 1.0 + inv_e22 * 0.0;

        // Calculate accelerations: x_ddot = E^-1 * F * tau_0 - E^-1 * G * theta
        let phi_ddot = (e_inv_f_1 * tau_0) - e_inv_g_1;
        let theta_ddot = (e_inv_f_2 * tau_0) - e_inv_g_2;

        // Euler Integration Step
        self.phi += self.phi_dot * dt;
        self.theta += self.theta_dot * dt;
        self.phi_dot += phi_ddot * dt;
        self.theta_dot += theta_ddot * dt;
    }
}

pub fn collect_full_batch_sim(
    sim: &mut LinearRobotSim,
    log_label: &str,
    batch_index: usize,
    was_balancing: &mut bool,
    pending_gains: &Arc<Mutex<Option<[f64; 4]>>>,
    active_gains: &mut [f64; 4],
    enable_noise: bool,
) -> Vec<StateAction> {
    let mut state_batch = Vec::with_capacity(SAMPLES_PER_ITER);
    let mut loop_counter = 0;
    let mut stability_counter = 0;

    let stability_threshold = 10;
    let stop_angle_rad = 60.0_f64.to_radians();

    // Simulation timing parameters
    let control_step: f64 = 0.01;
    let physics_dt: f64 = 0.001; // Internal physics step (1ms) for stable Euler integration
    let sim_steps = (control_step / physics_dt).round() as usize;

    let mut last_noise: f64 = 0.0;
    let mut rng = rng();

    println!(">>> [SIM] Running Pure Linear Simulation Loop...");

    while state_batch.len() < SAMPLES_PER_ITER {
        if let Some(new_gains) = pending_gains.lock().unwrap().take() {
            println!("---> [SIM] Applying New Gains: {:?}", new_gains);
            *active_gains = new_gains;
        }

        // --- EXTRACT LINEAR STATE ---
        let phi = sim.phi;
        let theta = sim.theta;
        let phi_dot = sim.phi_dot;
        let theta_dot = sim.theta_dot;

        let k1 = active_gains[0];
        let k2 = active_gains[1];
        let k3 = active_gains[2];
        let k4 = active_gains[3];

        // Linear control law
        let u_raw = k1 * phi + k2 * theta + k3 * phi_dot + k4 * theta_dot;

        // ==========================================
        // OU EXPLORATION NOISE
        // ==========================================
        let mut epsilon = 0.0;
        if enable_noise {
            let u1: f64 = rng.random_range(0.0001..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            epsilon = (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * PI * u2).cos();
            let dx = THETA_OU * (-last_noise) * 0.01 + SIGMA_OU * epsilon * 0.1;
            last_noise += dx;
        }

        // ==========================================
        // DIRECT TORQUE CONTROL (tau_0)
        // ==========================================
        // Stripped of all electrical dynamics, quantization, and mechanical limits.
        // We directly apply the requested control (plus noise) to the linear system.
        let tau_0 = u_raw + epsilon * SIGMA_OU;

        // ==========================================
        // STEP LINEAR PHYSICS
        // ==========================================
        for _ in 0..sim_steps {
            sim.step(tau_0, physics_dt);
        }

        let current_data = StateAction {
            phi: sim.phi,
            theta: sim.theta,
            phi_dot: sim.phi_dot,
            theta_dot: sim.theta_dot,
            u: tau_0, // Recording the pure torque applied
        };

        let is_sane = current_data.theta.is_finite() && current_data.theta_dot.abs() < 100.0;
        let is_upright = current_data.theta.abs() < stop_angle_rad;

        if is_sane && is_upright {
            stability_counter += 1;
            if stability_counter >= stability_threshold {
                if !*was_balancing {
                    println!(">>> [SIM] ROBOT STANDING: Resuming {}...", log_label);
                    *was_balancing = true;
                }
                state_batch.push(current_data);
                loop_counter += 1;
                if loop_counter >= 100 {
                    println!(
                        "[SIM] Collected {}/{} samples",
                        state_batch.len(),
                        SAMPLES_PER_ITER
                    );
                    loop_counter = 0;
                }
            }
        } else {
            stability_counter = 0;
            if *was_balancing {
                println!(
                    "<<< [SIM] ROBOT FELL: Pausing {} (Collected: {}/{})",
                    log_label,
                    state_batch.len(),
                    SAMPLES_PER_ITER
                );
                *was_balancing = false;

                // Reset the linear simulation to starting conditions
                sim.reset();
                last_noise = 0.0;
            }
        }
    }

    state_batch
}

pub fn run_online_mode_sim(_visualize: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing Linear Simulation Model...");

    let mut sim = LinearRobotSim::new();
    let initial_k_array = ANALYTIC_LQR_POLICY;

    // In a perfectly linear deterministic simulation, inherent process noise is zero.
    // We mock the noise mean and covariance to zeros for the LSTDQ calculations.
    let noise_mean = [0.0; 4];
    let noise_cov = nalgebra::DMatrix::<f64>::zeros(4, 4);
    let noise = (noise_mean, noise_cov);

    println!("============================================================");
    println!("        PROCESS NOISE DIAGNOSTIC REPORT (LINEAR)");
    println!("============================================================");

    // 1. Print the Mean Vector (The "Bias")
    println!("MEAN VECTOR (Systematic Bias / Residuals):");
    println!("  [ φ_dot,     θ_dot,     φ_ddot,    θ_ddot ]");
    println!(
        "  [{:+.4e}, {:+.4e}, {:+.4e}, {:+.4e}]",
        noise.0[0], noise.0[1], noise.0[2], noise.0[3]
    );

    // 2. Print the Covariance Matrix (The "Variance")
    println!("\nCOVARIANCE MATRIX (Sigma):");
    println!("            φ_dot           θ_dot           φ_ddot          θ_ddot");
    let labels = ["φ_dot ", "θ_dot ", "φ_ddot", "θ_ddot"];
    for i in 0..4 {
        print!("{} ", labels[i]);
        for j in 0..4 {
            print!("{:>15.4e} ", noise.1[(i, j)]);
        }
        println!();
    }

    println!("============================================================");

    let initial_k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(&[
        initial_k_array[0],
        initial_k_array[1],
        initial_k_array[2],
        initial_k_array[3],
    ]);

    let current_k = Arc::new(Mutex::new(initial_k_mat));
    let pending_gains: Arc<Mutex<Option<[f64; 4]>>> = Arc::new(Mutex::new(None));
    let mut active_gains = initial_k_array;

    let mut computations_completed = 0;
    let mut was_balancing = false;

    println!("Starting [LINEAR SIM] 100Hz control loop...");

    let enable_noise = true;

    // Define a maximum number of batches to run since we no longer have a viewer window to close
    let max_iterations = 1000;

    for _ in 0..max_iterations {
        let batch_to_process = collect_full_batch_sim(
            &mut sim,
            "LSTDQ Batch",
            computations_completed,
            &mut was_balancing,
            &pending_gains,
            &mut active_gains,
            enable_noise,
        );

        if batch_to_process.is_empty() {
            break;
        }

        computations_completed += 1;
        let k_clone = Arc::clone(&current_k);
        let pending_clone = Arc::clone(&pending_gains);

        // Clone the noise matrix to move it into the thread
        let noise_cov_clone = noise.1.clone();

        std::thread::spawn(move || {
            let k_to_use = { *k_clone.lock().unwrap() };
            let new_k_mat = calculate_k(batch_to_process, &k_to_use);

            {
                *k_clone.lock().unwrap() = new_k_mat;
            }

            let new_k_array = [
                new_k_mat[(0, 0)],
                new_k_mat[(0, 1)],
                new_k_mat[(0, 2)],
                new_k_mat[(0, 3)],
            ];

            *pending_clone.lock().unwrap() = Some(new_k_array);
            println!(">>> LSTDQ Update: New K vector queued for next SIM window.");
        });
    }

    println!("Simulation finished.");
    Ok(())
}

pub fn run_data_collection_mode_sim(_visualize: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing Linear Simulation Model...");

    let mut sim = LinearRobotSim::new();
    let data_dir = "collected_data";

    // Assuming get_next_file_index() is defined in your existing code
    let mut file_index = get_next_file_index();
    let mut was_balancing = false;
    let dummy_pending_gains: Arc<Mutex<Option<[f64; 4]>>> = Arc::new(Mutex::new(None));

    let mut active_gains = ANALYTIC_LQR_POLICY;

    println!(
        "Started [LINEAR SIM] data collection mode. Will start at index: {}",
        file_index
    );

    // Limit collection size to prevent filling the hard drive infinitely
    let batches_to_collect = 100;

    for _ in 0..batches_to_collect {
        let batch_to_process = collect_full_batch_sim(
            &mut sim,
            "CSV Collection",
            file_index,
            &mut was_balancing,
            &dummy_pending_gains,
            &mut active_gains,
            true, // enable_noise
        );

        if batch_to_process.is_empty() {
            break;
        }

        let filename = format!("{}/batch_{}.csv", data_dir, file_index);
        let mut file = std::fs::File::create(&filename)?;

        use std::io::Write;
        writeln!(file, "phi,theta,phi_dot,theta_dot,u")?;
        for s in &batch_to_process {
            writeln!(
                file,
                "{},{},{},{},{}",
                s.phi, s.theta, s.phi_dot, s.theta_dot, s.u
            )?;
        }

        println!(
            ">>> Saved SIM batch of size {} to {} (Next: {})",
            batch_to_process.len(),
            filename,
            file_index + 1
        );
        file_index += 1;
    }

    println!("Data collection finished.");
    Ok(())
}

pub fn run_sim_plot(
    visualize: bool,
    evaluation_threshold: f64,
    n_policies: usize,
    n_updates: usize,
    uniform_half_interval: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing Linear Simulation Model...");
    let mut sim = LinearRobotSim::new();

    let initial_k_array = ANALYTIC_LQR_POLICY;

    // --- 1. Evaluate the Original Baseline Policy ---
    println!("Evaluating original baseline policy...");
    let baseline_cost = evaluate_policy_sim(&mut sim, initial_k_array, false);
    println!("Baseline Cost: {:.4}", baseline_cost);

    // --- 2. Generate N Policies via Uniform Perturbation ---
    let mut rng = rng();
    let uniform_dist = Uniform::new_inclusive(
        1.0f64 - uniform_half_interval,
        1.0f64 + uniform_half_interval,
    )?;

    let mut active_policies: Vec<(usize, [f64; 4])> = (0..n_policies)
        .map(|id| {
            let p = [
                initial_k_array[0] * uniform_dist.sample(&mut rng),
                initial_k_array[1] * uniform_dist.sample(&mut rng),
                initial_k_array[2] * uniform_dist.sample(&mut rng),
                initial_k_array[3] * uniform_dist.sample(&mut rng),
            ];
            (id, p)
        })
        .collect();

    let mut cost_history: Vec<Vec<f64>> = vec![Vec::new(); n_policies];

    // --- 3. Main Loop: Evaluation & Update ---
    println!("Starting multi-policy evaluation and update loop...");

    for update_idx in 0..n_updates {
        println!("\r");
        println!("--- Update Step {} / {} ---", update_idx + 1, n_updates);

        let mut next_active_policies = Vec::new();

        for (p_idx, mut policy) in active_policies {
            // A) EVALUATION PHASE (Noise OFF)
            let empirical_cost = evaluate_policy_sim(&mut sim, policy, false);
            cost_history[p_idx].push(empirical_cost);
            println!("  Policy {} - Eval Cost: {:.4}", p_idx, empirical_cost);

            if empirical_cost > evaluation_threshold {
                println!(
                    "  [!] Policy {} exceeded threshold ({:.4} > {:.4}). Discarding from training.",
                    p_idx, empirical_cost, evaluation_threshold
                );
                continue;
            }

            // B) BATCH COLLECTION PHASE (Noise ON)
            let mut active_gains = policy;
            let pending_gains: Arc<Mutex<Option<[f64; 4]>>> = Arc::new(Mutex::new(None));
            let mut was_balancing = false;

            let batch_to_process = collect_full_batch_sim(
                &mut sim,
                &format!("LSTDQ P{} U{}", p_idx, update_idx),
                update_idx,
                &mut was_balancing,
                &pending_gains,
                &mut active_gains,
                true, // enable_noise = true
            );

            if batch_to_process.is_empty() {
                println!("  [!] Batch empty, skipped collection.");
                next_active_policies.push((p_idx, policy));
                continue;
            }

            // C) UPDATE PHASE (Synchronous)
            let current_k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(&policy);
            let new_k_mat = calculate_k(batch_to_process, &current_k_mat);

            policy = [
                new_k_mat[(0, 0)],
                new_k_mat[(0, 1)],
                new_k_mat[(0, 2)],
                new_k_mat[(0, 3)],
            ];

            next_active_policies.push((p_idx, policy));
        }

        active_policies = next_active_policies;

        if active_policies.is_empty() {
            println!("All policies have been discarded. Terminating training loop early.");
            break;
        }
    }

    // --- 4. Final Evaluation ---
    for (p_idx, policy) in active_policies.iter() {
        let final_cost = evaluate_policy_sim(&mut sim, *policy, false);
        cost_history[*p_idx].push(final_cost);
    }

    // --- 4.5 Print Surviving Policies Table ---
    println!("\n=================================================================================================");
    println!("                                 FINAL SURVIVING POLICIES REPORT                                 ");
    println!("=================================================================================================");
    println!("| Policy ID | Final Cost |   K1 (φ_dot)    |   K2 (θ_dot)    |   K3 (φ_ddot)   |   K4 (θ_ddot)   |");
    println!("|-----------|------------|-----------------|-----------------|-----------------|-----------------|");

    if active_policies.is_empty() {
        println!("|                           No policies survived the evaluation threshold.                        |");
    } else {
        for (p_idx, policy) in active_policies.iter() {
            let final_cost = cost_history[*p_idx].last().unwrap_or(&f64::NAN);
            println!(
                "| {:^9} | {:^10.4} | {:>15.4} | {:>15.4} | {:>15.4} | {:>15.4} |",
                p_idx, final_cost, policy[0], policy[1], policy[2], policy[3]
            );
        }
    }
    println!("=================================================================================================\n");

    // --- 5. Plotting ---
    println!("Generating plot 'policy_evolution.png'...");
    plot_cost_evolution(
        &cost_history,
        baseline_cost,
        n_updates,
        evaluation_threshold,
    )?;

    Ok(())
}

fn evaluate_policy_sim(
    sim: &mut LinearRobotSim,
    policy_gains: [f64; 4],
    enable_noise: bool,
) -> f64 {
    // --- EVALUATION PARAMETERS ---
    let eval_steps = 10000;
    let control_step: f64 = 0.01;
    let physics_dt: f64 = 0.001; // Internal solver dt
    let sim_steps = (control_step / physics_dt).round() as usize;
    let stop_angle_rad = 60.0_f64.to_radians();

    // --- LQR COST WEIGHTS ---
    let q_phi = 10.0;
    let q_theta = 100.0;
    let q_phi_dot = 1.0;
    let q_theta_dot = 10.0;
    let r_tau = 300.0;

    // --- RNG INITIALIZATION ---
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    // --- RESET SIMULATION ---
    sim.reset();

    // 1. Randomize initial pitch (Theta) between +/- 10 degrees (~0.1745 rad)
    sim.theta = rng.random_range(-0.1745..0.1745);
    // 2. Randomize initial pitch velocity (Theta_dot)
    sim.theta_dot = rng.random_range(-0.1..0.1);

    let mut total_cost = 0.0;
    let mut last_noise: f64 = 0.0;

    for step in 0..eval_steps {
        // --- 1. EXTRACT STATE ---
        let phi = sim.phi;
        let theta = sim.theta;
        let phi_dot = sim.phi_dot;
        let theta_dot = sim.theta_dot;

        // --- 2. STABILITY CHECK ---
        let is_sane = theta.is_finite() && theta_dot.abs() < 100.0;
        let is_upright = theta.abs() < stop_angle_rad;

        if !is_sane || !is_upright {
            // Heavily penalize remaining timesteps for falling
            let penalty_per_step = 100_000.0;
            total_cost += penalty_per_step * (eval_steps - step) as f64;
            break;
        }

        // --- 3. LQR CONTROL LAW ---
        let k1 = policy_gains[0];
        let k2 = policy_gains[1];
        let k3 = policy_gains[2];
        let k4 = policy_gains[3];

        let u_raw = k1 * phi + k2 * theta + k3 * phi_dot + k4 * theta_dot;

        // --- 4. OPTIONAL NOISE ---
        if enable_noise {
            let u1: f64 = rng.random_range(0.0001..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let epsilon = (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * PI * u2).cos();
            let dx = THETA_OU * (-last_noise) * 0.01 + SIGMA_OU * epsilon * 0.1;
            last_noise += dx;
        } else {
            last_noise = 0.0;
        }

        // --- 5. DIRECT TORQUE APPLICATION & EMPIRICAL COST ---
        let tau_total = u_raw + last_noise;

        let state_cost = q_phi * phi.powi(2)
            + q_theta * theta.powi(2)
            + q_phi_dot * phi_dot.powi(2)
            + q_theta_dot * theta_dot.powi(2);

        // Keep R penalty scale consistent with original code (which penalized tau_total/2)
        let raw_tau = tau_total / 2.0;
        let action_cost = r_tau * raw_tau.powi(2);

        total_cost += state_cost + action_cost;

        // --- 6. STEP PURE LINEAR PHYSICS ---
        for _ in 0..sim_steps {
            sim.step(tau_total, physics_dt);
        }
    }

    total_cost / (eval_steps as f64)
}
