use crate::learning::lstdq_2019::{
    calculate_k, StateAction, ANALYTIC_LQR_POLICY, DIM_U, DIM_X, SAMPLES_PER_ITER,
};
use crate::utils::file_utils::get_next_file_index;
use crate::utils::graphic_utils::{log_progress, plot_cost_evolution};
use mujoco_rs::prelude::*;
use mujoco_rs::viewer::MjViewer;
use nalgebra::{SMatrix, SVector};
use rand::rng;
use rand::RngExt;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const THETA_OU: f64 = 0.30;
const SIGMA_OU: f64 = 0.10;
const SEED: u64 = 42;
const BACKLASH_JOINTS: bool = false;

fn collect_full_batch_sim<'a>(
    model: &'a MjModel,
    data: &mut MjData<&'a MjModel>,
    viewer: &mut MjViewer,
    log_label: &str,
    batch_index: usize,
    was_balancing: &mut bool,
    pending_gains: &Arc<Mutex<Option<[f64; 4]>>>,
    active_gains: &mut [f64; 4],
    enable_rendering: bool,
    enable_noise: bool,
) -> Vec<StateAction> {
    let mut state_batch = Vec::with_capacity(SAMPLES_PER_ITER);
    let mut loop_counter = 0;
    let mut stability_counter = 0;

    let stability_threshold = 10;
    let stop_angle_rad = 60.0_f64.to_radians();

    let control_step = 0.01;
    let timestep = model.opt().timestep;
    let sim_steps = (control_step / timestep).round() as usize;

    let mut last_noise: f64 = 0.0;
    let mut rng = rng();

    let target_duration = Duration::from_secs_f64(control_step);

    println!(">>> [SIM] Running MuJoCo Simulation Loop...");

    while state_batch.len() < SAMPLES_PER_ITER && (!enable_rendering || viewer.running()) {
        let step_start = Instant::now();

        if let Some(new_gains) = pending_gains.lock().unwrap().take() {
            println!("---> [SIM] Applying New Gains: {:?}", new_gains);
            *active_gains = new_gains;
        }

        // --- EXTRACT RAW STATE VIA HELPER ---
        let (phi_left, phi_right, theta, phi_dot_left, phi_dot_right, theta_dot) =
            extract_state(data, BACKLASH_JOINTS);

        // --- TRANSFORM INTO LQR STATE ---
        let phi = (phi_left + phi_right) / 2.0;
        let phi_dot = (phi_dot_left + phi_dot_right) / 2.0;

        let k1 = active_gains[0];
        let k2 = active_gains[1];
        let k3 = active_gains[2];
        let k4 = active_gains[3];

        let u_raw = k1 * phi + k2 * theta + k3 * phi_dot + k4 * theta_dot;

        // ==========================================
        // OU EXPLORATION NOISE
        // ==========================================
        let u1: f64 = rng.random_range(0.0001..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let epsilon = (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * PI * u2).cos();
        let dx = THETA_OU * (-last_noise) * 0.01 + SIGMA_OU * epsilon * 0.1;
        last_noise += dx;

        let max_physical_torque = 0.1;
        let pwm_resolution = 400.0;

        let tau_total = u_raw + last_noise;
        let raw_tau = tau_total / 2.0;

        let raw_tau_offset = if phi_dot > 0.0f64 {
            raw_tau + 0.00
        } else {
            raw_tau - 0.00
        };

        // Back-EMF constraints using raw velocities
        let max_speed = 25.0;
        let avail_l = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_left.abs() / max_speed)));
        let avail_r = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_right.abs() / max_speed)));

        let pwm_l_raw = (raw_tau_offset / max_physical_torque) * pwm_resolution;
        let pwm_r_raw = (raw_tau_offset / max_physical_torque) * pwm_resolution;

        let pwm_l_quantized = pwm_l_raw.round();
        let pwm_r_quantized = pwm_r_raw.round();

        let tau_l_quantized = (pwm_l_quantized / pwm_resolution) * max_physical_torque;
        let tau_r_quantized = (pwm_r_quantized / pwm_resolution) * max_physical_torque;

        let tau_l_final = tau_l_quantized.clamp(-avail_l, avail_l);
        let tau_r_final = tau_r_quantized.clamp(-avail_r, avail_r);

        data.ctrl_mut()[0] = tau_l_final;
        data.ctrl_mut()[1] = tau_r_final;

        for _ in 0..sim_steps {
            data.step();
        }

        // ==========================================
        // OPTIONAL RENDERING & REAL-TIME PACING
        // ==========================================
        if enable_rendering {
            viewer.sync_data(data);
            let _ = viewer.render();

            let elapsed = step_start.elapsed();
            if elapsed < target_duration {
                thread::sleep(target_duration - elapsed);
            }
        }

        let current_data = StateAction {
            phi,
            theta,
            phi_dot,
            theta_dot,
            u: raw_tau,
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
                    log_progress(state_batch.len(), SAMPLES_PER_ITER, batch_index, log_label);
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

                data.reset();
                data.qpos_mut()[2] = 0.05;
                data.qpos_mut()[3] = 1.0;

                last_noise = 0.0;
            }
        }
    }

    state_batch
}

pub fn run_online_mode_sim(visualize: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MuJoCo model 'balboa.xml'...");

    // mujoco-rs uses MjModel::from_xml for loading files
    let model = MjModel::from_xml("balboa.xml").expect("Failed to load balboa.xml");

    // Automatically allocates the physics state array based on the model
    let mut data = model.make_data();

    // Spawn the viewer in a background thread running at roughly 60 FPS
    let mut viewer =
        MjViewer::launch_passive(&model, 60).expect("Failed to initialize MuJoCo viewer");

    let initial_k_array = ANALYTIC_LQR_POLICY;

    let noise = estimate_process_noise(&model, &mut data);
    println!("============================================================");
    println!("       PROCESS NOISE DIAGNOSTIC REPORT");
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
            // Alignment is key here to see the diagonal vs off-diagonal
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

    println!("Starting [SIMULATED] 100Hz control loop...");

    let enable_noise = true;

    while viewer.running() {
        let batch_to_process = collect_full_batch_sim(
            &model,
            &mut data,
            &mut viewer,
            "LSTDQ Batch",
            computations_completed,
            &mut was_balancing,
            &pending_gains,
            &mut active_gains,
            visualize,
            enable_noise,
        );

        // Break out of the loop if the user clicked the 'X' during batch collection
        if batch_to_process.is_empty() {
            break;
        }

        computations_completed += 1;
        let k_clone = Arc::clone(&current_k);
        let pending_clone = Arc::clone(&pending_gains);

        std::thread::spawn(move || {
            let k_to_use = { *k_clone.lock().unwrap() };
            let new_k_mat = calculate_k(batch_to_process, &k_to_use, &noise.1);

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

    Ok(())
}

pub fn run_data_collection_mode_sim(visualize: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MuJoCo model 'balboa.xml'...");
    let model = MjModel::from_xml("balboa.xml").expect("Failed to load balboa.xml");
    let mut data = model.make_data();
    let data_dir = "collected_data";

    // Initialize the visualizer window
    let mut viewer =
        MjViewer::launch_passive(&model, 60).expect("Failed to initialize MuJoCo viewer");

    let mut file_index = get_next_file_index();
    let mut was_balancing = false;
    let dummy_pending_gains: Arc<Mutex<Option<[f64; 4]>>> = Arc::new(Mutex::new(None));

    let mut active_gains = ANALYTIC_LQR_POLICY;

    println!(
        "Started [SIMULATED] data collection mode. Will start at index: {}",
        file_index
    );

    while viewer.running() {
        let batch_to_process = collect_full_batch_sim(
            &model,
            &mut data,
            &mut viewer,
            "CSV Collection",
            file_index,
            &mut was_balancing,
            &dummy_pending_gains,
            &mut active_gains,
            visualize,
            true,
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
            SAMPLES_PER_ITER,
            filename,
            file_index + 1
        );
        file_index += 1;
    }
    Ok(())
}

pub fn run_sim_plot(visualize: bool) -> Result<(), Box<dyn std::error::Error>> {
    let n_policies = 2;
    let n_updates = 3;
    let policy_variance: f64 = 0.01; // Variance for the Gaussian noise

    println!("Loading MuJoCo model 'balboa.xml'...");
    let model = MjModel::from_xml("balboa.xml").expect("Failed to load balboa.xml");
    let mut data = model.make_data();
    let mut viewer = MjViewer::launch_passive(&model, 60).expect("Failed to init viewer");

    let initial_k_array = ANALYTIC_LQR_POLICY;

    let noise = estimate_process_noise(&model, &mut data);

    println!("============================================================");
    println!("       PROCESS NOISE DIAGNOSTIC REPORT");
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
            // Alignment is key here to see the diagonal vs off-diagonal
            print!("{:>15.4e} ", noise.1[(i, j)]);
        }
        println!();
    }

    println!("============================================================");

    // --- 1. Evaluate the Original Baseline Policy ---
    println!("Evaluating original baseline policy...");
    let baseline_cost = evaluate_policy_sim(&model, &mut data, initial_k_array, false);
    println!("Baseline Cost: {:.4}", baseline_cost);

    // --- 2. Generate N Policies via Gaussian Perturbation ---
    let mut rng = rng();
    // Normal takes (mean, standard_deviation). std_dev is sqrt(variance).
    let normal_dist = Normal::new(1.0f64, policy_variance.sqrt()).unwrap();

    let mut policies: Vec<[f64; 4]> = (0..n_policies)
        .map(|_| {
            [
                initial_k_array[0] * normal_dist.sample(&mut rng),
                initial_k_array[1] * normal_dist.sample(&mut rng),
                initial_k_array[2] * normal_dist.sample(&mut rng),
                initial_k_array[3] * normal_dist.sample(&mut rng),
            ]
        })
        .collect();

    // To track the empirical cost evolution: costs[policy_idx][update_idx]
    let mut cost_history: Vec<Vec<f64>> = vec![Vec::new(); n_policies];

    // --- 3. Main Loop: Evaluation & Update ---
    println!("Starting multi-policy evaluation and update loop...");

    for update_idx in 0..n_updates {
        println!("\r");
        println!("--- Update Step {} / {} ---", update_idx + 1, n_updates);

        for (p_idx, policy) in policies.iter_mut().enumerate() {
            // A) EVALUATION PHASE (Noise OFF)
            let empirical_cost = evaluate_policy_sim(&model, &mut data, *policy, false);
            cost_history[p_idx].push(empirical_cost);
            println!("  Policy {} - Eval Cost: {:.4}", p_idx, empirical_cost);

            // B) BATCH COLLECTION PHASE (Noise ON)
            let mut active_gains = *policy;
            // We still use Arc/Mutex here to satisfy your existing function signature,
            // though the background thread is no longer strictly necessary in this sequential flow.
            let pending_gains: Arc<Mutex<Option<[f64; 4]>>> = Arc::new(Mutex::new(None));
            let mut was_balancing = false;

            let batch_to_process = collect_full_batch_sim(
                &model,
                &mut data,
                &mut viewer,
                &format!("LSTDQ P{} U{}", p_idx, update_idx),
                update_idx,
                &mut was_balancing,
                &pending_gains,
                &mut active_gains,
                visualize,
                true, // enable_noise = true
            );

            if batch_to_process.is_empty() {
                println!("  [!] Batch empty, user likely exited early.");
                continue;
            }

            // C) UPDATE PHASE (Synchronous)
            // Because we need the new K for the *next* update_idx, we compute it synchronously
            // instead of spawning a disconnected thread.
            let current_k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(policy);
            let new_k_mat = calculate_k(batch_to_process, &current_k_mat, &noise.1);

            // Overwrite the current policy with the newly computed LSTDQ gains
            *policy = [
                new_k_mat[(0, 0)],
                new_k_mat[(0, 1)],
                new_k_mat[(0, 2)],
                new_k_mat[(0, 3)],
            ];
        }
    }

    // --- 4. Final Evaluation ---
    // Evaluate one last time to capture the cost *after* the final update
    for (p_idx, policy) in policies.iter().enumerate() {
        let final_cost = evaluate_policy_sim(&model, &mut data, *policy, false);
        cost_history[p_idx].push(final_cost);
    }

    // --- 5. Plotting ---
    println!("Generating plot 'policy_evolution.png'...");
    plot_cost_evolution(&cost_history, baseline_cost, n_updates)?;

    Ok(())
}

fn evaluate_policy_sim<'a>(
    model: &'a MjModel,
    data: &mut MjData<&'a MjModel>,
    policy_gains: [f64; 4],
    enable_noise: bool,
) -> f64 {
    // --- EVALUATION PARAMETERS ---
    let eval_steps = 10000; // 10 seconds of simulation at 100Hz
    let control_step: f64 = 0.01;
    let timestep: f64 = model.opt().timestep;
    let sim_steps = (control_step / timestep).round() as usize;
    let stop_angle_rad = 60.0_f64.to_radians();

    // --- LQR COST WEIGHTS ---
    // Q = diag([10.0, 100.0, 1.0, 10.0])
    let q_phi = 10.0;
    let q_theta = 100.0;
    let q_phi_dot = 1.0;
    let q_theta_dot = 10.0;
    // R = [[300.0]]
    let r_tau = 300.0;

    // --- RNG INITIALIZATION ---
    // Seed the RNG using the globally defined constant SEED (e.g., const SEED: u64 = 42;)
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    // --- RESET SIMULATION ---
    data.reset();
    data.qpos_mut()[2] = 0.05; // Starting elevation

    // 1. Randomize initial pitch (Theta) between +/- 10 degrees (~0.1745 rad)
    let initial_theta: f64 = rng.random_range(-0.1745..0.1745);

    // 2. Convert to valid Quaternion for Y-axis rotation
    data.qpos_mut()[3] = (initial_theta / 2.0).cos(); // w
    data.qpos_mut()[4] = 0.0; // x
    data.qpos_mut()[5] = (initial_theta / 2.0).sin(); // y
    data.qpos_mut()[6] = 0.0; // z

    // 3. Randomize initial pitch velocity (Theta_dot)
    let initial_theta_dot: f64 = rng.random_range(-0.1..0.1);
    data.qvel_mut()[4] = initial_theta_dot;

    let mut total_cost = 0.0;

    // OU Noise setup
    let mut last_noise: f64 = 0.0;

    for step in 0..eval_steps {
        // --- 1. EXTRACT RAW STATE VIA HELPER ---
        let (phi_left, phi_right, theta, phi_dot_left, phi_dot_right, theta_dot) =
            extract_state(data, BACKLASH_JOINTS);

        // --- TRANSFORM INTO LQR STATE ---
        let phi = (phi_left + phi_right) / 2.0;
        let phi_dot = (phi_dot_left + phi_dot_right) / 2.0;

        // --- 2. STABILITY CHECK ---
        let is_sane = theta.is_finite() && theta_dot.abs() < 100.0;
        let is_upright = theta.abs() < stop_angle_rad;

        if !is_sane || !is_upright {
            // If the robot falls, heavily penalize the remaining timesteps
            // to aggressively filter out unstable policies.
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
            // Use the seeded RNG for deterministic noise generation across runs
            let u1: f64 = rng.random_range(0.0001..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let epsilon = (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * PI * u2).cos();
            let dx = THETA_OU * (-last_noise) * 0.01 + SIGMA_OU * epsilon * 0.1;
            last_noise += dx;
        } else {
            last_noise = 0.0;
        }

        let tau_total = u_raw + last_noise;
        let raw_tau = tau_total / 2.0;

        // --- 5. EMPIRICAL COST CALCULATION ---
        // J = x^T Q x + u^T R u
        let state_cost = q_phi * phi.powi(2)
            + q_theta * theta.powi(2)
            + q_phi_dot * phi_dot.powi(2)
            + q_theta_dot * theta_dot.powi(2);

        let action_cost = r_tau * raw_tau.powi(2);

        total_cost += state_cost + action_cost;

        // --- 6. MOTOR REALISM (Quantization & Limits) ---
        let raw_tau_offset = if phi_dot > 0.0f64 {
            raw_tau + 0.00
        } else {
            raw_tau - 0.00
        };

        let max_physical_torque = 0.1;
        let pwm_resolution = 400.0;
        let max_speed = 25.0;

        let avail_l = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_left.abs() / max_speed)));
        let avail_r = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_right.abs() / max_speed)));

        let pwm_l_raw = (raw_tau_offset / max_physical_torque) * pwm_resolution;
        let pwm_r_raw = (raw_tau_offset / max_physical_torque) * pwm_resolution;

        let pwm_l_quantized = pwm_l_raw.round();
        let pwm_r_quantized = pwm_r_raw.round();

        let tau_l_quantized = (pwm_l_quantized / pwm_resolution) * max_physical_torque;
        let tau_r_quantized = (pwm_r_quantized / pwm_resolution) * max_physical_torque;

        let tau_l_final = tau_l_quantized.clamp(-avail_l, avail_l);
        let tau_r_final = tau_r_quantized.clamp(-avail_r, avail_r);

        data.ctrl_mut()[0] = tau_l_final;
        data.ctrl_mut()[1] = tau_r_final;

        // --- 7. STEP SIMULATION ---
        for _ in 0..sim_steps {
            data.step();
        }
    }

    // Return the averaged cost across all timesteps
    total_cost / (eval_steps as f64)
}

// --- 1. CALCULATE ANALYTICAL A AND B MATRICES ---
/*
let mw: f64 = 0.0042;
let mp: f64 = 0.316;
let r: f64 = 0.040;
let l: f64 = 0.023;
let ip: f64 = 444.43e-6;
let iw: f64 = 26.89e-6;
let g: f64 = 9.81;
*/

pub fn estimate_process_noise<'a>(
    model: &'a MjModel,
    data: &mut MjData<&'a MjModel>,
) -> (SVector<f64, DIM_X>, SMatrix<f64, DIM_X, DIM_X>) {
    // --- EVALUATION PARAMETERS ---
    let eval_steps = 10000;
    let control_step: f64 = 0.01;
    let timestep: f64 = model.opt().timestep;
    let sim_steps = (control_step / timestep).round() as usize;

    // --- 1. CALCULATE ANALYTICAL A AND B MATRICES ---
    let mw: f64 = 0.032;
    let mp: f64 = 0.317;
    let r: f64 = 0.040;
    let l: f64 = 0.023;
    let ip: f64 = 444.43e-6;
    let iw: f64 = 0.004027;
    let g: f64 = 9.81;

    // E Matrix components
    let e00 = iw + (r.powi(2)) * (mw + mp);
    let e01 = mp * r * l;
    let e10 = mp * r * l;
    let e11 = ip + mp * (l.powi(2));

    // Invert E manually for a 2x2
    let det = e00 * e11 - e01 * e10;
    let inv_det = 1.0 / det;
    let e_inv_00 = e11 * inv_det;
    let e_inv_01 = -e01 * inv_det;
    let e_inv_10 = -e10 * inv_det;
    let e_inv_11 = e00 * inv_det;

    // G and F Vectors
    let g_vec = [0.0, -mp * g * l];
    let f_vec = [1.0, 0.0];

    // Construct A matrix directly as an nalgebra SMatrix
    let mut a_mat = SMatrix::<f64, DIM_X, DIM_X>::zeros();
    a_mat[(0, 2)] = 1.0;
    a_mat[(1, 3)] = 1.0;
    a_mat[(2, 1)] = -(e_inv_00 * g_vec[0] + e_inv_01 * g_vec[1]);
    a_mat[(3, 1)] = -(e_inv_10 * g_vec[0] + e_inv_11 * g_vec[1]);

    // Construct B matrix directly as an nalgebra SMatrix
    let mut b_mat = SMatrix::<f64, DIM_X, DIM_U>::zeros();
    b_mat[(2, 0)] = e_inv_00 * f_vec[0] + e_inv_01 * f_vec[1];
    b_mat[(3, 0)] = e_inv_10 * f_vec[0] + e_inv_11 * f_vec[1];

    // --- 2. CALCULATE OPTIMAL LQR GAIN ---
    let q_cost = SMatrix::<f64, DIM_X, DIM_X>::from_diagonal(&SVector::from([
        10.0,  // phi penalty
        100.0, // theta penalty
        1.0,   // phi_dot penalty
        10.0,  // theta_dot penalty
    ]));
    let r_cost = SMatrix::<f64, DIM_U, DIM_U>::from_diagonal(&SVector::from([300.0]));

    // Discretize A and B for the 100Hz controller
    let a_d = SMatrix::<f64, DIM_X, DIM_X>::identity() + a_mat * control_step;
    let b_d = b_mat * control_step;

    // Iterative DARE Solver (Discrete Algebraic Riccati Equation)
    let mut p_mat = q_cost;
    for _ in 0..1000 {
        let r_plus_bt_p_b = r_cost + b_d.transpose() * p_mat * b_d;
        let inv_term = r_plus_bt_p_b
            .try_inverse()
            .expect("DARE: R matrix inversion failed");
        let p_next = q_cost + a_d.transpose() * p_mat * a_d
            - a_d.transpose() * p_mat * b_d * inv_term * b_d.transpose() * p_mat * a_d;

        // Check for convergence
        if (p_next - p_mat).norm() < 1e-7 {
            p_mat = p_next;
            break;
        }
        p_mat = p_next;
    }

    // Calculate final K gain
    let inv_term = (r_cost + b_d.transpose() * p_mat * b_d)
        .try_inverse()
        .unwrap();
    let k_matrix = -(inv_term * b_d.transpose() * p_mat * a_d);

    println!("\n============================================================");
    println!("        ANALYTICAL LQR GAIN CALCULATION (100Hz)");
    println!("============================================================");
    println!("Optimal K Matrix (Control Law: u = Kx):");
    println!("  K_phi       = {:>10.4}", k_matrix[(0, 0)]);
    println!("  K_theta     = {:>10.4}", k_matrix[(0, 1)]);
    println!("  K_phi_dot   = {:>10.4}", k_matrix[(0, 2)]);
    println!("  K_theta_dot = {:>10.4}", k_matrix[(0, 3)]);
    println!("============================================================\n");

    // --- 3. RESET SIMULATION & INJECT SEEDED RANDOM STATE ---
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    data.reset();
    data.qpos_mut()[2] = 0.05;

    let initial_theta: f64 = rng.random_range(-0.1745..0.1745);
    data.qpos_mut()[3] = (initial_theta / 2.0).cos();
    data.qpos_mut()[4] = 0.0;
    data.qpos_mut()[5] = (initial_theta / 2.0).sin();
    data.qpos_mut()[6] = 0.0;

    let initial_theta_dot: f64 = rng.random_range(-0.1..0.1);
    data.qvel_mut()[4] = initial_theta_dot;

    let mut noises: Vec<SVector<f64, DIM_X>> = Vec::with_capacity(eval_steps);

    // --- 4. RUN SIMULATION & COLLECT NOISE SAMPLES ---
    for _ in 0..eval_steps {
        // Extract raw states via helper
        let (phi_left, phi_right, theta, phi_dot_left, phi_dot_right, theta_dot) =
            extract_state(data, BACKLASH_JOINTS);

        // Transform into LQR representation
        let phi = (phi_left + phi_right) / 2.0;
        let phi_dot = (phi_dot_left + phi_dot_right) / 2.0;
        let x_k = SVector::<f64, DIM_X>::from_column_slice(&[phi, theta, phi_dot, theta_dot]);

        // LQR Control Law (u = K * x)
        let u_mat = k_matrix * x_k;
        let u_raw = u_mat[(0, 0)];

        let raw_tau = u_raw / 2.0;

        // Motor Realism (Quantization & Limits)
        let max_physical_torque = 0.1;
        let pwm_resolution = 400.0;
        let max_speed = 25.0;

        let avail_l = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_left.abs() / max_speed)));
        let avail_r = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_right.abs() / max_speed)));

        let pwm_l_raw = (raw_tau / max_physical_torque) * pwm_resolution;
        let pwm_r_raw = (raw_tau / max_physical_torque) * pwm_resolution;

        let tau_l_final =
            ((pwm_l_raw.round() / pwm_resolution) * max_physical_torque).clamp(-avail_l, avail_l);
        let tau_r_final =
            ((pwm_r_raw.round() / pwm_resolution) * max_physical_torque).clamp(-avail_r, avail_r);

        data.ctrl_mut()[0] = tau_l_final;
        data.ctrl_mut()[1] = tau_r_final;

        let u_applied = tau_l_final + tau_r_final;

        // Step Simulation
        for _ in 0..sim_steps {
            data.step();
        }

        // Extract State k+1 via helper & Transform
        let (phi_left, phi_right, theta, phi_dot_left, phi_dot_right, theta_dot) =
            extract_state(data, BACKLASH_JOINTS);

        let phi = (phi_left + phi_right) / 2.0;
        let phi_dot = (phi_dot_left + phi_dot_right) / 2.0;
        let x_k1 = SVector::<f64, DIM_X>::from_column_slice(&[phi, theta, phi_dot, theta_dot]);

        // --- 5. CALCULATE PROCESS NOISE ---
        let x_dot_empirical = (x_k1 - x_k) / control_step;

        let u_applied_vec = SVector::<f64, DIM_U>::from_column_slice(&[u_applied]);
        let x_dot_theory = (a_mat * x_k) + (b_mat * u_applied_vec);

        let n = x_dot_empirical - x_dot_theory;
        noises.push(n);
    }

    // --- 6. COMPUTE EMPIRICAL MEAN & COVARIANCE ---
    let mut mean = SVector::<f64, DIM_X>::zeros();
    for n in &noises {
        mean += n;
    }
    mean /= eval_steps as f64;

    let mut covariance = SMatrix::<f64, DIM_X, DIM_X>::zeros();
    for n in &noises {
        let diff = n - mean;
        covariance += diff * diff.transpose();
    }
    covariance /= (eval_steps - 1) as f64;

    (mean, covariance)
}

// Helper function to extract raw state cleanly, supporting both XML models
fn extract_state<'a>(
    data: &MjData<&'a MjModel>,
    has_backlash: bool,
) -> (f64, f64, f64, f64, f64, f64) {
    let qpos = data.qpos();
    let qvel = data.qvel();

    // --- 1. Pitch (Theta) Extraction ---
    let qw = qpos[3];
    let qy = qpos[5];
    let theta = 2.0 * qy.atan2(qw);
    let theta_dot = qvel[4];

    // --- INDEX ROUTING ---
    let right_qpos_idx = if has_backlash { 9 } else { 8 };
    let right_qvel_idx = if has_backlash { 8 } else { 7 };

    // --- 2. Wheel Position & Velocity Extraction ---
    let phi_left = qpos[7];
    let phi_right = qpos[right_qpos_idx];

    let phi_dot_left = qvel[6];
    let phi_dot_right = qvel[right_qvel_idx];

    (
        phi_left,
        phi_right,
        theta,
        phi_dot_left,
        phi_dot_right,
        theta_dot,
    )
}
