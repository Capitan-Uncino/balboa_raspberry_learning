use crate::learning::lstdq::{calculate_k, StateAction, ANALYTIC_LQR_POLICY, SAMPLES_PER_ITER};
use crate::utils::file_utils::get_next_file_index;
use crate::utils::graphic_utils::{log_progress, plot_cost_evolution};
use mujoco_rs::prelude::*;
use mujoco_rs::viewer::MjViewer;
use rand::rng;
use rand::RngExt;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const THETA_OU: f64 = 0.30;
const SIGMA_OU: f64 = 0.10;

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

    // --- OU NOISE PARAMETERS ---
    let mut last_noise: f64 = 0.0;
    let mut rng = rng();

    // Calculate how much real time should pass per control step
    let target_duration = Duration::from_secs_f64(control_step);

    println!(">>> [SIM] Running MuJoCo Simulation Loop...");

    // Only check if the viewer is running if rendering is actually enabled
    while state_batch.len() < SAMPLES_PER_ITER && (!enable_rendering || viewer.running()) {
        // Start the timer for real-time pacing
        let step_start = Instant::now();

        if let Some(new_gains) = pending_gains.lock().unwrap().take() {
            println!("---> [SIM] Applying New Gains: {:?}", new_gains);
            *active_gains = new_gains;
        }

        let qpos = data.qpos();
        let qvel = data.qvel();

        // --- 1. Pitch (Theta) Extraction ---
        // Using atan2 is much more stable than asin for balancing robots.
        // It handles the full 360-degree range and avoids NaN errors.
        let qw = qpos[3];
        let qy = qpos[5];
        let theta = 2.0 * qy.atan2(qw);
        let theta_dot = qvel[4]; // Pitch velocity (Y-axis angular velocity)

        // --- 2. Wheel Position (Phi) Extraction ---
        // Use the motor joints (7 and 9), not the backlash joints.
        let phi_left = qpos[7];
        let phi_right = qpos[9]; // Fixed index
        let phi = (phi_left + phi_right) / 2.0;

        // --- 3. Wheel Velocity (Phi_dot) Extraction ---
        let phi_dot_left = qvel[6];
        let phi_dot_right = qvel[8]; // Fixed index
        let phi_dot = (phi_dot_left + phi_dot_right) / 2.0;

        let k1 = active_gains[0];
        let k2 = active_gains[1];
        let k3 = active_gains[2];
        let k4 = active_gains[3];

        // 1. LQR Control Law: u = -Kx
        // (Note the negative sign at the front!)
        let u_raw = k1 * phi + k2 * theta + k3 * phi_dot + k4 * theta_dot;

        // ==========================================
        // OU EXPLORATION NOISE
        // ==========================================
        let u1: f64 = rng.random_range(0.0001..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let epsilon = (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * PI * u2).cos();
        let dx = THETA_OU * (-last_noise) * 0.01 + SIGMA_OU * epsilon * 0.1;
        last_noise += dx;

        // --- Constants (adjust based on your setup) ---
        let max_physical_torque = 0.1;
        let pwm_resolution = 400.0; // The Balboa 32U4 uses 400 for max speed

        // 2. Add noise
        let tau_total = u_raw + last_noise;

        // 3. Split load
        let raw_tau = tau_total / 2.0;

        let raw_tau_offset = if phi_dot > 0.0f64 {
            raw_tau + 0.00
        } else {
            raw_tau - 0.00
        };

        // 4. Back-EMF constraints (as before)
        let max_speed = 25.0;
        let avail_l = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_left.abs() / max_speed)));
        let avail_r = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_right.abs() / max_speed)));

        // 5. Quantization Step (Simulating the 8-bit/10-bit PWM resolution)
        // First, normalize the requested torque to a -400 to 400 scale
        let pwm_l_raw = (raw_tau_offset / max_physical_torque) * pwm_resolution;
        let pwm_r_raw = (raw_tau_offset / max_physical_torque) * pwm_resolution;

        // Round to the nearest integer step (the actual "quantization")
        let pwm_l_quantized = pwm_l_raw.round();
        let pwm_r_quantized = pwm_r_raw.round();

        // Convert back to physical Torque (Nm) for MuJoCo
        let tau_l_quantized = (pwm_l_quantized / pwm_resolution) * max_physical_torque;
        let tau_r_quantized = (pwm_r_quantized / pwm_resolution) * max_physical_torque;

        // 6. Final Clamp (Apply the dynamic Back-EMF limits)
        let tau_l_final = tau_l_quantized.clamp(-avail_l, avail_l);
        let tau_r_final = tau_r_quantized.clamp(-avail_r, avail_r);

        // 7. Apply to actuators
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

                // Safe Reset (Quaternion reconstruction is still required)
                data.reset();
                data.qpos_mut()[2] = 0.05;
                data.qpos_mut()[3] = 1.0; // W must be 1.0!
                                          // We don't need to manually tilt it anymore; the OU noise will do it!

                // Reset the noise history so it doesn't jump aggressively on respawn
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

    Ok(())
}

pub fn run_data_collection_mode_sim(visualize: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MuJoCo model 'balboa.xml'...");
    let model = MjModel::from_xml("balboa.xml").expect("Failed to load balboa.xml");
    let mut data = model.make_data();

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

        let filename = format!("batch_{}.csv", file_index);
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
            let new_k_mat = calculate_k(batch_to_process, &current_k_mat);

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
    let eval_steps = 1000; // 10 seconds of simulation at 100Hz
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

    // --- RESET SIMULATION ---
    // Start from a clean, upright state for a fair evaluation of the policy
    data.reset();
    data.qpos_mut()[2] = 0.05; // Starting elevation
    data.qpos_mut()[3] = 1.0; // W component of quaternion must be 1.0

    let mut total_cost = 0.0;

    // OU Noise setup
    let mut last_noise: f64 = 0.0;
    let mut rng = rng();

    for step in 0..eval_steps {
        // --- 1. EXTRACT STATE ---
        let qpos = data.qpos();
        let qvel = data.qvel();

        // --- 1. Pitch (Theta) Extraction ---
        // Using atan2 is much more stable than asin for balancing robots.
        // It handles the full 360-degree range and avoids NaN errors.
        let qw = qpos[3];
        let qy = qpos[5];
        let theta = 2.0 * qy.atan2(qw);
        let theta_dot = qvel[4]; // Pitch velocity (Y-axis angular velocity)

        // --- 2. Wheel Position (Phi) Extraction ---
        // Use the motor joints (7 and 9), not the backlash joints.
        let phi_left = qpos[7];
        let phi_right = qpos[9]; // Fixed index
        let phi = (phi_left + phi_right) / 2.0;

        // --- 3. Wheel Velocity (Phi_dot) Extraction ---
        let phi_dot_left = qvel[6];
        let phi_dot_right = qvel[8]; // Fixed index
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
