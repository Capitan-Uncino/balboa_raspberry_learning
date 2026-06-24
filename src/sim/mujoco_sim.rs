use crate::file_utils::get_next_file_index;
use crate::graphic_utils::plot_cost_evolution;
use crate::learning::sysid_lqr::{
    calculate_k, StateAction, ANALYTIC_LQR_POLICY, DIM_U, DIM_X, SAMPLES_PER_ITER,
};
use crate::logging_utils::log_progress;
use mujoco_rs::prelude::*;
use mujoco_rs::viewer::MjViewer;
use nalgebra::{SMatrix, SVector};
use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const BACKLASH_ESTIMATION: bool = false;
const DEADZONE_EPSILON: f64 = 1.5 * std::f64::consts::PI / 180.0;

const THETA_OU: f64 = 0.60;
const SIGMA_OU: f64 = 0.30;
const SEED: u64 = 42;
const BACKLASH_JOINTS: bool = false;
const MAX_FALLS: usize = 20;

pub fn run_online_mode_sim(visualize: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MuJoCo model 'balboa.xml'...");

    // mujoco-rs uses MjModel::from_xml for loading files
    let model = if BACKLASH_JOINTS {
        MjModel::from_xml("balboa_delay.xml").expect("Failed to load balboa.xml")
    } else {
        MjModel::from_xml("balboa.xml").expect("Failed to load balboa.xml")
    };

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

    let initial_k_mat =
        nalgebra::SMatrix::<f64, 1, DIM_X>::from_iterator(initial_k_array.into_iter());

    let current_k = Arc::new(Mutex::new(initial_k_mat));
    let pending_gains: Arc<Mutex<Option<[f64; DIM_X]>>> = Arc::new(Mutex::new(None));
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
            let new_k_mat = calculate_k(&batch_to_process, &k_to_use);

            {
                *k_clone.lock().unwrap() = new_k_mat;
            }

            // Dynamically construct the array for pending_gains
            let new_k_array: [f64; DIM_X] = std::array::from_fn(|i| new_k_mat[(0, i)]);

            *pending_clone.lock().unwrap() = Some(new_k_array);
            println!(">>> LSTDQ Update: New K vector queued for next SIM window.");
        });
    }

    Ok(())
}

pub fn run_data_collection_mode_sim(visualize: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MuJoCo model 'balboa.xml'...");
    let model = if BACKLASH_JOINTS {
        MjModel::from_xml("balboa_delay.xml").expect("Failed to load balboa.xml")
    } else {
        MjModel::from_xml("balboa.xml").expect("Failed to load balboa.xml")
    };
    let mut data = model.make_data();
    let data_dir = "collected_data";

    // Initialize the visualizer window
    let mut viewer =
        MjViewer::launch_passive(&model, 60).expect("Failed to initialize MuJoCo viewer");

    let mut file_index = get_next_file_index();
    let mut was_balancing = false;
    let dummy_pending_gains: Arc<Mutex<Option<[f64; DIM_X]>>> = Arc::new(Mutex::new(None));

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

pub fn run_sim_plot(
    visualize: bool,
    evaluation_threshold: f64,
    n_policies: usize,
    n_updates: usize,
    uniform_half_interval: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MuJoCo model 'balboa.xml'...");
    let model = if BACKLASH_JOINTS {
        MjModel::from_xml("balboa_delay.xml").expect("Failed to load balboa.xml")
    } else {
        MjModel::from_xml("balboa.xml").expect("Failed to load balboa.xml")
    };
    let mut data = model.make_data();
    let mut viewer = MjViewer::launch_passive(&model, 60).expect("Failed to init viewer");

    let initial_k_array = ANALYTIC_LQR_POLICY;

    let noise = estimate_process_noise(&model, &mut data);

    println!("============================================================");
    println!("        PROCESS NOISE DIAGNOSTIC REPORT");
    println!("============================================================");

    // 1. Print the Mean Vector dynamically
    println!("MEAN VECTOR (Systematic Bias / Residuals):");
    if BACKLASH_ESTIMATION {
        println!("  [ φ_dot,     θ_dot,     φ_ddot,    θ_ddot,    backlash_dot ]");
        println!(
            "  [{:+.4e}, {:+.4e}, {:+.4e}, {:+.4e}, {:+.4e}]",
            noise.0[0], noise.0[1], noise.0[2], noise.0[3], noise.0[4]
        );
    } else {
        println!("  [ φ_dot,     θ_dot,     φ_ddot,    θ_ddot ]");
        println!(
            "  [{:+.4e}, {:+.4e}, {:+.4e}, {:+.4e}]",
            noise.0[0], noise.0[1], noise.0[2], noise.0[3]
        );
    }

    // 2. Print the Covariance Matrix dynamically
    println!("\nCOVARIANCE MATRIX (Sigma):");
    let labels = if BACKLASH_ESTIMATION {
        vec![
            "φ_dot       ",
            "θ_dot       ",
            "φ_ddot      ",
            "θ_ddot      ",
            "backlash_dot",
        ]
    } else {
        vec!["φ_dot ", "θ_dot ", "φ_ddot", "θ_ddot"]
    };

    // Print header row
    print!("            ");
    for label in &labels {
        print!("{:<15}", label.trim());
    }
    println!();

    for i in 0..DIM_X {
        print!("{} ", labels[i]);
        for j in 0..DIM_X {
            print!("{:>15.4e} ", noise.1[(i, j)]);
        }
        println!();
    }
    println!("============================================================");

    // --- 1. Evaluate the Original Baseline Policy ---
    println!("Evaluating original baseline policy...");
    let baseline_cost = evaluate_policy_sim(&model, &mut data, initial_k_array, false);
    println!("Baseline Cost: {:.4}", baseline_cost);

    // --- 2. Generate N Policies via Uniform Perturbation ---
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let uniform_dist = rand::distr::Uniform::new_inclusive(
        1.0f64 - uniform_half_interval,
        1.0f64 + uniform_half_interval,
    )
    .unwrap();

    // Dynamically size the perturbation using from_fn
    let mut active_policies: Vec<(usize, [f64; DIM_X])> = (0..n_policies)
        .map(|id| {
            let p: [f64; DIM_X] = std::array::from_fn(|i| {
                initial_k_array[i] * rand::distr::Distribution::sample(&uniform_dist, &mut rng)
            });
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
            let empirical_cost = evaluate_policy_sim(&model, &mut data, policy, false);
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
            let pending_gains: std::sync::Arc<std::sync::Mutex<Option<[f64; DIM_X]>>> =
                std::sync::Arc::new(std::sync::Mutex::new(None));
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
                true,
            );

            if batch_to_process.is_empty() {
                println!(
                    "  [!] Batch empty (too many falls or early exit). Discarding policy {}.",
                    p_idx
                );
                continue; // Do NOT push to next_active_policies
            }

            // C) UPDATE PHASE (Synchronous)
            let current_k_mat = nalgebra::SMatrix::<f64, 1, DIM_X>::from_row_slice(&policy);
            let new_k_mat = calculate_k(&batch_to_process, &current_k_mat);

            // Dynamically extract the new gains back into the array
            policy = std::array::from_fn(|i| new_k_mat[(0, i)]);

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
        let final_cost = evaluate_policy_sim(&model, &mut data, *policy, false);
        cost_history[*p_idx].push(final_cost);
    }

    // --- 4.5 Print Surviving Policies Table ---
    println!("\n===================================================================================================================");
    println!("                                   FINAL SURVIVING POLICIES REPORT                                                 ");
    println!("===================================================================================================================");

    if BACKLASH_ESTIMATION {
        println!("| Policy ID | Final Cost |   K1 (φ)        |   K2 (θ)        |   K3 (φ_dot)    |   K4 (θ_dot)    |   K5 (bklsh)    |");
        println!("|-----------|------------|-----------------|-----------------|-----------------|-----------------|-----------------|");
    } else {
        println!("| Policy ID | Final Cost |   K1 (φ)        |   K2 (θ)        |   K3 (φ_dot)    |   K4 (θ_dot)    |");
        println!("|-----------|------------|-----------------|-----------------|-----------------|-----------------|");
    }

    if active_policies.is_empty() {
        println!("|                           No policies survived the evaluation threshold.                                        |");
    } else {
        for (p_idx, policy) in active_policies.iter() {
            let final_cost = cost_history[*p_idx].last().unwrap_or(&f64::NAN);

            if BACKLASH_ESTIMATION {
                println!(
                    "| {:^9} | {:^10.4} | {:>15.4} | {:>15.4} | {:>15.4} | {:>15.4} | {:>15.4} |",
                    p_idx, final_cost, policy[0], policy[1], policy[2], policy[3], policy[4]
                );
            } else {
                println!(
                    "| {:^9} | {:^10.4} | {:>15.4} | {:>15.4} | {:>15.4} | {:>15.4} |",
                    p_idx, final_cost, policy[0], policy[1], policy[2], policy[3]
                );
            }
        }
    }
    println!("===================================================================================================================\n");

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

// Helper function to extract raw state cleanly, supporting both XML models
fn extract_state(data: &MjData<&MjModel>, has_backlash: bool) -> (f64, f64, f64, f64, f64, f64) {
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

pub enum SimTask<'v, 't> {
    CollectBatch {
        viewer: Option<&'v mut MjViewer>,
        log_label: &'t str,
        batch_index: usize,
        was_balancing: &'t mut bool,
        pending_gains: &'t Arc<Mutex<Option<[f64; DIM_X]>>>,
        active_gains: &'t mut [f64; DIM_X],
        enable_rendering: bool,
    },
    EvaluatePolicy {
        policy_gains: [f64; DIM_X],
    },
    EstimateProcessNoise,
}
pub enum SimResult {
    Batch(Vec<StateAction>),
    EvaluationCost(f64),
    NoiseData(SVector<f64, DIM_X>, SMatrix<f64, DIM_X, DIM_X>),
}

/// Bundles the tracking state so we don't pass 10 mutable arguments
#[derive(Default)]
struct SimTracker {
    state_batch: Vec<StateAction>,
    noises: Vec<SVector<f64, DIM_X>>,
    total_cost: f64,
    loop_counter: usize,
    stability_counter: usize,
    fall_count: usize,
    last_noise: f64,
}

/// Instructions for the main loop after evaluating a step
enum StepAction {
    Continue,
    Break,
    AbortBatch,
}

pub fn unified_sim_loop<'a, 'v, 't>(
    model: &'a MjModel,
    data: &mut MjData<&'a MjModel>,
    mut task: SimTask<'v, 't>,
    enable_noise: bool,
) -> SimResult {
    let control_step = 0.01;
    let sim_steps = (control_step / model.opt().timestep).round() as usize;
    let max_steps = match &task {
        SimTask::CollectBatch { .. } => SAMPLES_PER_ITER,
        SimTask::EvaluatePolicy { .. } | SimTask::EstimateProcessNoise => SAMPLES_PER_ITER / 10,
    };

    let mut tracker = SimTracker::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    // 1. Setup initial conditions & matrices
    let (analytical_k, a_mat, b_mat) = setup_simulation(data, &task, &mut rng);

    println!(">>> [SIM] Running MuJoCo Simulation Loop...");

    // --- NEW: Initialize tracking variables for the Hysteresis Estimator ---
    // Read the initial state to prime the tracking variables
    let (init_phi_left, init_phi_right, init_theta, _, _, _) = extract_state(data, BACKLASH_JOINTS);
    let mut last_phi = (init_phi_left + init_phi_right) / 2.0;
    let mut last_theta = init_theta;
    let mut current_backlash = 0.0;

    // Note: Ensure DEADZONE_EPSILON is in scope (e.g., 1.5 * PI / 180.0)
    const JUMP_THRESHOLD: f64 = 1.0; // Rads. Used to detect MuJoCo environment resets.

    // 2. Main Simulation Loop
    for current_step in 0..max_steps {
        if should_exit_early(&task, &tracker) {
            break;
        }

        let step_start = Instant::now();

        // Extract State
        let (phi_left, phi_right, theta, phi_dot_left, phi_dot_right, theta_dot) =
            extract_state(data, BACKLASH_JOINTS);
        let phi = (phi_left + phi_right) / 2.0;
        let phi_dot = (phi_dot_left + phi_dot_right) / 2.0;

        // --- NEW: Estimate Backlash via Play Operator ---
        // 1. Check if MuJoCo reset the robot's pose (discontinuity)
        if (phi - last_phi).abs() > JUMP_THRESHOLD || (theta - last_theta).abs() > JUMP_THRESHOLD {
            current_backlash = 0.0; // Reset estimator
        } else {
            // 2. Calculate the absolute rotation of the motor shaft in the world
            let delta_abs_motor = (phi - last_phi) + (theta - last_theta);

            // 3. Integrate and clamp to physical limits
            current_backlash =
                (current_backlash + delta_abs_motor).clamp(-DEADZONE_EPSILON, DEADZONE_EPSILON);
        }

        // Update tracking variables for the next step
        last_phi = phi;
        last_theta = theta;

        // Build state vector dynamically
        let mut x_k = SVector::<f64, DIM_X>::zeros();
        x_k[0] = phi;
        x_k[1] = theta;
        x_k[2] = phi_dot;
        x_k[3] = theta_dot;
        if BACKLASH_ESTIMATION {
            x_k[4] = current_backlash;
        }

        // Compute Control & Physics
        let u_raw = compute_control_effort(&mut task, &x_k, &analytical_k);

        let (raw_tau, u_applied) = apply_motor_physics(
            data,
            u_raw,
            phi_dot,
            phi_dot_left,
            phi_dot_right,
            enable_noise,
            &mut rng,
            &mut tracker.last_noise,
        );

        // Step Physics
        for _ in 0..sim_steps {
            data.step();
        }

        // Process Results (Metrics, Logging, Falls, Resets)
        let action = process_step_result(
            data,
            &mut task,
            &mut tracker,
            step_start,
            &x_k,
            u_applied,
            raw_tau,
            current_step,
            max_steps,
            &a_mat,
            &b_mat,
        );

        match action {
            StepAction::Break => break,
            StepAction::AbortBatch => return SimResult::Batch(Vec::new()),
            StepAction::Continue => {}
        }
    }

    // 3. Finalize and Return
    finalize_results(task, tracker, max_steps)
}

// ==========================================
// HELPER FUNCTIONS
// ==========================================

fn calculate_discrete_lqr(
    a_mat: &SMatrix<f64, 4, 4>,
    b_mat: &SMatrix<f64, 4, DIM_U>,
    control_step: f64,
) -> SMatrix<f64, DIM_U, 4> {
    // 1. Define strictly 4D LQR Cost Weights
    let q_cost = SMatrix::<f64, 4, 4>::from_diagonal(&SVector::from([
        10.0,  // phi penalty
        100.0, // theta penalty
        0.0,   // phi_dot penalty
        0.1,   // theta_dot penalty
    ]));
    let r_cost = SMatrix::<f64, DIM_U, DIM_U>::from_diagonal(&SVector::from([3.0]));

    // 2. Discretize 4D A and B
    let a_d = SMatrix::<f64, 4, 4>::identity() + a_mat * control_step;
    let b_d = b_mat * control_step;

    // 3. Iterative DARE Solver (4x4)
    let mut p_mat = q_cost;
    for _ in 0..1000 {
        let r_plus_bt_p_b = r_cost + b_d.transpose() * p_mat * b_d;
        let inv_term = r_plus_bt_p_b
            .try_inverse()
            .expect("DARE: R matrix inversion failed");

        let p_next = q_cost + a_d.transpose() * p_mat * a_d
            - a_d.transpose() * p_mat * b_d * inv_term * b_d.transpose() * p_mat * a_d;

        if (p_next - p_mat).norm() < 1e-7 {
            p_mat = p_next;
            break;
        }
        p_mat = p_next;
    }

    let inv_term = (r_cost + b_d.transpose() * p_mat * b_d)
        .try_inverse()
        .unwrap();
    -(inv_term * b_d.transpose() * p_mat * a_d)
}

fn setup_simulation<'a>(
    data: &mut MjData<&'a MjModel>,
    task: &SimTask,
    rng: &mut StdRng,
) -> (
    SMatrix<f64, DIM_U, DIM_X>,
    SMatrix<f64, DIM_X, DIM_X>,
    SMatrix<f64, DIM_X, DIM_U>,
) {
    // These are the dynamically-sized matrices we return to the main loop
    let mut analytical_k_full = SMatrix::<f64, DIM_U, DIM_X>::zeros();
    let mut a_mat_full = SMatrix::<f64, DIM_X, DIM_X>::zeros();
    let mut b_mat_full = SMatrix::<f64, DIM_X, DIM_U>::zeros();

    match task {
        SimTask::EstimateProcessNoise => {
            let control_step: f64 = 0.01;

            // --- 1. CALCULATE STRICTLY 4D ANALYTICAL MATRICES ---
            let mw: f64 = 0.032;
            let mp: f64 = 0.317;
            let r: f64 = 0.040;
            let l: f64 = 0.023;
            let ip: f64 = 444.43e-6;
            let iw: f64 = 0.004027;
            let g: f64 = 9.81;

            let e00 = iw + (r.powi(2)) * (mw + mp);
            let e01 = mp * r * l;
            let e10 = mp * r * l;
            let e11 = ip + mp * (l.powi(2));

            let det = e00 * e11 - e01 * e10;
            let inv_det = 1.0 / det;
            let e_inv_00 = e11 * inv_det;
            let e_inv_01 = -e01 * inv_det;
            let e_inv_10 = -e10 * inv_det;
            let e_inv_11 = e00 * inv_det;

            let g_vec = [0.0, -mp * g * l];
            let f_vec = [1.0, 0.0];

            let mut a_mat_4d = SMatrix::<f64, 4, 4>::zeros();
            a_mat_4d[(0, 2)] = 1.0;
            a_mat_4d[(1, 3)] = 1.0;
            a_mat_4d[(2, 1)] = -(e_inv_00 * g_vec[0] + e_inv_01 * g_vec[1]);
            a_mat_4d[(3, 1)] = -(e_inv_10 * g_vec[0] + e_inv_11 * g_vec[1]);

            let mut b_mat_4d = SMatrix::<f64, 4, DIM_U>::zeros();
            b_mat_4d[(2, 0)] = e_inv_00 * f_vec[0] + e_inv_01 * f_vec[1];
            b_mat_4d[(3, 0)] = e_inv_10 * f_vec[0] + e_inv_11 * f_vec[1];

            // --- 2. SOLVE LQR IN 4D ---
            let k_4d = calculate_discrete_lqr(&a_mat_4d, &b_mat_4d, control_step);

            // --- 3. EMBED 4D RESULTS INTO DIM_X MATRICES ---
            // This maps the 4x4 blocks into the top-left of the potentially 5x5 blocks.
            // If DIM_X is 5, the 5th row/column remains exactly 0.0,
            // completely ignoring the backlash state.
            a_mat_full.fixed_view_mut::<4, 4>(0, 0).copy_from(&a_mat_4d);
            b_mat_full.fixed_view_mut::<4, 1>(0, 0).copy_from(&b_mat_4d);
            analytical_k_full
                .fixed_view_mut::<1, 4>(0, 0)
                .copy_from(&k_4d);

            println!("\n============================================================");
            println!("        ANALYTICAL LQR GAIN CALCULATION (100Hz)");
            println!("============================================================");
            println!("Optimal K Matrix (Control Law: u = Kx):");
            println!("  K_phi       = {:>10.4}", k_4d[(0, 0)]);
            println!("  K_theta     = {:>10.4}", k_4d[(0, 1)]);
            println!("  K_phi_dot   = {:>10.4}", k_4d[(0, 2)]);
            println!("  K_theta_dot = {:>10.4}", k_4d[(0, 3)]);
            if BACKLASH_ESTIMATION {
                println!("  K_backlash  =     0.0000 (Ignored in Analytical LQR)");
            }
            println!("============================================================\n");
        }
        _ => {}
    }

    // Reset robot to random upright pose
    data.reset();
    data.qpos_mut()[2] = 0.05;
    let initial_theta: f64 = rng.random_range(-0.1745..0.1745);
    data.qpos_mut()[3] = (initial_theta / 2.0).cos();
    data.qpos_mut()[5] = (initial_theta / 2.0).sin();
    data.qvel_mut()[4] = rng.random_range(-0.1..0.1);

    (analytical_k_full, a_mat_full, b_mat_full)
}

fn should_exit_early(task: &SimTask, tracker: &SimTracker) -> bool {
    match task {
        SimTask::CollectBatch {
            enable_rendering,
            viewer,
            ..
        } => {
            let is_running = !enable_rendering || viewer.as_ref().map_or(true, |v| v.running());
            tracker.state_batch.len() >= SAMPLES_PER_ITER || !is_running
        }
        _ => false,
    }
}

fn compute_control_effort(
    task: &mut SimTask,
    x_k: &SVector<f64, DIM_X>,
    analytical_k: &SMatrix<f64, DIM_U, DIM_X>,
) -> f64 {
    match task {
        SimTask::CollectBatch {
            pending_gains,
            active_gains,
            ..
        } => {
            if let Some(new_gains) = pending_gains.lock().unwrap().take() {
                **active_gains = new_gains;
            }
            // Dot product between active gains and current state
            active_gains
                .iter()
                .zip(x_k.iter())
                .map(|(k, x)| k * x)
                .sum()
        }
        SimTask::EvaluatePolicy { policy_gains } => policy_gains
            .iter()
            .zip(x_k.iter())
            .map(|(k, x)| k * x)
            .sum(),
        SimTask::EstimateProcessNoise => (analytical_k * x_k)[(0, 0)],
    }
}

fn apply_motor_physics<'a>(
    data: &mut MjData<&'a MjModel>,
    u_raw: f64,
    phi_dot: f64,
    phi_dot_left: f64,
    phi_dot_right: f64,
    enable_noise: bool,
    rng: &mut StdRng,
    last_noise: &mut f64,
) -> (f64, f64) {
    let max_physical_torque = 0.22;
    let pwm_resolution = 400.0;
    let max_speed = 47.5;

    let _gaussian_noise = 0.0;

    // 1. OU Noise
    if enable_noise {
        let u1: f64 = rng.random_range(0.0001..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let epsilon = (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * PI * u2).cos();
        let dt: f64 = 0.01;
        let dx = THETA_OU * (-*last_noise) * dt + SIGMA_OU * epsilon * dt.sqrt();

        // Standard deviation of the exact discrete series
        let discrete_variance = (SIGMA_OU * SIGMA_OU) / (2.0 * THETA_OU - THETA_OU * THETA_OU * dt);
        let exact_std_dev = discrete_variance.sqrt();

        // The resulting Gaussian noise
        let _gaussian_noise = exact_std_dev * epsilon;

        *last_noise += dx;
    } else {
        *last_noise = 0.0;
    }

    let raw_tau = (u_raw + *last_noise) / 2.0;
    let raw_tau_offset = if phi_dot > 0.0 {
        raw_tau + 0.00
    } else {
        raw_tau - 0.00
    };

    // 2. Limits & Quantization
    let avail_l = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_left.abs() / max_speed)));
    let avail_r = max_physical_torque * (0.0f64.max(1.0 - (phi_dot_right.abs() / max_speed)));

    let pwm_l_raw = (raw_tau_offset / max_physical_torque) * pwm_resolution;
    let pwm_r_raw = (raw_tau_offset / max_physical_torque) * pwm_resolution;

    let tau_l_final =
        ((pwm_l_raw.round() / pwm_resolution) * max_physical_torque).clamp(-avail_l, avail_l);
    let tau_r_final =
        ((pwm_r_raw.round() / pwm_resolution) * max_physical_torque).clamp(-avail_r, avail_r);

    data.ctrl_mut()[0] = tau_l_final;
    data.ctrl_mut()[1] = tau_r_final;

    (raw_tau, tau_l_final + tau_r_final)
}

fn process_step_result<'a>(
    data: &mut MjData<&'a MjModel>,
    task: &mut SimTask,
    tracker: &mut SimTracker,
    step_start: Instant,
    x_k: &SVector<f64, DIM_X>,
    u_applied: f64,
    raw_tau: f64,
    current_step: usize,
    max_steps: usize,
    a_mat: &SMatrix<f64, DIM_X, DIM_X>,
    b_mat: &SMatrix<f64, DIM_X, DIM_U>,
) -> StepAction {
    let stop_angle_rad = 60.0_f64.to_radians();
    let control_step = 0.01;

    // Extract base state
    let (phi, theta, phi_dot, theta_dot) = (x_k[0], x_k[1], x_k[2], x_k[3]);

    // Safely extract backlash if it exists in the state vector
    let _backlash = if BACKLASH_ESTIMATION { x_k[4] } else { 0.0 };

    let is_sane = theta.is_finite() && theta_dot.abs() < 100.0;
    let is_upright = theta.abs() < stop_angle_rad;

    match task {
        SimTask::CollectBatch {
            enable_rendering,
            viewer,
            was_balancing,
            log_label,
            batch_index,
            ..
        } => {
            // Pacing & Rendering
            if *enable_rendering {
                if let Some(v) = viewer {
                    v.sync_data(data);
                    let _ = v.render();
                }
                let target_duration = Duration::from_secs_f64(control_step);
                let elapsed = step_start.elapsed();
                if elapsed < target_duration {
                    thread::sleep(target_duration - elapsed);
                }
            }

            if is_sane && is_upright {
                tracker.stability_counter += 1;
                if tracker.stability_counter >= 10 {
                    // stability_threshold
                    if !**was_balancing {
                        **was_balancing = true;
                    }
                    tracker.state_batch.push(StateAction {
                        phi,
                        theta,
                        phi_dot,
                        theta_dot,
                        //backlash, // <--- ADDED HERE
                        u: raw_tau,
                    });
                    tracker.loop_counter += 1;
                    if tracker.loop_counter >= 100 {
                        log_progress(
                            tracker.state_batch.len(),
                            SAMPLES_PER_ITER,
                            *batch_index,
                            log_label,
                        );
                        tracker.loop_counter = 0;
                    }
                }
            } else {
                tracker.stability_counter = 0;
                if **was_balancing {
                    tracker.fall_count += 1;
                    **was_balancing = false;
                    if tracker.fall_count > MAX_FALLS {
                        println!("\x1b[31m[WARNING] Falls ({}) exceeded MAX_FALLS ({}). Aborting.\x1b[0m", tracker.fall_count, MAX_FALLS);
                        return StepAction::AbortBatch;
                    }
                    data.reset();
                    data.qpos_mut()[2] = 0.05;
                    data.qpos_mut()[3] = 1.0;
                    tracker.last_noise = 0.0;
                }
            }
        }
        SimTask::EvaluatePolicy { .. } => {
            if !is_sane || !is_upright {
                let penalty_per_step = 100_000.0;
                tracker.total_cost += penalty_per_step * (max_steps - current_step) as f64;
                return StepAction::Break;
            }
            let state_cost = 10.0 * phi.powi(2)
                + 100.0 * theta.powi(2)
                + 0.0 * phi_dot.powi(2)
                + 0.1 * theta_dot.powi(2);
            tracker.total_cost += state_cost + 3.0 * raw_tau.powi(2);
        }
        SimTask::EstimateProcessNoise => {
            let (pl, pr, t, pdl, pdr, td) = extract_state(data, BACKLASH_JOINTS);

            // Build the next state vector dynamically to prevent panics
            // when mapping arrays to DIM_X
            let mut x_k1 = SVector::<f64, DIM_X>::zeros();
            x_k1[0] = (pl + pr) / 2.0;
            x_k1[1] = t;
            x_k1[2] = (pdl + pdr) / 2.0;
            x_k1[3] = td;

            if BACKLASH_ESTIMATION {
                x_k1[4] = x_k1[0] - x_k1[1]; //Fix not precise backlash = phi - theta
            }

            let x_dot_empirical = (x_k1 - x_k) / control_step;

            // Note: DIM_U shouldn't change, so from_column_slice here is safe
            let u_applied_vec = SVector::<f64, DIM_U>::from_column_slice(&[u_applied]);

            let x_dot_theory = (a_mat * x_k) + (b_mat * u_applied_vec);

            tracker.noises.push(x_dot_empirical - x_dot_theory);
        }
    }
    StepAction::Continue
}

fn finalize_results(task: SimTask, tracker: SimTracker, max_steps: usize) -> SimResult {
    match task {
        SimTask::CollectBatch { .. } => SimResult::Batch(tracker.state_batch),
        SimTask::EvaluatePolicy { .. } => {
            SimResult::EvaluationCost(tracker.total_cost / max_steps as f64)
        }
        SimTask::EstimateProcessNoise => {
            let mut mean = SVector::<f64, DIM_X>::zeros();
            for n in &tracker.noises {
                mean += n;
            }
            mean /= max_steps as f64;

            let mut covariance = SMatrix::<f64, DIM_X, DIM_X>::zeros();
            for n in &tracker.noises {
                let diff = n - mean;
                covariance += diff * diff.transpose();
            }
            covariance /= (max_steps - 1) as f64;

            SimResult::NoiseData(mean, covariance)
        }
    }
}

pub fn collect_full_batch_sim<'a, 'v, 't>(
    model: &'a MjModel,
    data: &mut MjData<&'a MjModel>,
    viewer: &'v mut MjViewer,
    log_label: &'t str,
    batch_index: usize,
    was_balancing: &'t mut bool,
    pending_gains: &'t Arc<Mutex<Option<[f64; DIM_X]>>>,
    active_gains: &'t mut [f64; DIM_X],
    enable_rendering: bool,
    enable_noise: bool,
) -> Vec<StateAction> {
    let task = SimTask::CollectBatch {
        viewer: Some(viewer),
        log_label,
        batch_index,
        was_balancing,
        pending_gains,
        active_gains,
        enable_rendering,
    };

    match unified_sim_loop(model, data, task, enable_noise) {
        SimResult::Batch(batch) => batch,
        _ => unreachable!("Expected SimResult::Batch from CollectBatch task"),
    }
}

pub fn estimate_process_noise<'a>(
    model: &'a MjModel,
    data: &mut MjData<&'a MjModel>,
) -> (SVector<f64, DIM_X>, SMatrix<f64, DIM_X, DIM_X>) {
    let task = SimTask::EstimateProcessNoise;

    // The original estimate_process_noise did not use the OU exploration noise
    let enable_noise = false;

    match unified_sim_loop(model, data, task, enable_noise) {
        SimResult::NoiseData(mean, covariance) => (mean, covariance),
        _ => unreachable!("Expected SimResult::NoiseData from EstimateProcessNoise task"),
    }
}

pub fn evaluate_policy_sim<'a>(
    model: &'a MjModel,
    data: &mut MjData<&'a MjModel>,
    policy_gains: [f64; DIM_X],
    enable_noise: bool,
) -> f64 {
    let task = SimTask::EvaluatePolicy { policy_gains };

    match unified_sim_loop(model, data, task, enable_noise) {
        SimResult::EvaluationCost(cost) => cost,
        _ => unreachable!("Expected SimResult::EvaluationCost from EvaluatePolicy task"),
    }
}
