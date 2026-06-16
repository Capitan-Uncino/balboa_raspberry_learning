use crate::file_utils::get_next_file_index;
use crate::learning::lstdq_lambda_standardized_polyak::{
    calculate_k, StateAction, ANALYTIC_LQR_POLICY, SAMPLES_PER_ITER,
};
use crate::logging_utils::log_progress;
use rppal::i2c::I2c;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// --- I2C ADDRESSES ---
const ARDUINO_ADDR: u16 = 0x08;
const LSM6_ADDR: u16 = 0x6B;

// IMU Control Registers
const LSM6_CTRL1_XL: u8 = 0x10; // Accelerometer Control
const LSM6_CTRL2_G: u8 = 0x11; // Gyroscope Control

// IMU Data Registers
const LSM6_OUTX_L_XL: u8 = 0x28; //  Accel X
const LSM6_OUTY_L_G: u8 = 0x24; // Gyro Y
const LSM6_OUTZ_L_XL: u8 = 0x2C; // Accel Z
const LSM6_OUTY_L_XL: u8 = 0x2A; // Accel Y
                                 //
                                 //

const STOP_TILT_RAD: f64 = 60.0 / RAD2DEG;
const START_TILT_RAD: f64 = 20.0 / RAD2DEG;

// --- EXACT PHYSICAL CONSTANTS ---
const CALIBRATION_ITERATIONS: i32 = 100;
const TICKS_RADIAN: f64 = 161.0; // 12 * 51.45 * 41 / 25
const BITS: f64 = 29000.0; // ±32768.0 -> 2**15 equivalent scalar
const DPS: f64 = 1000.0;
const RAD2DEG: f64 = 57.296; // 180 / pi
const K_LATERAL: f64 = 0.5; // Converted to radians for internal math

const DEBUG: bool = true;

// --- LOGGING HELPER ---
fn system_log(log_file: &Arc<Mutex<File>>, level: &str, msg: &str) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let log_line = format!("[{:.3}] [{}] {}", timestamp, level, msg);

    // Print to console
    if level == "ERROR" || level == "FATAL" {
        eprintln!("{}", log_line);
    } else {
        println!("{}", log_line);
    }

    // Append to file
    //if let Ok(mut file) = log_file.lock() {
    //    let _ = writeln!(file, "{}", log_line);
    //    let _ = file.flush(); // Ensure it writes immediately in case of power loss
    //}
}

// --- DATA PIPELINE STRUCTS ---

pub struct RawMeasurements {
    // Memory (Persists across loops)
    pub g_y_zero: i32,
    pub last_time: Instant,
    pub last_encoder_left: i32,
    pub last_encoder_right: i32,
    pub encoder_left_zero: i32,
    pub encoder_right_zero: i32,

    // Current Tick Data
    pub a_x_raw: i16,
    pub a_z_raw: i16,
    pub g_y_raw: i16,
    pub encoder_left: i32,
    pub encoder_right: i32,
    pub battery_mv: u16,
    pub dt: f64,
}

#[derive(Debug)]
pub struct ProcessedState {
    // Controller Memory (Persists across loops)
    pub last_direction_forward: bool,
    pub last_oscillation_time: Instant,

    // Current Tick Physics
    pub phi: f64,
    pub phi_dot: f64,
    pub theta: f64,
    pub theta_dot: f64,
    pub phi_diff: f64,
    pub battery_mv: u16,
}

// --- IMU HELPER FUNCTIONS ---

fn init_and_calibrate_imu(
    i2c_bus: &Arc<Mutex<I2c>>,
    log_file: &Arc<Mutex<File>>,
) -> Result<(RawMeasurements, ProcessedState), Box<dyn Error>> {
    let mut bus = i2c_bus.lock().unwrap();

    bus.set_slave_address(LSM6_ADDR)?;

    // 1. Turn on the Gyro (208 Hz, 1000 deg/s)
    bus.write(&[LSM6_CTRL2_G, 0b01011000])?;

    // 2. Turn on the Accelerometer (208 Hz, ±2g)
    bus.write(&[LSM6_CTRL1_XL, 0b01010000])?;

    thread::sleep(Duration::from_millis(500));

    system_log(
        log_file,
        "INFO",
        "Calibrating IMU (Do not touch the robot)...",
    );

    let mut total_g_y: i64 = 0;
    let mut total_accel_x: i64 = 0;
    let mut total_accel_y: i64 = 0;
    let mut total_accel_z: i64 = 0;

    for _ in 0..CALIBRATION_ITERATIONS {
        let mut buf_gy = [0u8; 2];
        let mut buf_ax = [0u8; 2];
        let mut buf_ay = [0u8; 2];
        let mut buf_az = [0u8; 2];

        // Read Gyro Y
        bus.write_read(&[LSM6_OUTY_L_G], &mut buf_gy)?;
        total_g_y += i16::from_le_bytes(buf_gy) as i64;

        // Read Accel X, Y & Z
        bus.write_read(&[LSM6_OUTX_L_XL], &mut buf_ax)?;
        bus.write_read(&[LSM6_OUTY_L_XL], &mut buf_ay)?;
        bus.write_read(&[LSM6_OUTZ_L_XL], &mut buf_az)?;

        total_accel_x += i16::from_le_bytes(buf_ax) as i64;
        total_accel_y += i16::from_le_bytes(buf_ay) as i64;
        total_accel_z += i16::from_le_bytes(buf_az) as i64;

        thread::sleep(Duration::from_millis(1));
    }

    // Averages
    let g_y_zero = (total_g_y / CALIBRATION_ITERATIONS as i64) as i32;
    let avg_accel_x = (total_accel_x / CALIBRATION_ITERATIONS as i64) as f64;
    let avg_accel_y = (total_accel_y / CALIBRATION_ITERATIONS as i64) as f64;
    let avg_accel_z = (total_accel_z / CALIBRATION_ITERATIONS as i64) as f64;

    // Log the raw values
    system_log(
        log_file,
        "DEBUG",
        &format!(
            "Raw Gravity - X: {}, Y: {}, Z: {}",
            avg_accel_x, avg_accel_y, avg_accel_z
        ),
    );

    let initial_theta = f64::atan2(avg_accel_z, avg_accel_x);

    system_log(
        log_file,
        "SUCCESS",
        &format!(
            "Calibration complete. Initial Angle: {:.2} degrees",
            initial_theta * RAD2DEG
        ),
    );

    let raw = RawMeasurements {
        g_y_zero: (total_g_y / CALIBRATION_ITERATIONS as i64) as i32,
        last_time: Instant::now(),
        last_encoder_left: 0,
        last_encoder_right: 0,
        encoder_left_zero: 0,
        encoder_right_zero: 0,
        g_y_raw: 0,
        a_x_raw: 0,
        a_z_raw: 0,
        encoder_left: 0,
        encoder_right: 0,
        battery_mv: 0,
        dt: 0.0,
    };

    let processed = ProcessedState {
        last_direction_forward: true,
        last_oscillation_time: Instant::now(),
        phi: 0.0,
        phi_dot: 0.0,
        theta: initial_theta,
        theta_dot: 0.0,
        phi_diff: 0.0,
        battery_mv: 0,
    };

    Ok((raw, processed))
}

// --- HARDWARE & MATH LOGIC ---

fn gather_raw_state(i2c_bus: &Arc<Mutex<I2c>>, raw: &mut RawMeasurements) -> bool {
    let mut bus = match i2c_bus.lock() {
        Ok(b) => b,
        Err(_) => return false,
    };

    // 1. Time Delta
    let now = Instant::now();
    raw.dt = now.duration_since(raw.last_time).as_secs_f64();
    raw.last_time = now;

    // 2. Gyro Read & Integration
    raw.g_y_raw = raw.g_y_zero as i16; // Fallback
    if bus.set_slave_address(LSM6_ADDR).is_ok() {
        let mut buf = [0u8; 2];
        if bus.write_read(&[LSM6_OUTY_L_G], &mut buf).is_ok() {
            raw.g_y_raw = i16::from_le_bytes(buf);
        }
    }

    if bus.set_slave_address(LSM6_ADDR).is_ok() {
        let mut buf = [0u8; 2];
        if bus.write_read(&[LSM6_OUTX_L_XL], &mut buf).is_ok() {
            raw.a_x_raw = i16::from_le_bytes(buf);
        }
    }

    if bus.set_slave_address(LSM6_ADDR).is_ok() {
        let mut buf = [0u8; 2];
        if bus.write_read(&[LSM6_OUTZ_L_XL], &mut buf).is_ok() {
            raw.a_z_raw = i16::from_le_bytes(buf);
        }
    }

    // 3. Telemetry Read
    if bus.set_slave_address(ARDUINO_ADDR).is_err() {
        return false;
    }
    let mut buf = [0u8; 10];
    if bus.read(&mut buf).is_err() {
        return false;
    }

    raw.encoder_left = i32::from_le_bytes(buf[0..4].try_into().unwrap());
    raw.encoder_right = i32::from_le_bytes(buf[4..8].try_into().unwrap());
    raw.battery_mv = u16::from_le_bytes(buf[8..10].try_into().unwrap());

    true
}

fn process_measurements(raw: &RawMeasurements, old_state: &ProcessedState) -> ProcessedState {
    // 1. Calculate physics
    let theta_dot = (raw.g_y_raw as f64 - raw.g_y_zero as f64) / BITS * DPS / RAD2DEG;

    let acc_weight = 0.01;

    let gyro_weight = 0.99;

    let acc_theta = raw.a_z_raw as f64 / raw.a_x_raw as f64;

    let gyro_theta = old_state.theta + theta_dot * raw.dt;

    let acc_theta_weighted = acc_weight * acc_theta;

    let gyro_theta_weighted = gyro_weight * gyro_theta;

    let theta = acc_theta_weighted + gyro_theta_weighted;

    let phi_left = (raw.encoder_left - raw.encoder_left_zero) as f64 / TICKS_RADIAN;
    let phi_right = (raw.encoder_right - raw.encoder_right_zero) as f64 / TICKS_RADIAN;

    let phi = (phi_left + phi_right) / 2.0;
    let phi_dot = ((raw.encoder_left - raw.last_encoder_left) as f64 / TICKS_RADIAN / raw.dt
        + (raw.encoder_right - raw.last_encoder_right) as f64 / TICKS_RADIAN / raw.dt)
        / 2.0;

    // 2. Return the new state, carrying forward the persistence
    ProcessedState {
        // Carry forward persistence
        last_direction_forward: old_state.last_direction_forward,
        last_oscillation_time: old_state.last_oscillation_time,

        // Update with fresh physics
        phi,
        phi_dot,
        theta,
        theta_dot,
        phi_diff: phi_left - phi_right,
        battery_mv: raw.battery_mv,
    }
}

fn compute_control_action(
    state: &mut ProcessedState,
    current_k: &Arc<Mutex<nalgebra::SMatrix<f64, 1, 4>>>,
    was_balancing: &bool,
    avoid_oscillations: bool,
) -> (f64, i16, i16) {
    if !was_balancing {
        return (0.0, 0, 0);
    }

    let u_physical = {
        let k = current_k.lock().unwrap();
        k[(0, 0)] * state.phi
            + k[(0, 1)] * state.theta
            + k[(0, 2)] * state.phi_dot
            + k[(0, 3)] * state.theta_dot
    };

    let u_left = u_physical - (state.phi_diff * K_LATERAL);
    let u_right = u_physical + (state.phi_diff * K_LATERAL);

    let mut v_batt = state.battery_mv as f64 / 1000.0;
    if v_batt < 1.0 {
        v_batt = 7.4;
    }

    let mut pwm_left = ((400.0 / v_batt) * u_left).clamp(-400.0, 400.0) as i16;
    let mut pwm_right = ((400.0 / v_batt) * u_right).clamp(-400.0, 400.0) as i16;

    if avoid_oscillations {
        let present_forward = u_physical >= 0.0;
        if present_forward != state.last_direction_forward {
            let now = Instant::now();
            if now.duration_since(state.last_oscillation_time).as_millis() < 100 {
                return (0.0, 0, 0);
            } else {
                state.last_oscillation_time = now;
                state.last_direction_forward = present_forward;
            }
        }
    }

    (u_physical, pwm_left, pwm_right)
}

fn write_commands(
    i2c_bus: &Arc<Mutex<I2c>>,
    left_speed: i16,
    right_speed: i16,
    log_file: &Arc<Mutex<File>>,
) {
    let mut write_buf = [0u8; 4];
    write_buf[0..2].copy_from_slice(&left_speed.to_le_bytes());
    write_buf[2..4].copy_from_slice(&right_speed.to_le_bytes());

    if let Ok(mut bus) = i2c_bus.lock() {
        let _ = bus.set_slave_address(ARDUINO_ADDR);
        if let Err(e) = bus.write(&write_buf) {
            system_log(
                log_file,
                "ERROR",
                &format!("I2C Write Motor Command Error: {:?}", e),
            );
        }
    }
}

// --- MAIN ORCHESTRATOR ---

pub fn collect_full_batch(
    i2c_bus: &Arc<Mutex<I2c>>,
    log_label: &str,
    batch_index: usize,
    was_balancing: &mut bool,
    current_k: &Arc<Mutex<nalgebra::SMatrix<f64, 1, 4>>>,
    log_file: &Arc<Mutex<File>>,
    raw: &mut RawMeasurements,
    state: &mut ProcessedState,
) -> Vec<StateAction> {
    let mut state_batch = Vec::with_capacity(SAMPLES_PER_ITER);
    let mut loop_counter = 0;
    let mut stability_counter = 0;
    let stability_threshold = 10;
    let mut i2c_error_state = false;

    if gather_raw_state(i2c_bus, raw) {
        raw.last_encoder_left = raw.encoder_left;
        raw.last_encoder_right = raw.encoder_right;
        raw.encoder_left_zero = raw.encoder_left;
        raw.encoder_right_zero = raw.encoder_right;
        raw.last_time = Instant::now();
    }

    let mut iteration_count = 0;

    while state_batch.len() < SAMPLES_PER_ITER {
        iteration_count += 1;
        // 1. GATHER
        if gather_raw_state(i2c_bus, raw) {
            if i2c_error_state {
                system_log(log_file, "SUCCESS", "Hardware I2C Connection Restored");
                i2c_error_state = false;
            }

            // 2. PROCESS
            // Create a new tick state, carrying over memory from the persistent ProcessedState
            let mut current_state = process_measurements(raw, state);

            if DEBUG && iteration_count % 1000 == 1 {
                println!("[DEBUG] current state {:?}", current_state);
            }

            // 3. COMPUTE CONTROL
            // Returns (Average Physical U, PWM Left, PWM Right)
            let (u_avg, speed_left, speed_right) =
                compute_control_action(&mut current_state, current_k, was_balancing, true);

            // 4. ACTUATE
            write_commands(i2c_bus, speed_left, speed_right, log_file);

            // 5. STABILITY & LOGGING
            if current_state.theta.abs() < START_TILT_RAD {
                stability_counter += 1;

                if stability_counter >= stability_threshold {
                    if !*was_balancing {
                        system_log(
                            log_file,
                            "STATE",
                            &format!("ROBOT STANDING: Resuming {}...", log_label),
                        );
                        *was_balancing = true;

                        // Reset positional drift on the raw measurement tracker
                        current_state.theta = 0.0;
                        current_state.phi = 0.0;
                        if gather_raw_state(i2c_bus, raw) {
                            raw.last_encoder_left = raw.encoder_left;
                            raw.last_encoder_right = raw.encoder_right;
                            raw.encoder_left_zero = raw.encoder_left;
                            raw.encoder_right_zero = raw.encoder_right;
                            raw.last_time = Instant::now();
                        }
                    }

                    // Log the snapshot
                    state_batch.push(StateAction {
                        phi: current_state.phi,
                        theta: current_state.theta,
                        phi_dot: current_state.phi_dot,
                        theta_dot: current_state.theta_dot,
                        u: u_avg,
                    });

                    loop_counter += 1;
                    if loop_counter >= 100 {
                        log_progress(state_batch.len(), SAMPLES_PER_ITER, batch_index, log_label);
                        loop_counter = 0;
                    }
                }
            } else if current_state.theta.abs() > STOP_TILT_RAD {
                stability_counter = 0;
                if *was_balancing {
                    system_log(
                        log_file,
                        "WARN",
                        &format!("ROBOT FELL: Pausing {}...", log_label),
                    );
                    *was_balancing = false;
                }
            }

            // 6. UPDATE HISTORY for next tick's derivatives
            raw.last_encoder_left = raw.encoder_left;
            raw.last_encoder_right = raw.encoder_right;

            *state = current_state;
        } else {
            // Failsafe if I2C fails
            stability_counter = 0;
            if !i2c_error_state {
                system_log(
                    log_file,
                    "ERROR",
                    "Hardware Read Failed (Suppressing log until restored)",
                );
                i2c_error_state = true;
            }
        }

        thread::sleep(Duration::from_millis(10));
    }

    system_log(
        log_file,
        "SUCCESS",
        &format!("Completed Batch {} for {}", batch_index, log_label),
    );
    state_batch
}

// Helper function that blocks and retries until the robot is physically powered on and answering
fn connect_i2c_with_retry(log_file: &Arc<Mutex<File>>) -> Arc<Mutex<I2c>> {
    system_log(log_file, "INFO", "Waiting for Robot I2C connection...");

    loop {
        match I2c::new() {
            Ok(mut i2c) => {
                if let Ok(_) = i2c.set_slave_address(0x08) {
                    // 0x08 is ARDUINO_ADDR
                    // PING: Try reading 1 byte to verify the Arduino is actually powered on
                    let mut buf = [0u8; 1];
                    if i2c.read(&mut buf).is_ok() {
                        system_log(
                            log_file,
                            "SUCCESS",
                            "I2C connection established and Robot is ONLINE.",
                        );
                        return Arc::new(Mutex::new(i2c));
                    } else {
                        system_log(log_file, "WARN", "I2C open, but Robot did not respond (Is it powered on?). Retrying in 3s...");
                    }
                } else {
                    system_log(
                        log_file,
                        "WARN",
                        "Failed to set I2C address. Retrying in 3s...",
                    );
                }
            }
            Err(e) => {
                system_log(
                    log_file,
                    "WARN",
                    &format!("Failed to init I2C bus ({}). Retrying in 3s...", e),
                );
            }
        }
        thread::sleep(Duration::from_secs(3));
    }
}

pub fn run_online_mode() -> Result<(), Box<dyn Error>> {
    let log_target = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("robot_system.log")?,
    ));

    system_log(&log_target, "START", "=== INIT ONLINE MODE ===");
    let i2c_bus = connect_i2c_with_retry(&log_target);

    // ✨ CALIBRATE ONCE HERE
    let (mut measures, mut state) = init_and_calibrate_imu(&i2c_bus, &log_target)?;

    let initial_k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(&ANALYTIC_LQR_POLICY);
    let current_k = Arc::new(Mutex::new(initial_k_mat));

    let mut computations_completed = 0;
    let mut was_balancing = false;

    system_log(&log_target, "INFO", "Starting 100Hz I2C control loop...");

    loop {
        let batch_to_process = collect_full_batch(
            &i2c_bus,
            "LSTDQ Batch",
            computations_completed,
            &mut was_balancing,
            &current_k,
            &log_target,
            &mut measures,
            &mut state,
        );

        computations_completed += 1;
        let k_clone = Arc::clone(&current_k);
        let log_clone = Arc::clone(&log_target);

        thread::spawn(move || {
            if let Some(core_ids) = core_affinity::get_core_ids() {
                if core_ids.len() > 2 {
                    core_affinity::set_for_current(core_ids[2]);
                }
            }

            let k_to_use = { *k_clone.lock().unwrap() };
            let new_k_mat = calculate_k(&batch_to_process, &k_to_use);

            {
                *k_clone.lock().unwrap() = new_k_mat;
            }

            system_log(&log_clone, "UPDATE", "LSTDQ: New K matrix applied.");
        });
    }
}

pub fn run_data_collection_mode() -> Result<(), Box<dyn Error>> {
    let log_target = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("robot_system.log")?,
    ));

    system_log(&log_target, "START", "=== INIT DATA COLLECTION MODE ===");
    let i2c_bus = connect_i2c_with_retry(&log_target);

    // ✨ CALIBRATE ONCE HERE
    let (mut measures, mut state) = init_and_calibrate_imu(&i2c_bus, &log_target)?;

    let file_index = get_next_file_index();
    let mut current_file_index = file_index;
    let mut was_balancing = false;
    let data_dir = "collected_data";

    let dummy_k = Arc::new(Mutex::new(nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(
        &ANALYTIC_LQR_POLICY,
    )));

    system_log(
        &log_target,
        "INFO",
        &format!(
            "Started data collection. Start index: {}",
            current_file_index
        ),
    );

    loop {
        let batch_to_process = collect_full_batch(
            &i2c_bus,
            "LSTDQ Batch",
            current_file_index,
            &mut was_balancing,
            &dummy_k,
            &log_target,
            &mut measures,
            &mut state,
        );

        let filename = format!("{}/batch_{}.csv", data_dir, current_file_index);
        let mut file = match File::create(&filename) {
            Ok(f) => f,
            Err(e) => {
                system_log(
                    &log_target,
                    "ERROR",
                    &format!("Failed to create CSV {}: {:?}", filename, e),
                );
                continue;
            }
        };

        if let Err(e) = writeln!(file, "phi,phi_dot,theta,theta_dot,u") {
            system_log(
                &log_target,
                "ERROR",
                &format!("Failed to write CSV Headers: {:?}", e),
            );
        }

        for s in &batch_to_process {
            let _ = writeln!(
                file,
                "{},{},{},{},{}",
                s.phi, s.theta, s.phi_dot, s.theta_dot, s.u
            );
        }

        system_log(
            &log_target,
            "SUCCESS",
            &format!(
                "Saved batch to {} (Next: {})",
                filename,
                current_file_index + 1
            ),
        );
        current_file_index += 1;
    }
}
