use crate::file_utils::get_next_file_index;
use crate::learning::policy::Policy;
use crate::learning::single_batch_lspi::{
    get_policy, StateAction, ANALYTIC_LQR_POLICY, DIM_U, DIM_X, Q_COST, R_COST, SAMPLES_PER_ITER,
};
use crate::logging_utils::log_progress;
use chrono::Local;
use nalgebra::{SMatrix, SVector};
use rand_distr::{Distribution, Normal};
use rppal::i2c::I2c;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{f64, thread};

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
const K_LATERAL_P: f64 = 5.0;
const K_LATERAL_I: f64 = 5.0;
const K_LATERAL_D: f64 = 0.0;
const BALANCE_ANGLE_RADIANS: f64 = 0.1553;
const DEBUG: bool = false;
const COMPLEMENTARY_FILTER_ENABLED: bool = true;

// --- LOGGING HELPER ---
fn system_log(_log_file: &Arc<Mutex<File>>, level: &str, msg: &str) {
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
    pub a_x_accumulator: f64,
    pub a_z_accumulator: f64,
    pub encoder_accumulator: f64,

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
    pub ou_noise: f64,
    pub phi_diff_i: f64,
    pub phi_diff_d: f64,

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
) -> Result<(RawMeasurements, ProcessedState), Box<dyn Error>> {
    let mut bus = i2c_bus.lock().unwrap();

    bus.set_slave_address(LSM6_ADDR)?;

    // 1. Turn on the Gyro (208 Hz, 1000 deg/s)
    bus.write(&[LSM6_CTRL2_G, 0b01011000])?;

    // 2. Turn on the Accelerometer (208 Hz, ±2g)
    bus.write(&[LSM6_CTRL1_XL, 0b01010000])?;

    thread::sleep(Duration::from_millis(500));

    println!("\x1b[36m[INFO]\x1b[0m Calibrating IMU (Do not touch the robot)...");

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
    // Log the raw values
    println!(
        "\x1b[35m[DEBUG]\x1b[0m Raw Gravity - X: {}, Y: {}, Z: {}",
        avg_accel_x, avg_accel_y, avg_accel_z
    );

    let initial_theta = f64::atan2(avg_accel_z, avg_accel_x);

    println!(
        "\x1b[32m[SUCCESS]\x1b[0m Calibration complete. Initial Angle: {:.2} degrees",
        initial_theta * RAD2DEG
    );

    let raw = RawMeasurements {
        g_y_zero,
        last_time: Instant::now(),
        last_encoder_left: 0,
        last_encoder_right: 0,
        encoder_left_zero: 0,
        encoder_right_zero: 0,
        encoder_accumulator: 0.0,
        g_y_raw: 0,
        a_x_raw: 0,
        a_z_raw: 0,
        a_x_accumulator: 0.0,
        a_z_accumulator: 0.0,
        encoder_left: 0,
        encoder_right: 0,
        battery_mv: 0,
        dt: 0.0,
    };

    let processed = ProcessedState {
        last_direction_forward: true,
        last_oscillation_time: Instant::now(),
        ou_noise: 0.0,
        phi_diff_i: 0.0,
        phi_diff_d: 0.0,
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

pub fn gather_raw_state(i2c_bus: &Arc<Mutex<I2c>>, raw: &mut RawMeasurements) -> bool {
    // --- TIMED: MUTEX LOCK ---
    let t_lock = Instant::now();
    let mut bus = match i2c_bus.lock() {
        Ok(b) => b,
        Err(_) => return false,
    };
    let d_lock = t_lock.elapsed();
    if d_lock.as_micros() > 2000 {
        eprintln!(
            "\x1b[35m[DEBUG-I2C]\x1b[0m Mutex lock took {} us",
            d_lock.as_micros()
        );
    }

    // 1. Time Delta
    let now = Instant::now();
    raw.dt = now.duration_since(raw.last_time).as_secs_f64();
    raw.last_time = now;

    if raw.dt > 0.015 {
        eprintln!(
            "\x1b[31m[CRITICAL GAP]\x1b[0m {:.1} ms elapsed since last sensor read! (Target: 10 ms)",
            raw.dt * 1000.0
        );
    }

    // 2. Gyro Read & Integration
    raw.g_y_raw = raw.g_y_zero as i16; // Fallback

    // --- TIMED: GYRO Y ---
    let t_gyro = Instant::now();
    if bus.set_slave_address(LSM6_ADDR).is_ok() {
        let mut buf = [0u8; 2];
        if bus.write_read(&[LSM6_OUTY_L_G], &mut buf).is_ok() {
            raw.g_y_raw = i16::from_le_bytes(buf);
        }
    }
    let d_gyro = t_gyro.elapsed();
    if d_gyro.as_micros() > 2000 {
        eprintln!(
            "\x1b[35m[DEBUG-I2C]\x1b[0m LSM6 Gyro Y read took {} us",
            d_gyro.as_micros()
        );
    }

    // --- TIMED: ACCEL X ---
    let t_acc_x = Instant::now();
    if bus.set_slave_address(LSM6_ADDR).is_ok() {
        let mut buf = [0u8; 2];
        if bus.write_read(&[LSM6_OUTX_L_XL], &mut buf).is_ok() {
            raw.a_x_raw = i16::from_le_bytes(buf);
        }
    }
    let d_acc_x = t_acc_x.elapsed();
    if d_acc_x.as_micros() > 2000 {
        eprintln!(
            "\x1b[35m[DEBUG-I2C]\x1b[0m LSM6 Accel X read took {} us",
            d_acc_x.as_micros()
        );
    }

    // --- TIMED: ACCEL Z ---
    let t_acc_z = Instant::now();
    if bus.set_slave_address(LSM6_ADDR).is_ok() {
        let mut buf = [0u8; 2];
        if bus.write_read(&[LSM6_OUTZ_L_XL], &mut buf).is_ok() {
            raw.a_z_raw = i16::from_le_bytes(buf);
        }
    }
    let d_acc_z = t_acc_z.elapsed();
    if d_acc_z.as_micros() > 2000 {
        eprintln!(
            "\x1b[35m[DEBUG-I2C]\x1b[0m LSM6 Accel Z read took {} us",
            d_acc_z.as_micros()
        );
    }

    // --- TIMED: TELEMETRY (ARDUINO) ---
    let t_telem = Instant::now();
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

    let d_telem = t_telem.elapsed();
    if d_telem.as_micros() > 2000 {
        eprintln!(
            "\x1b[35m[DEBUG-I2C]\x1b[0m Arduino Telemetry read took {} us",
            d_telem.as_micros()
        );
    }

    true
}

fn process_measurements(
    raw: &mut RawMeasurements,
    old_state: &ProcessedState,
    complementary_filter: bool,
    smoothed_derivative: bool,
) -> ProcessedState {
    // 1. Calculate physics
    let theta_dot = (raw.g_y_raw as f64 - raw.g_y_zero as f64) / BITS * DPS / RAD2DEG;

    let theta = if complementary_filter {
        let alpha_theta = 0.05;

        let acc_weight = 0.01;

        let gyro_weight = 0.99;

        raw.a_z_accumulator =
            alpha_theta * raw.a_z_raw as f64 + (1.0 - alpha_theta) * raw.a_z_accumulator;

        raw.a_x_accumulator =
            alpha_theta * raw.a_x_raw as f64 + (1.0 - alpha_theta) * raw.a_x_accumulator;

        let acc_theta = f64::atan2(raw.a_z_accumulator, raw.a_x_accumulator);

        let gyro_theta = old_state.theta + theta_dot * raw.dt;

        let acc_theta_weighted = acc_weight * acc_theta;

        let gyro_theta_weighted = gyro_weight * gyro_theta;

        acc_theta_weighted + gyro_theta_weighted
    } else {
        let mut angle = old_state.theta + theta_dot * raw.dt;
        if old_state.theta.abs() < STOP_TILT_RAD {
            angle *= 0.999;
        }
        angle
    };

    let phi_left = (raw.encoder_left - raw.encoder_left_zero) as f64 / TICKS_RADIAN;
    let phi_right = (raw.encoder_right - raw.encoder_right_zero) as f64 / TICKS_RADIAN;

    let phi = (phi_left + phi_right) / 2.0;

    let phi_dot = if smoothed_derivative {
        let alpha_phi = 0.50;
        let phi_dot_raw =
            ((raw.encoder_left - raw.last_encoder_left) as f64 / TICKS_RADIAN / raw.dt
                + (raw.encoder_right - raw.last_encoder_right) as f64 / TICKS_RADIAN / raw.dt)
                / 2.0;
        raw.encoder_accumulator =
            alpha_phi * phi_dot_raw + (1.0 - alpha_phi) * raw.encoder_accumulator;
        raw.encoder_accumulator
    } else {
        ((raw.encoder_left - raw.last_encoder_left) as f64 / TICKS_RADIAN / raw.dt
            + (raw.encoder_right - raw.last_encoder_right) as f64 / TICKS_RADIAN / raw.dt)
            / 2.0
    };

    let phi_diff = phi_left - phi_right;

    // 2. Return the new state, carrying forward the persistence
    ProcessedState {
        // Carry forward persistence
        last_direction_forward: old_state.last_direction_forward,
        last_oscillation_time: old_state.last_oscillation_time,
        ou_noise: old_state.ou_noise,

        // Update with fresh physics
        phi,
        phi_dot,
        theta,
        theta_dot,
        phi_diff,
        phi_diff_i: if old_state.theta.abs() < STOP_TILT_RAD {
            old_state.phi_diff_i + phi_diff * raw.dt
        } else {
            0.0
        },
        phi_diff_d: phi_diff - old_state.phi_diff,
        battery_mv: raw.battery_mv,
    }
}

fn compute_control_action(
    state: &mut ProcessedState,
    current_policy: Arc<Mutex<Policy>>, // <-- Changed from SMatrix to Policy
    balancing: &bool,
    avoid_oscillations: bool,
    enable_noise: bool,
    enable_balance_angle_compensation: bool,
) -> (f64, i16, i16) {
    if !balancing {
        return (0.0, 0, 0);
    }

    // 1. Construct the state vector for the Policy
    let theta_input = if enable_balance_angle_compensation {
        state.theta - BALANCE_ANGLE_RADIANS
    } else {
        state.theta
    };

    let x = nalgebra::SVector::<f64, DIM_X>::new(
        state.phi,
        theta_input,
        state.phi_dot,
        state.theta_dot,
    );

    // 2. Query the policy (Agnostic to whether it's LQR or a Neural Network)
    // The lock is strictly scoped so it drops immediately after getting the action
    let mut u_physical = {
        let policy = current_policy.lock().unwrap();
        policy.get_action(&x)
    };

    // --- The rest remains completely unchanged ---
    if enable_noise {
        let sigma_ou = 0.60;
        let theta_ou = 0.60;
        let dt: f64 = 0.01;

        let mut rng = rand::rng();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let standard_normal = rand_distr::Distribution::sample(&normal, &mut rng);

        // Euler-Maruyama discretization of the Ornstein-Uhlenbeck process
        // dX_t = -theta * X_t * dt + sigma * dW_t
        let dw = dt.sqrt() * standard_normal;
        state.ou_noise += -theta_ou * state.ou_noise * dt + sigma_ou * dw;

        u_physical += state.ou_noise;
    }

    let u_left = u_physical
        - (state.phi_diff * K_LATERAL_P
            + state.phi_diff_i * K_LATERAL_I
            + state.phi_diff_d * K_LATERAL_D);

    let u_right = u_physical
        + (state.phi_diff * K_LATERAL_P
            + state.phi_diff_i * K_LATERAL_I
            + state.phi_diff_d * K_LATERAL_D);

    let mut v_batt = state.battery_mv as f64 / 1000.0;
    if v_batt < 1.0 {
        v_batt = 7.4;
    }

    let pwm_left = ((400.0 / v_batt) * u_left).clamp(-400.0, 400.0) as i16;
    let pwm_right = ((400.0 / v_batt) * u_right).clamp(-400.0, 400.0) as i16;

    if avoid_oscillations {
        let present_forward = u_physical >= 0.0;
        if present_forward != state.last_direction_forward {
            let now = Instant::now();
            if now.duration_since(state.last_oscillation_time).as_millis() < 100 {
                return (u_physical, 0, 0);
            } else {
                state.last_oscillation_time = now;
                state.last_direction_forward = present_forward;
            }
        }
    }
    let _u_average = (pwm_left as f64 + pwm_right as f64) / 2.0;

    (u_physical, pwm_left, pwm_right)
}

fn write_commands(i2c_bus: &Arc<Mutex<I2c>>, left_speed: i16, right_speed: i16) {
    let mut write_buf = [0u8; 4];
    write_buf[0..2].copy_from_slice(&left_speed.to_le_bytes());
    write_buf[2..4].copy_from_slice(&right_speed.to_le_bytes());

    if let Ok(mut bus) = i2c_bus.lock() {
        let _ = bus.set_slave_address(ARDUINO_ADDR);
        if let Err(e) = bus.write(&write_buf) {
            eprintln!("\x1b[31m[ERROR] \x1bI2C Write Motor Command Error: {}", e);
        }
    }
}
// --- MAIN ORCHESTRATOR ---

pub fn collect_full_batch(
    i2c_bus: &Arc<Mutex<I2c>>,
    log_label: &str,
    batch_index: usize,
    current_policy: Arc<Mutex<Policy>>,
    raw: &mut RawMeasurements,
    state: &mut ProcessedState,
    batch_size: usize,
    enable_noise: bool,
    was_balancing: &mut bool,
) -> Vec<StateAction> {
    let t_alloc_start = Instant::now();
    let mut state_batch = Vec::with_capacity(batch_size);
    let alloc_duration = t_alloc_start.elapsed();

    let mut loop_counter = 0;
    let stability_threshold = 10;
    let mut stability_counter = stability_threshold;
    let mut i2c_error_state = false;
    let t_i2c_start = Instant::now();
    let mut initial_read_success = false;
    if !*was_balancing {
        initial_read_success = gather_raw_state(i2c_bus, raw);
    }

    let i2c_duration = t_i2c_start.elapsed();
    if alloc_duration.as_millis() > 5 || i2c_duration.as_millis() > 5 {
        eprintln!(
            "\x1b[31m[CRITICAL]\x1b[0m STARTUP DELAY for {}: Alloc took {} us, Initial I2C took {} us",
            log_label,
            alloc_duration.as_micros(),
            i2c_duration.as_micros()
        );
    }

    if !*was_balancing && initial_read_success {
        raw.last_encoder_left = raw.encoder_left;
        raw.last_encoder_right = raw.encoder_right;
        raw.encoder_left_zero = raw.encoder_left;
        raw.encoder_right_zero = raw.encoder_right;
        raw.last_time = Instant::now();
    }

    let mut iteration_count = 0;
    let mut last_tick_start = Instant::now();

    while state_batch.len() < batch_size {
        let now = Instant::now();
        let true_tick_gap = now.duration_since(last_tick_start);

        // --- TIMED: OS THREAD SLEEP OVERLAP ---
        if true_tick_gap.as_millis() > 15 && iteration_count > 0 {
            eprintln!("\x1b[31m[CRITICAL]\x1b[0m BLIND SPOT: {} ms passed since last control tick start! (OS Oversleep)", true_tick_gap.as_millis());
        }
        last_tick_start = now;
        let start_time = now;
        iteration_count += 1;

        // --- TIMED: I2C READ ---
        let t_gather = Instant::now();
        let gather_success = gather_raw_state(i2c_bus, raw);
        let d_gather = t_gather.elapsed();
        if d_gather.as_micros() > 1500 {
            eprintln!(
                "\x1b[35m[DEBUG-TIME]\x1b[0m gather_raw_state took {} us",
                d_gather.as_micros()
            );
        }

        if gather_success {
            if i2c_error_state {
                println!("\x1b[32m[SUCCESS]\x1b[0m Hardware I2C Connection Restored");
                i2c_error_state = false;
            }

            // --- TIMED: PROCESS ---
            let t_process = Instant::now();
            let complementary_filter = COMPLEMENTARY_FILTER_ENABLED;
            let mut current_state = process_measurements(raw, state, complementary_filter, true);
            let d_process = t_process.elapsed();
            if d_process.as_micros() > 1000 {
                eprintln!(
                    "\x1b[35m[DEBUG-TIME]\x1b[0m process_measurements took {} us",
                    d_process.as_micros()
                );
            }

            if DEBUG && iteration_count % 1000 == 1 {
                println!("[DEBUG] current state {:?}", current_state);
            }

            // --- TIMED: CONTROL COMPUTE ---
            let t_control = Instant::now();

            // Pass the cloned policy instead of current_k
            let (u_avg, speed_left, speed_right) = compute_control_action(
                &mut current_state,
                current_policy.clone(),
                was_balancing,
                false,
                enable_noise,
                complementary_filter,
            );

            let d_control = t_control.elapsed();
            if d_control.as_micros() > 1000 {
                eprintln!(
                    "\x1b[35m[DEBUG-TIME]\x1b[0m compute_control_action took {} us",
                    d_control.as_micros()
                );
            }

            // --- TIMED: I2C WRITE ---
            let t_write = Instant::now();
            write_commands(i2c_bus, speed_left, speed_right);
            let d_write = t_write.elapsed();
            if d_write.as_micros() > 1500 {
                eprintln!(
                    "\x1b[35m[DEBUG-TIME]\x1b[0m write_commands took {} us",
                    d_write.as_micros()
                );
            }

            let elapsed = start_time.elapsed();
            if elapsed.as_micros() > 3000 {
                eprintln!(
                    "\x1b[31m[ERROR]\x1b[0m control computation > 3.0ms ({} µs)",
                    elapsed.as_micros()
                );
            }

            // --- TIMED: STATE MANAGEMENT ---
            let t_state = Instant::now();
            if current_state.theta.abs() < START_TILT_RAD {
                stability_counter += 1;

                if stability_counter >= stability_threshold {
                    if !*was_balancing {
                        println!(
                            "\x1b[36m[STATE]\x1b[0m ROBOT STANDING: Resuming {}...",
                            log_label
                        );
                        *was_balancing = true;

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

                    if complementary_filter {
                        state_batch.push(StateAction {
                            phi: current_state.phi,
                            theta: current_state.theta - BALANCE_ANGLE_RADIANS,
                            phi_dot: current_state.phi_dot,
                            theta_dot: current_state.theta_dot,
                            u: u_avg,
                        });
                    } else {
                        state_batch.push(StateAction {
                            phi: current_state.phi,
                            theta: current_state.theta,
                            phi_dot: current_state.phi_dot,
                            theta_dot: current_state.theta_dot,
                            u: u_avg,
                        });
                    }

                    loop_counter += 1;
                    if loop_counter >= 100 {
                        log_progress(state_batch.len(), batch_size, batch_index, log_label);
                        loop_counter = 0;
                    }
                }
            } else if current_state.theta.abs() > STOP_TILT_RAD {
                stability_counter = 0;
                if *was_balancing {
                    eprintln!("\x1b[33m[WARN]\x1b[0m ROBOT FELL: Pausing {}...", log_label);
                    *was_balancing = false;
                }
            }
            let d_state = t_state.elapsed();
            if d_state.as_micros() > 1000 {
                eprintln!(
                    "\x1b[35m[DEBUG-TIME]\x1b[0m State push/management took {} us",
                    d_state.as_micros()
                );
            }

            raw.last_encoder_left = raw.encoder_left;
            raw.last_encoder_right = raw.encoder_right;
            *state = current_state;
        } else {
            stability_counter = 0;
            if !i2c_error_state {
                eprintln!(
                    "\x1b[31m[ERROR]\x1b[0m Hardware Read Failed (Suppressing log until restored)"
                );
                i2c_error_state = true;
            }
        }
        let elapsed_final = start_time.elapsed();
        if elapsed_final.as_micros() > 5000 {
            eprintln!(
                "\x1b[31m[ERROR]\x1b[0m full loop > 5ms ({} µs)",
                elapsed_final.as_micros()
            );
        }

        thread::sleep(Duration::from_millis(10).saturating_sub(elapsed_final));
    }

    println!(
        "\x1b[32m[SUCCESS]\x1b[0m Completed Batch {} for {}",
        batch_index, log_label
    );
    state_batch
}

// Helper function that blocks and retries until the robot is physically powered on and answering
pub fn connect_i2c_with_retry() -> Arc<Mutex<I2c>> {
    println!("\x1b[36m[INFO]\x1b[0m Waiting for Robot I2C connection...");

    loop {
        match I2c::new() {
            Ok(mut i2c) => {
                if let Ok(_) = i2c.set_slave_address(0x08) {
                    // 0x08 is ARDUINO_ADDR
                    // PING: Try reading 1 byte to verify the Arduino is actually powered on
                    let mut buf = [0u8; 1];
                    if i2c.read(&mut buf).is_ok() {
                        println!(
                            "\x1b[32m[SUCCESS]\x1b[0m I2C connection established and Robot is ONLINE."
                        );
                        return Arc::new(Mutex::new(i2c));
                    } else {
                        eprintln!(
                            "\x1b[33m[WARN]\x1b[0m I2C open, but Robot did not respond (Is it powered on?). Retrying in 3s..."
                        );
                    }
                } else {
                    eprintln!("\x1b[33m[WARN]\x1b[0m Failed to set I2C address. Retrying in 3s...");
                }
            }
            Err(e) => {
                eprintln!(
                    "\x1b[33m[WARN]\x1b[0m Failed to init I2C bus ({}). Retrying in 3s...",
                    e
                );
            }
        }
        thread::sleep(Duration::from_secs(3));
    }
}

pub fn run_online_mode() -> Result<(), Box<dyn Error>> {
    println!("\x1b[32m[START]\x1b[0m === INIT ONLINE MODE ===");

    let i2c_bus = connect_i2c_with_retry();

    let t_init = Instant::now();
    let (mut measures, mut state) = init_and_calibrate_imu(&i2c_bus)?;
    let d_init = t_init.elapsed();
    if d_init.as_millis() > 50 {
        eprintln!(
            "\x1b[35m[DEBUG-TIME]\x1b[0m IMU Calibration took {} ms",
            d_init.as_millis()
        );
    }

    // Initialize the starting Policy struct with analytic explicit gains
    let initial_k_array = [
        ANALYTIC_LQR_POLICY[0],
        ANALYTIC_LQR_POLICY[1],
        ANALYTIC_LQR_POLICY[2],
        ANALYTIC_LQR_POLICY[3],
    ];
    let initial_policy = Policy::new(
        move |x| {
            let k_mat = SMatrix::<f64, 1, 4>::from_row_slice(&initial_k_array);
            (k_mat * x)[0]
        },
        Some(initial_k_array),
    );

    let current_policy = Arc::new(Mutex::new(initial_policy));

    let mut computations_completed = 0;
    println!("\x1b[36m[INFO]\x1b[0m Starting 100Hz I2C control loop...");

    let mut balancing = false;
    let (k_tx, k_rx) = std::sync::mpsc::channel::<Policy>();

    let mut last_loop_end = Instant::now();

    loop {
        // --- TIMED: LOOP BOUNDARY (Catches Memory Deallocation/Restart Stalls) ---
        let loop_gap = last_loop_end.elapsed();
        if loop_gap.as_millis() > 5 && computations_completed > 0 {
            eprintln!("\x1b[31m[CRITICAL]\x1b[0m Loop restart gap took {} ms! (Robot was completely blind)", loop_gap.as_millis());
        }

        let _ = collect_full_batch(
            &i2c_bus,
            "Reposition Robot if necessary",
            computations_completed,
            current_policy.clone(),
            &mut measures,
            &mut state,
            SAMPLES_PER_ITER / 20,
            false,
            &mut balancing,
        );

        let big_batch = collect_full_batch(
            &i2c_bus,
            "Train Batch",
            computations_completed,
            current_policy.clone(),
            &mut measures,
            &mut state,
            SAMPLES_PER_ITER,
            true,
            &mut balancing,
        );

        // Take a cheap, native snapshot of the current Policy
        let policy_snapshot = { current_policy.lock().unwrap().clone() };
        let k_tx_clone = k_tx.clone();

        // --- TIMED: MATH THREAD SPAWN ---
        let t_spawn_math = Instant::now();
        thread::spawn(move || {
            if let Some(core_ids) = core_affinity::get_core_ids() {
                if core_ids.len() > 1 {
                    core_affinity::set_for_current(core_ids[1]);
                }
            }

            // Pass the native Policy object directly
            let new_policy = get_policy(&big_batch, &policy_snapshot);
            let _ = k_tx_clone.send(new_policy);
        });

        let d_spawn_math = t_spawn_math.elapsed();
        if d_spawn_math.as_millis() > 2 {
            eprintln!(
                "\x1b[35m[DEBUG-TIME]\x1b[0m Math thread spawn took {} us",
                d_spawn_math.as_micros()
            );
        }

        let _ = collect_full_batch(
            &i2c_bus,
            "Reposition Robot if necessary",
            computations_completed,
            current_policy.clone(),
            &mut measures,
            &mut state,
            SAMPLES_PER_ITER / 20,
            false,
            &mut balancing,
        );

        // Take another snapshot for the CSV thread
        let eval_policy_snapshot = { current_policy.lock().unwrap().clone() };

        let small_batch = collect_full_batch(
            &i2c_bus,
            "Eval Batch",
            computations_completed,
            current_policy.clone(),
            &mut measures,
            &mut state,
            SAMPLES_PER_ITER / 5,
            false,
            &mut balancing,
        );

        // --- TIMED: CHANNEL RECEIVE ---
        let t_recv = Instant::now();
        match k_rx.recv() {
            Ok(new_policy) => {
                {
                    *current_policy.lock().unwrap() = new_policy;
                }
                println!(
                    "\x1b[36m[UPDATE]\x1b[0m Main Thread: New Policy applied & sent to Arduino."
                );
            }
            Err(_) => {
                eprintln!("\x1b[31m[ERROR]\x1b[0m Math thread failed to send new Policy.");
            }
        }
        let d_recv = t_recv.elapsed();
        if d_recv.as_millis() > 5 {
            eprintln!(
                "\x1b[35m[DEBUG-TIME]\x1b[0m Waiting for Math Thread channel took {} ms",
                d_recv.as_millis()
            );
        }

        let iter_index = computations_completed;
        computations_completed += 1;

        // --- TIMED: CSV THREAD SPAWN ---
        let t_spawn_csv = Instant::now();
        thread::spawn(move || {
            if let Some(core_ids) = core_affinity::get_core_ids() {
                if core_ids.len() > 2 {
                    core_affinity::set_for_current(core_ids[2]);
                }
            }

            let q_cost = SMatrix::<f64, DIM_X, DIM_X>::from_diagonal(&SVector::from(Q_COST));
            let r_cost = SMatrix::<f64, DIM_U, DIM_U>::from_diagonal(&SVector::from(R_COST));
            let mut total_cost = 0.0;

            for s in &small_batch {
                let x = SVector::<f64, DIM_X>::new(
                    s.phi,
                    if COMPLEMENTARY_FILTER_ENABLED {
                        s.theta - BALANCE_ANGLE_RADIANS
                    } else {
                        s.theta
                    },
                    s.phi_dot,
                    s.theta_dot,
                );
                let u = SVector::<f64, DIM_U>::new(s.u);

                let state_cost_mat = x.transpose() * q_cost * x;
                let action_cost_mat = u.transpose() * r_cost * u;
                total_cost += state_cost_mat[(0, 0)] + action_cost_mat[(0, 0)];
            }

            let avg_cost = total_cost / small_batch.len() as f64;
            let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();

            // The CSV thread dynamically resolves the logging values here natively
            let k_log = eval_policy_snapshot
                .get_gains()
                .unwrap_or_else(|| eval_policy_snapshot.get_pseudogains());
            let k0 = k_log[0];
            let k1 = k_log[1];
            let k2 = k_log[2];
            let k3 = k_log[3];

            match OpenOptions::new()
                .create(true)
                .append(true)
                .open("empirical_costs.csv")
            {
                Ok(mut file) => {
                    if iter_index == 0 {
                        let _ = writeln!(file, "timestamp,iteration,avg_cost,k0,k1,k2,k3");
                    }
                    if let Err(e) = writeln!(
                        file,
                        "{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
                        timestamp, iter_index, avg_cost, k0, k1, k2, k3
                    ) {
                        eprintln!("\x1b[31m[ERROR]\x1b[0m Failed to write cost CSV: {:?}", e);
                    }
                }
                Err(e) => eprintln!("\x1b[31m[ERROR]\x1b[0m Failed to open cost CSV: {:?}", e),
            }
        });

        let d_spawn_csv = t_spawn_csv.elapsed();
        if d_spawn_csv.as_millis() > 2 {
            eprintln!(
                "\x1b[35m[DEBUG-TIME]\x1b[0m CSV Thread spawn took {} us",
                d_spawn_csv.as_micros()
            );
        }

        last_loop_end = Instant::now();
    }
}

pub fn run_data_collection_mode() -> Result<(), Box<dyn Error>> {
    println!("\x1b[32m[START]\x1b[0m === INIT DATA COLLECTION MODE ===");

    // Assumes connect_i2c_with_retry no longer requires log_file
    let i2c_bus = connect_i2c_with_retry();

    // ✨ CALIBRATE ONCE HERE
    let (mut measures, mut state) = init_and_calibrate_imu(&i2c_bus)?;

    let file_index = get_next_file_index();
    let mut current_file_index = file_index;
    let data_dir = "collected_data";

    // Initialize the starting Policy struct with analytic explicit gains
    let initial_k_array = [
        ANALYTIC_LQR_POLICY[0],
        ANALYTIC_LQR_POLICY[1],
        ANALYTIC_LQR_POLICY[2],
        ANALYTIC_LQR_POLICY[3],
    ];
    let initial_policy = Policy::new(
        move |x| {
            let k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(&initial_k_array);
            (k_mat * x)[0]
        },
        Some(initial_k_array),
    );

    let k = Arc::new(Mutex::new(initial_policy));

    println!(
        "\x1b[36m[INFO]\x1b[0m Started data collection. Start index: {}",
        current_file_index
    );

    let mut balancing = false;

    loop {
        let batch_to_process = collect_full_batch(
            &i2c_bus,
            "LSTDQ Batch",
            current_file_index,
            k.clone(),
            &mut measures, // log_target removed
            &mut state,
            SAMPLES_PER_ITER,
            true,
            &mut balancing,
        );

        let filename = format!("{}/batch_{}.csv", data_dir, current_file_index);

        let file = match File::create(&filename) {
            Ok(f) => f,
            Err(e) => {
                eprintln!(
                    "\x1b[31m[ERROR]\x1b[0m Failed to create CSV {}: {:?}",
                    filename, e
                );
                continue;
            }
        };

        let mut writer = BufWriter::new(file);

        if let Err(e) = writeln!(writer, "phi,theta,phi_dot,theta_dot,u") {
            eprintln!(
                "\x1b[31m[ERROR]\x1b[0m Failed to write CSV Headers: {:?}",
                e
            );
        }

        let mut write_failed = false;

        for s in &batch_to_process {
            if let Err(e) = writeln!(
                writer,
                "{},{},{},{},{}",
                s.phi, s.theta, s.phi_dot, s.theta_dot, s.u
            ) {
                eprintln!(
                    "\x1b[31m[ERROR]\x1b[0m Failed to write data row to {}: {:?}",
                    filename, e
                );
                write_failed = true;
                break;
            }
        }

        if let Err(e) = writer.flush() {
            eprintln!(
                "\x1b[31m[ERROR]\x1b[0m Failed to flush buffer to {}: {:?}",
                filename, e
            );
            write_failed = true;
        }

        if !write_failed {
            println!(
                "\x1b[32m[SUCCESS]\x1b[0m Saved batch to {} (Next: {})",
                filename,
                current_file_index + 1
            );
            current_file_index += 1;
        }
    }
}
