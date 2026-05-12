use crate::file_utils::get_next_file_index;
use crate::learning::lstdq::{calculate_k, StateAction, ANALYTIC_LQR_POLICY, SAMPLES_PER_ITER};
use crate::logging_utils::log_progress;
use rppal::gpio::{Gpio, Trigger};
use rppal::i2c::I2c;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use rand::Rng;

// ==========================================
// 1. NOISE GENERATOR (Ported from C++)
// ==========================================
pub struct NoiseGenerator {
    last_noise: f64,
    theta_ou: f64,
    sigma_ou: f64,
}

impl NoiseGenerator {
    pub fn new() -> Self {
        Self {
            last_noise: 0.0,
            theta_ou: 0.30,
            sigma_ou: 0.80,
        }
    }

    fn generate_gaussian(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let u1: f64 = rng.gen_range(0.0001..=1.0);
        let u2: f64 = rng.gen_range(0.0..=1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn generate_exploration_noise(&mut self) -> f64 {
        let epsilon = self.generate_gaussian();
        let dx = self.theta_ou * (-self.last_noise) * 0.01 + self.sigma_ou * epsilon * 0.1;
        self.last_noise += dx;
        self.last_noise
    }
}

// ==========================================
// 2. STATE TRACKER
// ==========================================
pub struct RobotState {
    pub phi: f64,
    pub phi_dot: f64,
    pub phi_dif: f64,
    pub theta: f64,
    pub theta_dot: f64,
    last_counts_left: f64,
    last_counts_right: f64,
}

impl RobotState {
    pub fn new() -> Self {
        Self {
            phi: 0.0, phi_dot: 0.0, phi_dif: 0.0,
            theta: 0.0, theta_dot: 0.0,
            last_counts_left: 0.0, last_counts_right: 0.0,
        }
    }

    pub fn update_encoders(&mut self, enc_left: i32, enc_right: i32, dt_ms: f64) {
        let ticks_radian = 161.0;
        let counts_left = (enc_left as f64) / ticks_radian;
        let counts_right = (enc_right as f64) / ticks_radian;

        let phi_dot_left = (counts_left - self.last_counts_left) * 1000.0 / dt_ms;
        let phi_dot_right = (counts_right - self.last_counts_right) * 1000.0 / dt_ms;

        self.phi += (counts_left - self.last_counts_left + counts_right - self.last_counts_right) / 2.0;
        self.phi_dif = counts_left - counts_right;
        self.phi_dot = (phi_dot_left + phi_dot_right) / 2.0;

        self.last_counts_left = counts_left;
        self.last_counts_right = counts_right;
    }

    // Call this if you are calculating theta natively by reading the IMU in Rust
    pub fn update_gyro(&mut self, gyro_rate: f64, dt_ms: f64) {
        self.theta_dot = gyro_rate;
        self.theta += self.theta_dot * dt_ms / 1000.0;
    }
}

// ==========================================
// 3. CORE CONTROL LOOP
// ==========================================
fn collect_full_batch(
    i2c_bus: &Arc<Mutex<I2c>>,
    log_label: &str,
    batch_index: usize,
    was_balancing: &mut bool,
    pending_gains: &Arc<Mutex<Option<[f32; 4]>>>,
    current_gains: &mut [f32; 4],
    state: &mut RobotState,
    noise_gen: &mut NoiseGenerator,
) -> Vec<StateAction> {
    let mut state_batch = Vec::with_capacity(SAMPLES_PER_ITER);
    let mut loop_counter = 0;

    let gpio = Gpio::new().expect("Failed to initialize GPIO");
    let mut sync_pin = gpio.get(22).expect("GPIO 22 busy").into_input();
    sync_pin.set_interrupt(Trigger::RisingEdge, None).expect("Failed to set interrupt");

    let mut stability_counter = 0;
    let stability_threshold = 10;
    let stop_angle_rad = 60.0_f64.to_radians();
    let dt_ms = 10.0; // 10ms loop time

    println!(">>> Syncing with Hardware Clock...");

    while state_batch.len() < SAMPLES_PER_ITER {
        let _ = sync_pin.poll_interrupt(true, Some(Duration::from_millis(15)));

        if sync_pin.is_low() {
            continue;
        }

        // 1. UPDATE GAINS LOCALLY (If background thread finished)
        if let Some(new_gains) = pending_gains.lock().unwrap().take() {
            *current_gains = new_gains;
        }

        let mut telemetry_buf = [0u8; 10];
        let mut is_read_successful = false;

        {
            let mut bus = i2c_bus.lock().unwrap();
            
            // 2. READ ENCODERS & BATTERY FROM BALBOA (Address 0x08)
            let _ = bus.set_slave_address(0x08);
            if bus.read(&mut telemetry_buf).is_ok() {
                is_read_successful = true;
            }

            // 3. READ IMU (Address 0x6B)
            // (Assuming you have implemented LSM6 I2C reads here)
            // let _ = bus.set_slave_address(0x6B);
            // let gyro_rate = ... read from LSM6 ...
            // state.update_gyro(gyro_rate, dt_ms);
            
            // MOCK GYRO FOR NOW (Replace with actual IMU read)
            state.update_gyro(0.0, dt_ms); 
        }

        if is_read_successful {
            // Parse Telemetry
            let enc_left = i32::from_le_bytes(telemetry_buf[0..4].try_into().unwrap());
            let enc_right = i32::from_le_bytes(telemetry_buf[4..8].try_into().unwrap());
            let battery_mv = u16::from_le_bytes(telemetry_buf[8..10].try_into().unwrap());

            state.update_encoders(enc_left, enc_right, dt_ms);

            // 4. CALCULATE CONTROL EFFORT (Policy + Noise)
            let u_raw = current_gains[0] as f64 * state.phi + 
                        current_gains[1] as f64 * state.theta + 
                        (current_gains[2] as f64 + 0.19) * state.phi_dot + 
                        current_gains[3] as f64 * state.theta_dot;

            let noise = noise_gen.generate_exploration_noise();
            let u_noisy = u_raw + noise;

            // 5. CALCULATE PHYSICAL MOTOR COMMANDS
            let mut u_physical = u_noisy;
            let offset = 0.45;
            if state.phi_dot > 0.0 { u_physical += offset; } else { u_physical -= offset; }

            let actual_voltage = (battery_mv as f64 / 1000.0).max(5.0);
            let base_speed = (400.0 / actual_voltage) * u_physical;

            let distance_diff_response = 80.0;
            let left_speed = (base_speed - state.phi_dif * distance_diff_response).clamp(-400.0, 400.0) as i16;
            let right_speed = (base_speed + state.phi_dif * distance_diff_response).clamp(-400.0, 400.0) as i16;

            // 6. WRITE MOTOR COMMANDS TO BALBOA (Address 0x08)
            write_motor_commands(i2c_bus, left_speed, right_speed);

            // 7. RECORD DATA
            let data = StateAction {
                phi: state.phi,
                theta: state.theta,
                phi_dot: state.phi_dot,
                theta_dot: state.theta_dot,
                u: u_noisy,
            };

            let is_sane = data.theta.is_finite() && data.theta_dot.abs() < 100.0;
            let is_upright = data.theta.abs() < stop_angle_rad;

            if is_sane && is_upright {
                stability_counter += 1;
                if stability_counter >= stability_threshold {
                    if !*was_balancing {
                        println!(">>> ROBOT STANDING: Resuming {}...", log_label);
                        *was_balancing = true;
                    }

                    state_batch.push(data);
                    loop_counter += 1;

                    if loop_counter >= 100 {
                        log_progress(state_batch.len(), SAMPLES_PER_ITER, batch_index, log_label);
                        loop_counter = 0;
                    }
                }
            } else {
                stability_counter = 0;
                if *was_balancing {
                    println!("<<< ROBOT FELL: Pausing {}", log_label);
                    *was_balancing = false;
                }
            }
        } else {
            stability_counter = 0;
        }
    }

    let _ = sync_pin.clear_interrupt();
    state_batch
}

fn write_motor_commands(i2c_bus: &Arc<Mutex<I2c>>, left: i16, right: i16) {
    let mut write_buf = Vec::with_capacity(4);
    write_buf.extend_from_slice(&left.to_le_bytes());
    write_buf.extend_from_slice(&right.to_le_bytes());

    if let Ok(mut i2c) = i2c_bus.lock() {
        let _ = i2c.set_slave_address(0x08);
        if let Err(e) = i2c.write(&write_buf) {
            eprintln!("I2C Write Error: {:?}", e);
        }
    }
}

pub fn run_online_mode() -> Result<(), Box<dyn Error>> {
    let mut i2c = I2c::new().map_err(|e| format!("Failed to init I2C: {}", e))?;
    let i2c_bus = Arc::new(Mutex::new(i2c));

    let initial_k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(&ANALYTIC_LQR_POLICY);
    let current_k = Arc::new(Mutex::new(initial_k_mat));
    let pending_gains: Arc<Mutex<Option<[f32; 4]>>> = Arc::new(Mutex::new(None));

    let mut current_gains = [
        initial_k_mat[(0, 0)] as f32, initial_k_mat[(0, 1)] as f32,
        initial_k_mat[(0, 2)] as f32, initial_k_mat[(0, 3)] as f32,
    ];

    let mut state = RobotState::new();
    let mut noise_gen = NoiseGenerator::new();
    let mut computations_completed = 0;
    let mut was_balancing = false;

    println!("Starting 100Hz I2C control loop...");

    loop {
        let batch_to_process = collect_full_batch(
            &i2c_bus,
            "LSTDQ Batch",
            computations_completed,
            &mut was_balancing,
            &pending_gains,
            &mut current_gains,
            &mut state,
            &mut noise_gen,
        );

        computations_completed += 1;
        let k_clone = Arc::clone(&current_k);
        let pending_clone = Arc::clone(&pending_gains);

        thread::spawn(move || {
            let k_to_use = { *k_clone.lock().unwrap() };
            let new_k_mat = calculate_k(batch_to_process, &k_to_use);

            { *k_clone.lock().unwrap() = new_k_mat; }

            let new_k_array = [
                new_k_mat[(0, 0)] as f32, new_k_mat[(0, 1)] as f32,
                new_k_mat[(0, 2)] as f32, new_k_mat[(0, 3)] as f32,
            ];

            *pending_clone.lock().unwrap() = Some(new_k_array);
            println!(">>> LSTDQ Update: New K vector queued for next I2C window.");
        });
    }
}

pub fn run_data_collection_mode() -> Result<(), Box<dyn Error>> {
    let mut i2c = I2c::new().map_err(|e| format!("Failed to init I2C: {}", e))?;
    let i2c_bus = Arc::new(Mutex::new(i2c));

    let mut file_index = get_next_file_index();
    let mut was_balancing = false;
    let data_dir = "collected_data";
    
    let dummy_pending_gains: Arc<Mutex<Option<[f32; 4]>>> = Arc::new(Mutex::new(None));
    let mut initial_gains = [0.182, 4.412, 0.098, 0.441]; // Fallback safe values

    let mut state = RobotState::new();
    let mut noise_gen = NoiseGenerator::new();

    println!("Started data collection mode. Will start at index: {}", file_index);

    loop {
        let batch_to_process = collect_full_batch(
            &i2c_bus,
            "CSV Collection",
            file_index,
            &mut was_balancing,
            &dummy_pending_gains,
            &mut initial_gains,
            &mut state,
            &mut noise_gen,
        );

        let filename = format!("{}/batch_{}.csv", data_dir, file_index);
        let mut file = File::create(&filename)?;

        writeln!(file, "phi,phi_dot,theta,theta_dot,u")?;
        for s in &batch_to_process {
            writeln!(file, "{},{},{},{},{}", s.phi, s.theta, s.phi_dot, s.theta_dot, s.u)?;
        }

        println!(">>> Saved batch of size {} to {} (Next: {})", SAMPLES_PER_ITER, filename, file_index + 1);
        file_index += 1;
    }
}
