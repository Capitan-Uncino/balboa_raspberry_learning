use crate::file_utils::get_next_file_index;
use crate::learning::lstdq::{calculate_k, StateAction, ANALYTIC_LQR_POLICY, SAMPLES_PER_ITER};
use crate::logging_utils::log_progress;
use serialport::SerialPort;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const PORT_NAME: &str = "/dev/ttyACM0";
const BAUD_RATE: u32 = 115200;

/// Reads exactly one telemetry packet, immune to float byte collisions
fn read_telemetry_packet(port: &mut dyn SerialPort, buf: &mut [u8; 20]) -> bool {
    let mut header = [0u8; 4];
    let magic = [0xDD, 0xCC, 0xBB, 0xAA];

    // Shift buffer one byte at a time until the exact 4-byte header matches
    loop {
        let mut b = [0u8; 1];
        if port.read(&mut b).is_ok() {
            header[0] = header[1];
            header[1] = header[2];
            header[2] = header[3];
            header[3] = b[0];

            if header == magic {
                return port.read_exact(buf).is_ok();
            }
        } else {
            return false;
        }
    }
}

fn collect_full_batch(
    serial_bus: &Arc<Mutex<Box<dyn SerialPort>>>,
    log_label: &str,
    batch_index: usize,
    was_balancing: &mut bool,
    pending_gains: &Arc<Mutex<Option<[f32; 4]>>>,
) -> Vec<StateAction> {
    let mut state_batch = Vec::with_capacity(SAMPLES_PER_ITER);
    let mut loop_counter = 0;

    let mut stability_counter = 0;
    let stability_threshold = 10;
    let stop_angle_rad = 60.0_f64.to_radians();

    println!(">>> Listening for Arduino Telemetry on {}...", PORT_NAME);

    while state_batch.len() < SAMPLES_PER_ITER {
        let mut buf = [0u8; 20];

        // 1. SYNC & READ DATA OVER USB
        // This naturally blocks until the Arduino sends its next 10ms update
        let read_success = {
            let mut port = serial_bus.lock().unwrap();
            read_telemetry_packet(&mut **port, &mut buf)
        };

        if read_success {
            // =======================================================
            // 2. WRITE NEW GAINS (IF READY)
            // Done immediately after reading to minimize latency
            // =======================================================
            if let Some(new_gains) = pending_gains.lock().unwrap().take() {
                write_gains(serial_bus, new_gains);
            }

            let data = StateAction {
                phi: f32::from_le_bytes(buf[0..4].try_into().unwrap()) as f64,
                theta: f32::from_le_bytes(buf[4..8].try_into().unwrap()) as f64,
                phi_dot: f32::from_le_bytes(buf[8..12].try_into().unwrap()) as f64,
                theta_dot: f32::from_le_bytes(buf[12..16].try_into().unwrap()) as f64,
                u: f32::from_le_bytes(buf[16..20].try_into().unwrap()) as f64,
            };

            // 4. DATA SANITY CHECKS
            let is_sane = data.theta.is_finite() && data.theta_dot.abs() < 100.0;
            let is_upright = data.theta < 900.0 && data.theta.abs() < stop_angle_rad;

            // 5. STABILITY LOGIC
            if is_sane && is_upright {
                stability_counter += 1;
                if stability_counter >= stability_threshold {
                    if !*was_balancing {
                        println!(">>> ROBOT STANDING: Resuming {}...", log_label);
                        *was_balancing = true;
                    }

                    state_batch.push(data);
                    loop_counter += 1;

                    // Progress logging
                    if loop_counter >= 100 {
                        log_progress(state_batch.len(), SAMPLES_PER_ITER, batch_index, log_label);
                        loop_counter = 0;
                    }
                }
            } else {
                // If data is corrupted or robot is down, reset the stability window
                stability_counter = 0;
                if *was_balancing {
                    println!(
                        "<<< ROBOT FELL: Pausing {} (Collected: {}/{})",
                        log_label,
                        state_batch.len(),
                        SAMPLES_PER_ITER
                    );
                    *was_balancing = false;
                }
            }
        } else {
            // Serial read failed/timed out, reset stability
            stability_counter = 0;
        }
    }

    state_batch
}

fn write_gains(serial_bus: &Arc<Mutex<Box<dyn SerialPort>>>, gains: [f32; 4]) {
    let mut write_buf = Vec::with_capacity(20);
    // Add the new 4-byte bulletproof header
    write_buf.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

    for k in &gains {
        write_buf.extend_from_slice(&k.to_le_bytes());
    }

    if let Ok(mut port) = serial_bus.lock() {
        if let Err(e) = port.write_all(&write_buf) {
            eprintln!("Serial Write Error: {:?}", e);
        }
    }
}

pub fn run_online_mode() -> Result<(), Box<dyn Error>> {
    let port = serialport::new(PORT_NAME, BAUD_RATE)
        .timeout(Duration::from_millis(50)) // 50ms timeout handles dropped frames cleanly
        .open()
        .map_err(|e| format!("Failed to open Serial Port {}: {}", PORT_NAME, e))?;

    let serial_bus = Arc::new(Mutex::new(port));

    let initial_k_mat = nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(&ANALYTIC_LQR_POLICY);
    let current_k = Arc::new(Mutex::new(initial_k_mat));

    let pending_gains: Arc<Mutex<Option<[f32; 4]>>> = Arc::new(Mutex::new(None));

    let mut computations_completed = 0;
    let mut was_balancing = false;

    println!("Starting USB control loop...");

    loop {
        let batch_to_process = collect_full_batch(
            &serial_bus,
            "LSTDQ Batch",
            computations_completed,
            &mut was_balancing,
            &pending_gains,
        );

        computations_completed += 1;
        let k_clone = Arc::clone(&current_k);
        let pending_clone = Arc::clone(&pending_gains);

        // Spawn math thread
        thread::spawn(move || {
            let k_to_use = { *k_clone.lock().unwrap() };
            let new_k_mat = calculate_k(batch_to_process, &k_to_use);

            {
                *k_clone.lock().unwrap() = new_k_mat;
            }

            let new_k_array = [
                new_k_mat[(0, 0)] as f32, // k_phi
                new_k_mat[(0, 2)] as f32, // k_theta    (Index 2 in LQR math!)
                new_k_mat[(0, 1)] as f32, // k_phiDot   (Index 1 in LQR math!)
                new_k_mat[(0, 3)] as f32, // k_thetaDot
            ];

            *pending_clone.lock().unwrap() = Some(new_k_array);
        });
    }
}

pub fn run_data_collection_mode() -> Result<(), Box<dyn Error>> {
    let port = serialport::new(PORT_NAME, BAUD_RATE)
        .timeout(Duration::from_millis(50))
        .open()
        .map_err(|e| format!("Failed to open Serial Port {}: {}", PORT_NAME, e))?;

    let serial_bus = Arc::new(Mutex::new(port));

    let mut file_index = get_next_file_index();
    let mut was_balancing = false;
    let data_dir = "collected_data";

    let dummy_pending_gains: Arc<Mutex<Option<[f32; 4]>>> = Arc::new(Mutex::new(None));

    println!(
        "Started data collection mode over USB. Will start at index: {}",
        file_index
    );

    loop {
        let batch_to_process = collect_full_batch(
            &serial_bus,
            "CSV Collection",
            file_index,
            &mut was_balancing,
            &dummy_pending_gains,
        );

        let filename = format!("{}/batch_{}.csv", data_dir, file_index);
        let mut file = File::create(&filename)?;

        writeln!(file, "phi,theta,phi_dot,theta_dot,u")?;
        for s in &batch_to_process {
            writeln!(
                file,
                "{},{},{},{},{}",
                s.phi, s.theta, s.phi_dot, s.theta_dot, s.u
            )?;
        }

        println!(
            ">>> Saved batch of size {} to {} (Next: {})",
            SAMPLES_PER_ITER,
            filename,
            file_index + 1
        );
        file_index += 1;
    }
}
