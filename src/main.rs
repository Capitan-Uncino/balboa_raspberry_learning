mod batch_analisys;
mod file_utils;
mod learning;
mod logging_utils;
mod robot_comunication;

// Conditionally compile the optional modules
#[cfg(feature = "graphics")]
mod graphic_utils;

#[cfg(feature = "sim")]
mod sim;

use batch_analisys::single_batch_analisys::run_offline_computation_mode;
use robot_comunication::serial_comunication::{run_data_collection_mode, run_online_mode};
use std::error::Error;

// Conditionally import the sim functions
#[cfg(feature = "sim")]
use sim::mujoco_sim::{run_data_collection_mode_sim, run_online_mode_sim, run_sim_plot};

const ONLINE: bool = false;
const NEW_BATCH: bool = true;
const SIM: bool = false;
const VISUALIZE: bool = false;
const PLOT: bool = false;

fn main() -> Result<(), Box<dyn Error>> {
    println!("========================================");
    println!("    BALBOA BRAIN v2.0 - log    ");
    println!("========================================");
    println!("Mode flags: ONLINE={}, NEW_BATCH={}", ONLINE, NEW_BATCH);

    // Safety check: Prevent silently ignoring the SIM flag if the feature isn't compiled
    if SIM && !cfg!(feature = "sim") {
        eprintln!("⚠️ ERROR: SIM mode is true, but the 'sim' feature was not compiled.");
        eprintln!("Recompile without --no-default-features, or set SIM to false.");
        return Ok(());
    }

    if ONLINE {
        if SIM {
            // This block is entirely pruned by the compiler if the "sim" feature is missing
            #[cfg(feature = "sim")]
            {
                if PLOT {
                    run_sim_plot(VISUALIZE, 30.0, 10, 5, 0.5)?;
                } else {
                    run_online_mode_sim(VISUALIZE)?;
                }
            }
        } else {
            run_online_mode()?;
        }
    } else if NEW_BATCH {
        if SIM {
            #[cfg(feature = "sim")]
            {
                run_data_collection_mode_sim(VISUALIZE)?;
            }
        } else {
            run_data_collection_mode()?;
        }
    } else {
        run_offline_computation_mode()?;
    }

    Ok(())
}
