mod batch_analisys;
mod learning;
mod robot_comunication;
mod sim;
mod utils;

use batch_analisys::single_batch_analisys::run_offline_computation_mode;
use robot_comunication::i2c_comunication::{run_data_collection_mode, run_online_mode};
use sim::mujoco_sim::{run_data_collection_mode_sim, run_online_mode_sim, run_sim_plot};
use std::error::Error;

const ONLINE: bool = true;
const NEW_BATCH: bool = false;
const SIM: bool = true;
const VISUALIZE: bool = false;
const PLOT: bool = false;

fn main() -> Result<(), Box<dyn Error>> {
    println!("========================================");
    println!("    BALBOA BRAIN v2.0 - I2C ENABLED     ");
    println!("========================================");
    println!("Mode flags: ONLINE={}, NEW_BATCH={}", ONLINE, NEW_BATCH);

    if ONLINE {
        if SIM {
            run_online_mode_sim(VISUALIZE)?;
        } else {
            run_online_mode()?;
        }
    } else if NEW_BATCH {
        if SIM {
            run_data_collection_mode_sim(VISUALIZE)?;
        } else {
            run_data_collection_mode()?;
        }
    } else if PLOT {
        run_sim_plot(VISUALIZE)?
    } else {
        run_offline_computation_mode()?;
    }

    Ok(())
}
