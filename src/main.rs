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
use clap::Parser;
use robot_comunication::i2c_comunication_external_controller::{
    run_data_collection_mode, run_online_mode,
};
use std::error::Error;

// Conditionally import the sim functions
#[cfg(feature = "sim")]
use sim::mujoco_sim::{run_data_collection_mode_sim, run_online_mode_sim, run_sim_plot};

/// Balboa Brain v2.0 - Configuration Arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run in online mode
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    online: bool,

    /// Run a new batch for data collection
    #[arg(long, action = clap::ArgAction::Set, default_value_t = false)]
    new_batch: bool,

    /// Enable simulation mode
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    sim: bool,

    /// Enable visualization
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    visualize: bool,

    /// Enable plotting
    #[arg(long, action = clap::ArgAction::Set, default_value_t = false)]
    plot: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse arguments from the command line
    let args = Args::parse();

    println!("========================================");
    println!("    BALBOA BRAIN v2.0 - log    ");
    println!("========================================");
    println!(
        "Mode flags: ONLINE={}, NEW_BATCH={}",
        args.online, args.new_batch
    );

    // Safety check: Prevent silently ignoring the SIM flag if the feature isn't compiled
    if args.sim && !cfg!(feature = "sim") {
        eprintln!("⚠️ ERROR: SIM mode is true, but the 'sim' feature was not compiled.");
        eprintln!("Recompile without --no-default-features, or pass --sim=false.");
        return Ok(());
    }

    if args.online {
        if args.sim {
            // This block is entirely pruned by the compiler if the "sim" feature is missing
            #[cfg(feature = "sim")]
            {
                if args.plot {
                    run_sim_plot(args.visualize, 15.0, 5, 4, 0.5)?;
                } else {
                    run_online_mode_sim(args.visualize)?;
                }
            }
        } else {
            run_online_mode()?;
        }
    } else if args.new_batch {
        if args.sim {
            #[cfg(feature = "sim")]
            {
                run_data_collection_mode_sim(args.visualize)?;
            }
        } else {
            run_data_collection_mode()?;
        }
    } else {
        run_offline_computation_mode()?;
    }

    Ok(())
}
