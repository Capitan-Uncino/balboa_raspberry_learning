use crate::learning::policy::Policy;
use crate::learning::sysid_lqr::{get_policy, StateAction, ANALYTIC_LQR_POLICY};
use std::error::Error;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

pub fn run_offline_computation_mode() -> Result<(), Box<dyn Error>> {
    print!("Enter the CSV file name to process (e.g., batch_0.csv): ");
    io::stdout().flush()?;

    let mut filename = String::new();
    io::stdin().read_line(&mut filename)?;
    let filename = filename.trim();

    println!("Reading data from {}...", filename);
    let filename = Path::new("collected_data").join(filename);
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut batch: Vec<StateAction> = Vec::new();

    // Skip the header row
    for line in contents.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 5 {
            batch.push(StateAction {
                phi: parts[0].parse()?,
                theta: parts[1].parse()?,
                phi_dot: parts[2].parse()?,
                theta_dot: parts[3].parse()?,
                u: parts[4].parse()?,
            });
        }
    }

    if batch.is_empty() {
        println!("No valid data found in file.");
        return Ok(());
    }

    println!("Loaded {} records. Computing new Policy...", batch.len());

    // 1. Initialize the starting Policy struct with analytic explicit gains
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

    // 2. Call the computation algorithm
    let new_policy = get_policy(&batch, &initial_policy);

    // 3. Extract the gains to display them to the user
    println!("========================================");
    println!(">>> COMPUTED POLICY RESULT <<<");

    if let Some(gains) = new_policy.get_gains() {
        println!("Type: Linear LQR (Explicit Matrix)");
        println!(
            "K_PHI: {:.6}, K_THETA: {:.6}, K_PHIDOT: {:.6}, K_THETADOT: {:.6}",
            gains[0], gains[1], gains[2], gains[3]
        );
    } else {
        let pseudogains = new_policy.get_pseudogains();
        println!("Type: Non-Linear Network (IQL)");
        println!("Local LQR Approximation (Evaluated at equilibrium):");
        println!(
            "K_PHI: {:.6}, K_THETA: {:.6}, K_PHIDOT: {:.6}, K_THETADOT: {:.6}",
            pseudogains[0], pseudogains[1], pseudogains[2], pseudogains[3]
        );
    }

    println!("========================================");

    Ok(())
}

