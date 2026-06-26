use crate::learning::lstdq_lambda_standardized_polyak::{
    calculate_k, StateAction, ANALYTIC_LQR_POLICY,
};
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
    let filepath = Path::new("collected_data").join(filename);
    let mut file = File::open(filepath)?;
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

    let m = 5; // Number of policy iterations
    println!(
        "Loaded {} records. Starting {} policy iterations...",
        batch.len(),
        m
    );

    // Initialize the K matrix with the baseline analytic policy
    let mut current_k_mat =
        nalgebra::SMatrix::<f64, 1, 4>::from_row_slice(ANALYTIC_LQR_POLICY.as_slice());

    // Iteratively evaluate and improve the policy
    for iteration in 1..=m {
        // Calculate the new policy based on the batch and the CURRENT policy
        current_k_mat = calculate_k(&batch, &current_k_mat);

        println!("--- Iteration {} Completed ---", iteration);
        println!("K = {}", current_k_mat);
    }

    println!("========================================");
    println!(
        ">>> FINAL COMPUTED K MATRIX RESULT (After {} Iterations) <<<",
        m
    );
    println!("{}", current_k_mat);
    println!("========================================");

    Ok(())
}
