use std::fs;

/// Scans the directory for batch_X.csv files and returns the next available index.
pub fn get_next_file_index() -> usize {
    let mut highest = 0;
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("batch_") && name.ends_with(".csv") {
                    // Extract X from "batch_X.csv"
                    let parts: Vec<&str> = name.split('_').collect();
                    if let Some(num_part) = parts.get(1) {
                        if let Some(num_str) = num_part.split('.').next() {
                            if let Ok(num) = num_str.parse::<usize>() {
                                if num >= highest {
                                    highest = num + 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    highest
}
