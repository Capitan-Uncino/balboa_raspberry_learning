pub fn log_progress(current: usize, total: usize, completed: usize, mode: &str) {
    let step = (total / 10).max(1);
    if current > 0 && current % step == 0 {
        let percent = (current as f32 / total as f32) * 100.0;
        let bars = (percent / 10.0) as usize;
        let bar_str = format!("[{}{}]", "=".repeat(bars), " ".repeat(10 - bars));

        println!(
            "[INFO] {} {} {:>3.0}% ({:>4}/{:>4}) | Total Completed: {}",
            mode, bar_str, percent, current, total, completed
        );
    }
}
