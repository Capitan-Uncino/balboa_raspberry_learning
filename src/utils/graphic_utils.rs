use plotters::prelude::*;

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

pub fn plot_cost_evolution(
    costs: &[Vec<f64>],
    baseline_cost: f64,
    n_updates: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("policy_evolution.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find max cost to scale the Y axis
    let max_cost = costs
        .iter()
        .flatten()
        .copied()
        .fold(f64::NAN, f64::max)
        .max(baseline_cost);

    let mut chart = ChartBuilder::on(&root)
        .caption("Empirical LQR Cost over Updates", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..n_updates, 0.0..(max_cost * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("Update Iteration")
        .y_desc("Empirical Cost")
        .draw()?;

    // Plot each perturbed policy
    for (i, policy_costs) in costs.iter().enumerate() {
        let color = Palette99::pick(i);
        chart
            .draw_series(LineSeries::new(
                policy_costs.iter().enumerate().map(|(x, &y)| (x, y)),
                color.stroke_width(2),
            ))?
            .label(format!("Policy {}", i))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });
    }

    // Plot the original policy baseline as a horizontal line for comparison
    chart
        .draw_series(LineSeries::new(
            (0..=n_updates).map(|x| (x, baseline_cost)),
            BLACK.stroke_width(4),
        ))?
        .label(format!("Original Policy (Cost: {:.2})", baseline_cost))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(4)));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    Ok(())
}
