use plotters::prelude::*;

pub fn plot_cost_evolution(
    costs: &[Vec<f64>],
    baseline_cost: f64,
    n_updates: usize,
    evaluation_threshold: f64, // <-- Added parameter
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("policy_evolution.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Cap the Y-axis using the threshold (plus a small 5% margin) so diverging
    // policies don't squash the scale of the successful ones.
    let max_y = evaluation_threshold.max(baseline_cost) * 1.05;

    // [NEW] Find the minimum cost to set the lower bound of the logarithmic Y-axis.
    // A log scale cannot start at 0, so we need a positive minimum value.
    let min_cost = costs
        .iter()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .min(baseline_cost);

    // Ensure min_y is strictly greater than 0 (fallback to 1e-3 if costs are exactly 0)
    let min_y = if min_cost > 0.0 { min_cost * 0.9 } else { 1e-3 };

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Empirical LQR Cost over Updates (Log Scale)",
            ("sans-serif", 40),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        // [NEW] Use .log_scale() on the Y-axis range
        .build_cartesian_2d(0..n_updates, (min_y..max_y).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("Update Iteration")
        .y_desc("Empirical Cost (Log Scale)")
        .draw()?;

    // Plot the evaluation threshold as a red line
    chart
        .draw_series(LineSeries::new(
            (0..=n_updates).map(|x| (x, evaluation_threshold)),
            RED.stroke_width(2),
        ))?
        .label(format!("Discard Threshold ({:.2})", evaluation_threshold))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

    // Plot each perturbed policy
    for (i, policy_costs) in costs.iter().enumerate() {
        let color = Palette99::pick(i);
        chart
            .draw_series(LineSeries::new(
                // This naturally stops plotting when the discarded policy's vector ends
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
