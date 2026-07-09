//! Timing check on realistic load-forecasting dimensions:
//! one year of 15-minute data (35,040 rows), 8 features, quantile model.
//!
//! Run: cargo run --release --example bench

use gbt_quantile::config::GBTConfig;
use gbt_quantile::trainer::train;
use std::time::Instant;

fn main() {
    let n = 35_040;
    let n_features = 8;

    // Deterministic synthetic consumption-like data
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let quarter = (i % 96) as f64;
        let dow = ((i / 96) % 7) as f64;
        let temp = 10.0 + 15.0 * ((i as f64 / 3504.0).sin());
        let lag = 50.0 + 30.0 * ((quarter / 96.0) * std::f64::consts::TAU).sin();
        let noise = ((i * 7 + 3) % 101) as f64 / 10.0;
        x.push(vec![
            quarter,
            dow,
            if dow >= 5.0 { 1.0 } else { 0.0 },
            temp,
            lag,
            (i % 4) as f64,
            ((i / 96) % 365) as f64,
            noise,
        ]);
        y.push(lag + 0.5 * temp + noise + if dow >= 5.0 { -10.0 } else { 0.0 });
    }

    let config = GBTConfig {
        n_trees: 50,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_leaf: 20,
        quantile: Some(0.5),
        early_stopping_rounds: None,
        n_bins: 255,
    };

    let start = Instant::now();
    let model = train(&x, &y, &config, None);
    let elapsed = start.elapsed();
    println!(
        "train: {} rows x {} features, {} trees -> {:.2?}",
        n,
        n_features,
        model.n_trees(),
        elapsed
    );

    let start = Instant::now();
    let preds: Vec<f64> = x.iter().map(|row| model.predict(row)).collect();
    println!(
        "predict: {} rows -> {:.2?} (sum {:.1})",
        n,
        start.elapsed(),
        preds.iter().sum::<f64>()
    );
}
