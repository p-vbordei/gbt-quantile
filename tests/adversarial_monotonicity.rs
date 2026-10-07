//! Adversarial test harness for quantile ensemble monotonicity.
//!
//! Validates that under the optimized trainer, QuantileEnsemble strictly
//! guarantees tau_1 <= tau_2 => y_hat(tau_1) <= y_hat(tau_2) across
//! adversarial and pathological datasets.

use gbt_quantile::config::GBTConfig;
use gbt_quantile::ensemble::QuantileEnsemble;
use gbt_quantile::trainer::train;

fn assert_strictly_monotone(preds: &[(f64, f64)], context: &str) {
    for window in preds.windows(2) {
        let (q1, v1) = window[0];
        let (q2, v2) = window[1];
        assert!(
            q1 <= q2,
            "[{context}] Quantiles must be sorted: q1={q1} > q2={q2}"
        );
        assert!(
            v1 <= v2 + 1e-12,
            "[{context}] Monotonicity violation: q1={q1} (val={v1}) > q2={q2} (val={v2})"
        );
    }
}

#[test]
fn test_adversarial_heteroscedastic_quantile_monotonicity() {
    // Heteroscedastic noise: variance explodes for large x, causing unconstrained
    // quantile estimators to cross without monotonicity enforcement.
    let n = 2000;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let xi = (i as f64) / (n as f64) * 20.0;
        let noise_amplitude = (xi * 0.5).powi(2);
        let pseudo_rand = (((i * 1103515245 + 12345) / 65536) % 32768) as f64 / 32768.0 - 0.5;
        let yi = 2.0 * xi + 5.0 + pseudo_rand * noise_amplitude;

        x.push(vec![xi, (i % 24) as f64, ((i / 24) % 7) as f64]);
        y.push(yi);
    }

    let config = GBTConfig {
        n_trees: 60,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    let quantiles = vec![0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];
    let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, None);

    // Test across a dense grid of evaluation points
    for step in 0..500 {
        let eval_x = step as f64 / 25.0;
        let sample = vec![eval_x, (step % 24) as f64, ((step / 24) % 7) as f64];
        let pred = ensemble.predict(&sample);
        assert_strictly_monotone(pred.values(), &format!("eval_x={eval_x}"));

        assert!(pred.lower() <= pred.median() + 1e-12);
        assert!(pred.median() <= pred.upper() + 1e-12);
    }
}

#[test]
fn test_adversarial_unordered_and_duplicate_quantiles() {
    let n = 500;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let xi = i as f64 / 50.0;
        x.push(vec![xi]);
        y.push(xi.sin() * 10.0 + (i % 5) as f64);
    }

    let config = GBTConfig {
        n_trees: 30,
        max_depth: 3,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    // Passed completely out-of-order with duplicate quantiles
    let disordered_quantiles = vec![0.90, 0.10, 0.50, 0.50, 0.80, 0.20, 0.01, 0.99];
    let ensemble = QuantileEnsemble::train(&x, &y, &disordered_quantiles, &config, None);

    for eval_i in 0..100 {
        let sample = vec![eval_i as f64 / 10.0];
        let pred = ensemble.predict(&sample);
        assert_strictly_monotone(pred.values(), &format!("disordered_eval={eval_i}"));
    }
}

#[test]
fn test_adversarial_fine_grained_quantiles() {
    // 19 fine-grained quantiles: 0.05, 0.10, ..., 0.95
    let quantiles: Vec<f64> = (1..20).map(|i| i as f64 * 0.05).collect();

    let n = 800;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let xi = i as f64 / 40.0;
        let noise = ((i * 17 + 23) % 97) as f64 / 50.0 - 1.0;
        x.push(vec![xi, ((i % 96) as f64)]);
        y.push(50.0 + 10.0 * (xi * 0.2).cos() + noise * (1.0 + xi * 0.1));
    }

    let config = GBTConfig {
        n_trees: 40,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_leaf: 10,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, None);

    for eval_i in 0..200 {
        let sample = vec![eval_i as f64 / 10.0, (eval_i % 96) as f64];
        let pred = ensemble.predict(&sample);
        assert_eq!(pred.values().len(), 19);
        assert_strictly_monotone(pred.values(), &format!("fine_grained_eval={eval_i}"));
    }
}

#[test]
fn test_adversarial_pathological_constant_and_zero_variance() {
    // Degenerate case: all targets identical, all features identical
    let n = 200;
    let x = vec![vec![42.0, 7.0]; n];
    let y = vec![100.0; n];

    let config = GBTConfig {
        n_trees: 20,
        max_depth: 3,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    let quantiles = vec![0.10, 0.50, 0.90];
    let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, None);

    let pred = ensemble.predict(&[42.0, 7.0]);
    assert_strictly_monotone(pred.values(), "constant_data");
    for &(_, val) in pred.values() {
        assert!((val - 100.0).abs() < 1e-5, "Expected 100.0, got {val}");
    }
}

#[test]
fn test_adversarial_bimodal_distribution_cross_check() {
    // Bimodal mixture: 50% around 10.0, 50% around 100.0
    let n = 1000;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let xi = (i % 96) as f64;
        let is_high = (i % 2) == 1;
        let yi = if is_high { 100.0 + (i % 7) as f64 } else { 10.0 + (i % 3) as f64 };
        x.push(vec![xi]);
        y.push(yi);
    }

    let config = GBTConfig {
        n_trees: 50,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    let quantiles = vec![0.10, 0.25, 0.50, 0.75, 0.90];
    let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, None);

    for quarter in 0..96 {
        let pred = ensemble.predict(&[quarter as f64]);
        assert_strictly_monotone(pred.values(), &format!("bimodal_quarter_{quarter}"));
    }
}

#[test]
fn test_unconstrained_trainer_vs_ensemble_monotonicity() {
    // Verify empirically whether unconstrained independent models cross on adversarial data,
    // and that QuantileEnsemble's enforce_monotonicity strictly fixes it.
    let n = 300;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let xi = i as f64;
        // Bizarre step-discontinuous function with asymmetric outliers
        let yi = if i < 150 {
            50.0 + if i % 10 == 0 { -40.0 } else { 10.0 }
        } else {
            30.0 + if i % 7 == 0 { 60.0 } else { -5.0 }
        };
        x.push(vec![xi]);
        y.push(yi);
    }

    let config_p10 = GBTConfig {
        n_trees: 30,
        max_depth: 4,
        learning_rate: 0.2,
        min_samples_leaf: 2,
        quantile: Some(0.10),
        early_stopping_rounds: None,
        n_bins: 255,
    };
    let config_p90 = GBTConfig {
        quantile: Some(0.90),
        ..config_p10.clone()
    };

    let model_p10 = train(&x, &y, &config_p10, None);
    let model_p90 = train(&x, &y, &config_p90, None);

    // Now test with QuantileEnsemble on the identical data and configs
    let ensemble = QuantileEnsemble::train(&x, &y, &[0.10, 0.90], &config_p10, None);

    let mut unconstrained_crossings = 0;
    for eval_i in 0..300 {
        let sample = vec![eval_i as f64];
        let raw_p10 = model_p10.predict(&sample);
        let raw_p90 = model_p90.predict(&sample);

        if raw_p10 > raw_p90 {
            unconstrained_crossings += 1;
        }

        let ens_pred = ensemble.predict(&sample);
        assert_strictly_monotone(ens_pred.values(), &format!("sample_{eval_i}"));
        assert!(
            ens_pred.lower() <= ens_pred.upper(),
            "Ensemble lower must be <= upper"
        );
    }

    println!("Unconstrained models crossed {unconstrained_crossings} times out of 300 samples; Ensemble had 0 crossings.");
}
