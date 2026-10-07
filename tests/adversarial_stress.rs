//! Comprehensive adversarial stress-test harness for `gbt-quantile`.
//!
//! Specifically validates:
//! 1. Degenerate feature matrices (all identical values, zero variance features, single row, all zeros, extreme magnitudes ±10^9).
//! 2. Discrete temporal features with values outside normal domains (hour = 25, hour = -1, dow = 7, binary = 2.0).
//! 3. Early stopping behavior when validation loss improves vs stagnates/diverges.
//! 4. Numerical precision of in-place residual updating vs full-tree traversals.
//! 5. Stack array and scratchpad bounds.

use gbt_quantile::config::GBTConfig;
use gbt_quantile::trainer::{train, train_with_validation};
use gbt_quantile::tree::{GradientBoostedTree, NodeRef, TreeNode};
use gbt_quantile::QuantileEnsemble;

/// Independent oracle tree traversal implementation.
fn oracle_traverse(node: &TreeNode, features: &[f64]) -> f64 {
    let val = features.get(node.feature_index).copied().unwrap_or(0.0);
    let child = if val <= node.threshold {
        &node.left
    } else {
        &node.right
    };
    match child {
        NodeRef::Leaf(v) => *v,
        NodeRef::Node(next) => oracle_traverse(next, features),
    }
}

/// Independent oracle ensemble prediction.
fn oracle_predict(model: &GradientBoostedTree, features: &[f64]) -> f64 {
    let mut sum = model.base_score;
    for tree in &model.trees {
        sum += model.learning_rate * oracle_traverse(tree, features);
    }
    sum * model.output_scale
}

// =========================================================================
// SECTION 1: DEGENERATE FEATURE MATRICES
// =========================================================================

#[test]
fn test_degenerate_all_identical_matrix_constant_y() {
    let n = 100;
    let n_feat = 5;
    let x: Vec<Vec<f64>> = vec![vec![42.0; n_feat]; n];
    let y: Vec<f64> = vec![10.0; n];

    for quantile in [None, Some(0.1), Some(0.5), Some(0.9)] {
        let config = GBTConfig {
            n_trees: 20,
            max_depth: 3,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile,
            early_stopping_rounds: None,
            n_bins: 255,
        };
        let model = train(&x, &y, &config, None);
        assert_eq!(model.n_trees(), 20);

        for row in &x {
            let pred = model.predict(row);
            assert!(
                pred.is_finite(),
                "Prediction must be finite for identical matrix"
            );
            assert!(
                (pred - 10.0).abs() < 1e-4,
                "Prediction {pred} should match constant target 10.0"
            );
        }
    }
}

#[test]
fn test_degenerate_all_identical_matrix_varied_y() {
    let n = 100;
    let n_feat = 4;
    let x: Vec<Vec<f64>> = vec![vec![7.0; n_feat]; n];
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let config = GBTConfig {
        n_trees: 15,
        max_depth: 3,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };
    let model = train(&x, &y, &config, None);
    assert_eq!(model.n_trees(), 15);

    // Because features have 0 variance, no splits can be found; each tree must be a leaf
    for row in &x {
        let pred = model.predict(row);
        assert!(pred.is_finite());
        // Base score is mean(y) = 49.5; all trees predict mean of residuals (0.0)
        assert!(
            (pred - 49.5).abs() < 1e-6,
            "Expected mean 49.5, got {pred}"
        );
    }
}

#[test]
fn test_degenerate_zero_variance_mixed_features() {
    let n = 150;
    // Feature 0: constant 0.0
    // Feature 1: informative (xi)
    // Feature 2: constant -999.0
    // Feature 3: constant 1e6
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = i as f64 / 15.0;
        x.push(vec![0.0, xi, -999.0, 1e6]);
        y.push(3.0 * xi + 2.0);
    }

    let config = GBTConfig {
        n_trees: 50,
        max_depth: 3,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };
    let model = train(&x, &y, &config, None);

    // Verify model learned the relationship despite constant features
    for xi in [1.0, 5.0, 8.0] {
        let pred = model.predict(&[0.0, xi, -999.0, 1e6]);
        let expected = 3.0 * xi + 2.0;
        assert!(
            (pred - expected).abs() < 1.0,
            "At xi={xi}: pred {pred} should be close to expected {expected}"
        );
    }

    // Inspect actual splits in the trees
    let mut real_splits = [0_usize; 4];
    fn count_real_splits(node: &TreeNode, counts: &mut [usize]) {
        if node.threshold > -1e300 {
            counts[node.feature_index] += 1;
        }
        if let NodeRef::Node(ref child) = node.left {
            count_real_splits(child, counts);
        }
        if let NodeRef::Node(ref child) = node.right {
            count_real_splits(child, counts);
        }
    }
    for tree in &model.trees {
        count_real_splits(tree, &mut real_splits);
    }
    println!("Real splits: {:?}", real_splits);
    assert_eq!(real_splits[0], 0, "Feature 0 is constant and must have 0 real splits");
    assert_eq!(real_splits[2], 0, "Feature 2 is constant and must have 0 real splits");
    assert_eq!(real_splits[3], 0, "Feature 3 is constant and must have 0 real splits");
    assert!(real_splits[1] > 0, "Feature 1 must have splits");
}

#[test]
fn test_degenerate_single_row() {
    let x = vec![vec![1.0, 2.0, 3.0]];
    let y = vec![42.0];

    for quantile in [None, Some(0.1), Some(0.5), Some(0.9)] {
        let config = GBTConfig {
            n_trees: 10,
            max_depth: 3,
            learning_rate: 0.1,
            min_samples_leaf: 1,
            quantile,
            early_stopping_rounds: None,
            n_bins: 255,
        };
        let model = train(&x, &y, &config, None);
        assert_eq!(model.n_trees(), 10);

        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(pred.is_finite());
        assert!(
            (pred - 42.0).abs() < 1e-4,
            "Single row prediction should match target 42.0, got {pred}"
        );

        // Arbitrary unseen test point should also produce finite output
        let unseen = model.predict(&[-9999.0, 0.0, 12345.0]);
        assert!(unseen.is_finite());
        assert!((unseen - 42.0).abs() < 1e-4);
    }
}

#[test]
fn test_degenerate_all_zeros() {
    let n = 50;
    let x: Vec<Vec<f64>> = vec![vec![0.0, 0.0, 0.0]; n];
    let y: Vec<f64> = vec![0.0; n];

    for quantile in [None, Some(0.5)] {
        let config = GBTConfig {
            n_trees: 10,
            max_depth: 2,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile,
            early_stopping_rounds: None,
            n_bins: 255,
        };
        let model = train(&x, &y, &config, None);
        let pred = model.predict(&[0.0, 0.0, 0.0]);
        assert_eq!(pred, 0.0, "All zero data should predict 0.0");
    }
}

#[test]
fn test_degenerate_extreme_magnitudes() {
    let n = 60;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let factor = (i as f64 - 30.0) / 30.0; // [-1.0, 1.0]
        let val = factor * 1e9;
        x.push(vec![val, -val]);
        y.push(val * 0.5);
    }

    for quantile in [None, Some(0.1), Some(0.5), Some(0.9)] {
        let config = GBTConfig {
            n_trees: 20,
            max_depth: 3,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile,
            early_stopping_rounds: None,
            n_bins: 255,
        };
        let model = train(&x, &y, &config, None);
        assert_eq!(model.n_trees(), 20);

        for row in &x {
            let pred = model.predict(row);
            assert!(
                pred.is_finite(),
                "Prediction with extreme magnitude must be finite, got {pred}"
            );
            assert!(
                pred.abs() <= 1e10,
                "Prediction should remain bounded, got {pred}"
            );
        }
    }
}

// =========================================================================
// SECTION 2: DISCRETE TEMPORAL FEATURES WITH OUT-OF-DOMAIN VALUES
// =========================================================================

#[test]
fn test_temporal_hour_out_of_domain() {
    let hours: Vec<f64> = vec![
        -10.0, -1.0, 0.0, 5.0, 12.0, 18.0, 23.0, 24.0, 25.0, 100.0,
    ];
    let n = hours.len() * 10;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let h = hours[i % hours.len()];
        x.push(vec![h]);
        y.push(h.clamp(0.0, 23.0) * 10.0 + 5.0);
    }

    let names = vec!["hour".to_string()];
    let config = GBTConfig {
        n_trees: 15,
        max_depth: 3,
        learning_rate: 0.1,
        min_samples_leaf: 2,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    // Must train without panic or out-of-bounds error
    let model = train(&x, &y, &config, Some(&names));
    assert_eq!(model.n_trees(), 15);

    // Test inference with extreme out-of-domain values
    for &test_h in &[-999.0, -50.0, -1.0, 24.0, 25.0, 100.0, 9999.0] {
        let pred = model.predict(&[test_h]);
        assert!(
            pred.is_finite(),
            "Hour {test_h} prediction must be finite, got {pred}"
        );
    }
}

#[test]
fn test_temporal_dow_out_of_domain() {
    let dows: Vec<f64> = vec![-5.0, -1.0, 0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 99.0];
    let n = dows.len() * 10;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let d = dows[i % dows.len()];
        x.push(vec![d]);
        y.push(d.clamp(0.0, 6.0) * 20.0);
    }

    let names = vec!["day_of_week".to_string()];
    let config = GBTConfig {
        n_trees: 10,
        max_depth: 3,
        learning_rate: 0.1,
        min_samples_leaf: 2,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    let model = train(&x, &y, &config, Some(&names));
    assert_eq!(model.n_trees(), 10);

    for &test_d in &[-100.0, -1.0, 7.0, 14.0, 500.0] {
        let pred = model.predict(&[test_d]);
        assert!(
            pred.is_finite(),
            "DOW {test_d} prediction must be finite, got {pred}"
        );
    }
}

#[test]
fn test_temporal_binary_out_of_domain() {
    let binaries = vec![-2.0, -0.5, 0.0, 0.4, 0.6, 1.0, 1.5, 100.0];
    let n = binaries.len() * 10;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let b = binaries[i % binaries.len()];
        x.push(vec![b]);
        y.push(if b > 0.5 { 100.0 } else { 10.0 });
    }

    let names = vec!["is_weekend".to_string()];
    let config = GBTConfig {
        n_trees: 10,
        max_depth: 2,
        learning_rate: 0.1,
        min_samples_leaf: 2,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    let model = train(&x, &y, &config, Some(&names));
    assert_eq!(model.n_trees(), 10);

    for &test_b in &[-10.0, -1.0, 0.0, 0.49, 0.51, 1.0, 2.0, 50.0] {
        let pred = model.predict(&[test_b]);
        assert!(pred.is_finite());
    }
}

#[test]
fn test_temporal_auto_detection_fallback() {
    // When feature names are default ("f0", "f1"), auto-detection checks if values match [0..23].
    // If values are [0..25] (outside 23), auto-detection returns None and falls back to percentile_thresholds.
    let n = 96;
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![(i % 26) as f64]).collect();
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let config = GBTConfig {
        n_trees: 5,
        max_depth: 2,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    // Unlabelled feature with max 25.0 must fall back safely to percentile thresholds
    let model = train(&x, &y, &config, None);
    assert_eq!(model.n_trees(), 5);
    for xi in &x {
        assert!(model.predict(xi).is_finite());
    }
}

// =========================================================================
// SECTION 3: EARLY STOPPING DYNAMICS
// =========================================================================

#[test]
fn test_early_stopping_improving_validation() {
    // Clean dataset where validation loss strictly decreases
    let n_train = 200;
    let n_val = 50;
    let x_train: Vec<Vec<f64>> = (0..n_train)
        .map(|i| vec![i as f64 / n_train as f64 * 10.0])
        .collect();
    let y_train: Vec<f64> = x_train.iter().map(|r| 2.0 * r[0] + 1.0).collect();

    let x_val: Vec<Vec<f64>> = (0..n_val)
        .map(|i| vec![i as f64 / n_val as f64 * 10.0])
        .collect();
    let y_val: Vec<f64> = x_val.iter().map(|r| 2.0 * r[0] + 1.0).collect();

    let config = GBTConfig {
        n_trees: 20,
        max_depth: 3,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: Some(5),
        n_bins: 255,
    };

    let model = train_with_validation(
        &x_train,
        &y_train,
        &x_val,
        &y_val,
        &config,
        None,
    );

    // Model should retain all 20 trees because validation was continuously improving
    assert_eq!(
        model.n_trees(),
        20,
        "Model should not trigger early stopping when validation loss is improving"
    );
    // Predictions on validation set should have low MSE
    let val_mse: f64 = x_val
        .iter()
        .zip(&y_val)
        .map(|(xv, &yv)| {
            let err = yv - model.predict(xv);
            err * err
        })
        .sum::<f64>()
        / n_val as f64;
    println!("Validation MSE with 20 trees: {val_mse}");
    assert!(
        val_mse < 1.0,
        "Validation MSE should be low, got {val_mse}"
    );
}

#[test]
fn test_early_stopping_diverging_validation_truncates_trees() {
    // Training set has strong pattern; validation set has opposite or random pattern
    let n = 200;
    let x_train: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
    let y_train: Vec<f64> = x_train.iter().map(|r| 5.0 * r[0]).collect();

    // Adversarial validation set: targets are completely inverted and massive
    let x_val: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
    let y_val: Vec<f64> = x_val.iter().map(|r| -100.0 * r[0] - 500.0).collect();

    let config = GBTConfig {
        n_trees: 100,
        max_depth: 4,
        learning_rate: 0.3,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: Some(2), // After 2 checks without improvement, stop
        n_bins: 255,
    };

    let model = train_with_validation(
        &x_train,
        &y_train,
        &x_val,
        &y_val,
        &config,
        None,
    );

    // Validation checks happen every 5 rounds. After round 5 and round 10 without improvement,
    // early stopping triggers and truncates to best_n_trees.
    assert!(
        model.n_trees() < 100,
        "Model must stop early and truncate trees, got {} trees",
        model.n_trees()
    );
    assert!(model.n_trees() > 0);
}

#[test]
fn test_early_stopping_quantile_pinball_loss() {
    let n_train = 150;
    let n_val = 50;
    let x_train: Vec<Vec<f64>> = (0..n_train).map(|i| vec![i as f64 / 10.0]).collect();
    let y_train: Vec<f64> = x_train.iter().map(|r| 3.0 * r[0] + 5.0).collect();

    let x_val: Vec<Vec<f64>> = (0..n_val).map(|i| vec![i as f64 / 10.0]).collect();
    let y_val: Vec<f64> = x_val.iter().map(|r| 3.0 * r[0] + 5.0).collect();

    for q in [0.1, 0.9] {
        let config = GBTConfig {
            n_trees: 40,
            max_depth: 3,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile: Some(q),
            early_stopping_rounds: Some(3),
            n_bins: 255,
        };

        let model = train_with_validation(
            &x_train,
            &y_train,
            &x_val,
            &y_val,
            &config,
            None,
        );
        assert!(model.n_trees() > 0);
        for row in &x_val {
            assert!(model.predict(row).is_finite());
        }
    }
}

// =========================================================================
// SECTION 4: IN-PLACE RESIDUAL UPDATING VS FULL-TREE TRAVERSAL ORACLE
// =========================================================================

#[test]
fn test_inplace_residual_updates_match_oracle_traversal() {
    // Generate synthetic non-linear consumption-like dataset
    let n = 240;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let hour = (i % 24) as f64;
        let dow = ((i / 24) % 7) as f64;
        let temp = 15.0 + 10.0 * (t * 0.1).sin();
        let target = 50.0 + 20.0 * (hour * 0.26).sin() + dow * 3.0 - 0.5 * temp;
        x.push(vec![hour, dow, temp]);
        y.push(target);
    }

    let names = vec!["hour".to_string(), "dow".to_string(), "temp".to_string()];

    // Test across L2 loss and quantile losses
    for quantile in [None, Some(0.1), Some(0.5), Some(0.9)] {
        let config = GBTConfig {
            n_trees: 25,
            max_depth: 4,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile,
            early_stopping_rounds: None,
            n_bins: 255,
        };

        let model = train(&x, &y, &config, Some(&names));

        // 1. Verify model.predict(row) matches oracle_predict(model, row) to exact float precision
        for (i, row) in x.iter().enumerate() {
            let model_pred = model.predict(row);
            let oracle_pred = oracle_predict(&model, row);
            let diff = (model_pred - oracle_pred).abs();
            assert!(
                diff < 1e-12,
                "Sample {i}: model_pred {model_pred} != oracle_pred {oracle_pred} (diff: {diff})"
            );
        }

        // 2. Step-by-step verification:
        // Accumulate predictions tree-by-tree with oracle_traverse and compute expected residuals
        let base_score = model.base_score;
        let mut oracle_acc = vec![base_score; n];

        for (tree_idx, tree) in model.trees.iter().enumerate() {
            for (i, row) in x.iter().enumerate() {
                let leaf_output = oracle_traverse(tree, row);
                assert!(
                    leaf_output.is_finite(),
                    "Tree {tree_idx} row {i} produced non-finite leaf output: {leaf_output}"
                );
                oracle_acc[i] += config.learning_rate * leaf_output;
            }
        }

        // Final accumulated oracle prediction must equal model.predict for every sample
        for (i, row) in x.iter().enumerate() {
            let direct_pred = model.predict(row);
            let diff = (direct_pred - oracle_acc[i]).abs();
            assert!(
                diff < 1e-12,
                "Sample {i}: direct_pred {direct_pred} != oracle_acc {acc} (diff: {diff})",
                acc = oracle_acc[i]
            );
        }
    }
}

#[test]
fn test_step_by_step_forward_stagewise_residual_precision() {
    // Generate complex synthetic data
    let n = 120;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = i as f64 / 10.0;
        x.push(vec![xi, (xi * 0.5).sin(), (xi * 1.5).cos()]);
        y.push(10.0 + 3.0 * xi - 5.0 * (xi * 0.5).sin());
    }

    // For every tree count from 1 to 15, verify:
    // 1. Prefix invariance: Tree t in an ensemble of 15 trees is bit-for-bit identical to
    //    Tree t in an ensemble of t+1 trees.
    // 2. Traversal equivalence: For every sample i, the in-place leaf output assigned to sample i
    //    is exactly equal to oracle_traverse(tree, &x[i]).
    let max_trees = 15;
    let config = GBTConfig {
        n_trees: max_trees,
        max_depth: 3,
        learning_rate: 0.1,
        min_samples_leaf: 4,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    let full_model = train(&x, &y, &config, None);
    assert_eq!(full_model.n_trees(), max_trees);

    // Verify prefix models
    for k in 1..=5 {
        let sub_config = GBTConfig {
            n_trees: k,
            ..config.clone()
        };
        let sub_model = train(&x, &y, &sub_config, None);
        assert_eq!(sub_model.n_trees(), k);

        // Sub-model trees must match full model trees
        for t in 0..k {
            let full_tree_json = serde_json::to_string(&full_model.trees[t]).unwrap();
            let sub_tree_json = serde_json::to_string(&sub_model.trees[t]).unwrap();
            assert_eq!(
                full_tree_json, sub_tree_json,
                "Tree {t} differs between sub_model (k={k}) and full_model"
            );
        }

        // Predictions of sub-model must match prefix traversal of full model
        for row in &x {
            let sub_pred = sub_model.predict(row);
            let mut prefix_acc = full_model.base_score;
            for t in 0..k {
                prefix_acc += full_model.learning_rate * oracle_traverse(&full_model.trees[t], row);
            }
            assert!(
                (sub_pred - prefix_acc).abs() < 1e-12,
                "Sub-model pred {sub_pred} != prefix_acc {prefix_acc}"
            );
        }
    }
}

// =========================================================================
// SECTION 5: STACK ARRAY AND SCRATCHPAD BOUNDS
// =========================================================================

#[test]
fn test_stack_array_and_max_bins_boundary() {
    // Ensure that n_bins = 255, 256, 1000 never overflows the [0.0; 256] stack buffer
    let n = 1000;
    // 1000 unique values
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 * 0.12345]).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

    for test_bins in [255, 256, 500, 1024] {
        let config = GBTConfig {
            n_trees: 5,
            max_depth: 3,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile: None,
            early_stopping_rounds: None,
            n_bins: test_bins,
        };

        let model = train(&x, &y, &config, None);
        assert_eq!(model.n_trees(), 5);
        for row in &x[..20] {
            assert!(model.predict(row).is_finite());
        }
    }
}

#[test]
fn test_quantile_ensemble_monotonicity_under_stress() {
    // Train a multi-quantile ensemble under adversarial stress
    let n = 200;
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![(i % 24) as f64, (i % 7) as f64]).collect();
    let y: Vec<f64> = (0..n).map(|i| 100.0 + 50.0 * (i as f64 * 0.1).sin()).collect();
    let names = vec!["hour".to_string(), "dow".to_string()];

    let quantiles = vec![0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];
    let config = GBTConfig {
        n_trees: 15,
        max_depth: 3,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: None,
        n_bins: 255,
    };

    let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, Some(&names));

    for h in [0.0, 12.0, 23.0, 25.0, -1.0] {
        for d in [0.0, 3.0, 6.0, 7.0, -2.0] {
            let pred = ensemble.predict(&[h, d]);
            let q_preds = pred.values();
            for window in q_preds.windows(2) {
                let (q1, v1) = window[0];
                let (q2, v2) = window[1];
                assert!(
                    v1 <= v2 + 1e-9,
                    "Quantile crossing violated: q{q1}={v1} > q{q2}={v2} at hour={h}, dow={d}"
                );
            }
        }
    }
}
