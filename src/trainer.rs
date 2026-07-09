//! Core gradient-boosted tree training algorithm.
//!
//! Supports squared-error (L2) loss for mean regression and pinball loss
//! for quantile regression. Includes early stopping with periodic
//! validation checks and automatic tree truncation.
//!
//! Training is histogram-based: each feature is quantized to at most 255
//! bins once up front, so finding the best split at a node costs a single
//! O(n) pass per feature instead of rescanning all rows per candidate
//! threshold. Nodes track row indices rather than copying feature rows.

use crate::config::GBTConfig;
use crate::tree::{traverse_node, GradientBoostedTree, NodeRef, TreeNode};
use rayon::prelude::*;

/// Pre-binned training data: per-feature bin indices and the threshold
/// edges separating consecutive bins.
struct BinnedData {
    /// `bins[f][i]` = bin index of row `i` for feature `f` (column-major).
    bins: Vec<Vec<u8>>,
    /// `edges[f][b]` = value threshold separating bin `b` from bin `b + 1`.
    /// A split "bin <= b" is equivalent to "value <= edges[f][b]".
    edges: Vec<Vec<f64>>,
}

fn bin_data(x: &[Vec<f64>], n_features: usize, n_bins: usize) -> BinnedData {
    let n_bins = n_bins.min(255); // bin indices must fit in u8
    let (bins, edges): (Vec<Vec<u8>>, Vec<Vec<f64>>) = (0..n_features)
        .into_par_iter()
        .map(|f| {
            let edges = percentile_thresholds(x, f, n_bins);
            let bins = x
                .iter()
                .map(|row| edges.partition_point(|&t| row[f] > t) as u8)
                .collect();
            (bins, edges)
        })
        .unzip();
    BinnedData { bins, edges }
}

/// Train a gradient-boosted tree model.
///
/// If `feature_names` is `None`, default names `["f0", "f1", ...]` are generated.
/// For early stopping, use [`train_with_validation`] instead.
pub fn train(
    x: &[Vec<f64>],
    y: &[f64],
    config: &GBTConfig,
    feature_names: Option<&[String]>,
) -> GradientBoostedTree {
    train_with_validation(x, y, &[], &[], config, feature_names)
}

/// Train a gradient-boosted tree model with optional validation data for early stopping.
///
/// If `x_val` and `y_val` are empty, no early stopping is performed.
/// If `feature_names` is `None`, default names `["f0", "f1", ...]` are generated.
///
/// For quantile regression (`config.quantile = Some(q)`):
/// - Pseudo-residuals are `q` for positive errors, `q - 1` for negative errors.
/// - Leaf values are the target quantile of the residuals (not the mean).
/// - Validation loss uses pinball loss instead of MSE.
pub fn train_with_validation(
    x: &[Vec<f64>],
    y: &[f64],
    x_val: &[Vec<f64>],
    y_val: &[f64],
    config: &GBTConfig,
    feature_names: Option<&[String]>,
) -> GradientBoostedTree {
    let n = y.len();
    let n_features = x.first().map_or(0, |r| r.len());

    let names: Vec<String> = match feature_names {
        Some(names) => names.to_vec(),
        None => (0..n_features).map(|i| format!("f{i}")).collect(),
    };

    if n == 0 {
        return GradientBoostedTree {
            schema_version: 1,
            trees: vec![],
            base_score: 0.0,
            learning_rate: config.learning_rate,
            feature_names: names,
            quantile: config.quantile,
            output_scale: 1.0,
            metadata: None,
        };
    }

    // Quantize features once; every tree reuses the same bins.
    let binned = bin_data(x, n_features, config.n_bins);
    let all_indices: Vec<usize> = (0..n).collect();

    // Initialize: base_score = mean(y) for L2, quantile for pinball
    let base_score = if let Some(q) = config.quantile {
        quantile_value(y, q)
    } else {
        y.iter().sum::<f64>() / n as f64
    };

    // Residuals: what the model still needs to learn
    let mut residuals: Vec<f64> = y.iter().map(|yi| yi - base_score).collect();
    let mut trees = Vec::with_capacity(config.n_trees);

    // Early stopping state
    let has_validation = !x_val.is_empty() && !y_val.is_empty();
    let mut best_val_loss = f64::MAX;
    let mut rounds_without_improvement = 0_usize;
    let mut best_n_trees = 0_usize;

    for round in 0..config.n_trees {
        // For quantile loss, compute pseudo-residuals
        let pseudo_residuals = if let Some(q) = config.quantile {
            residuals
                .iter()
                .map(|&r| if r >= 0.0 { q } else { q - 1.0 })
                .collect::<Vec<_>>()
        } else {
            residuals.clone() // L2: gradient = residual
        };

        // Build one tree to fit the pseudo-residuals
        let tree = build_tree(
            &all_indices,
            &pseudo_residuals,
            &residuals,
            config,
            0,
            &binned,
        );

        // Update residuals
        for i in 0..n {
            let pred = traverse_node(&tree, &x[i]);
            residuals[i] -= config.learning_rate * pred;
        }

        trees.push(tree);

        // Early stopping: check validation loss periodically
        if let Some(es_rounds) = config.early_stopping_rounds {
            if has_validation && ((round + 1) % 5 == 0 || round == config.n_trees - 1) {
                let val_loss = compute_validation_loss(
                    x_val,
                    y_val,
                    &trees,
                    base_score,
                    config.learning_rate,
                    config.quantile,
                );

                if val_loss < best_val_loss - 1e-8 {
                    best_val_loss = val_loss;
                    best_n_trees = trees.len();
                    rounds_without_improvement = 0;
                } else {
                    rounds_without_improvement += 1;
                }

                if rounds_without_improvement >= es_rounds {
                    trees.truncate(best_n_trees);
                    break;
                }
            }
        }
    }

    GradientBoostedTree {
        schema_version: 1,
        trees,
        base_score,
        learning_rate: config.learning_rate,
        feature_names: names,
        quantile: config.quantile,
        output_scale: 1.0,
        metadata: Some(serde_json::json!({
            "trainer": "gbt-quantile",
            "n_samples": n,
        })),
    }
}

/// Build a single decision tree to fit pseudo-residuals.
///
/// `indices` are the rows belonging to this node; child nodes get
/// partitioned index sets rather than copies of the feature rows.
fn build_tree(
    indices: &[usize],
    pseudo_residuals: &[f64],
    residuals: &[f64],
    config: &GBTConfig,
    depth: usize,
    binned: &BinnedData,
) -> TreeNode {
    let n = indices.len();
    let n_features = binned.bins.len();

    // Leaf conditions: max depth reached, too few samples, or no features
    if depth >= config.max_depth || n <= config.min_samples_leaf * 2 || n_features == 0 {
        return make_leaf_node(leaf_value(indices, residuals, config.quantile));
    }

    let total_sum: f64 = indices.iter().map(|&i| pseudo_residuals[i]).sum();
    let total_count = n as f64;
    let min_count = config.min_samples_leaf as f64;

    // Find the best split: one histogram pass per feature, in parallel.
    let best_split = (0..n_features)
        .into_par_iter()
        .filter_map(|feat_idx| {
            let edges = &binned.edges[feat_idx];
            let n_edges = edges.len();
            if n_edges == 0 {
                return None; // constant feature
            }

            let feature_bins = &binned.bins[feat_idx];
            let mut sums = vec![0.0_f64; n_edges + 1];
            let mut counts = vec![0.0_f64; n_edges + 1];
            for &i in indices {
                let b = feature_bins[i] as usize;
                sums[b] += pseudo_residuals[i];
                counts[b] += 1.0;
            }

            // Cumulative scan: split after bin b means "value <= edges[b]" goes left.
            let mut left_sum = 0.0;
            let mut left_count = 0.0;
            let mut local_best: Option<(f64, usize)> = None; // (gain, edge index)
            for b in 0..n_edges {
                left_sum += sums[b];
                left_count += counts[b];
                let right_sum = total_sum - left_sum;
                let right_count = total_count - left_count;

                if left_count < min_count || right_count < min_count {
                    continue;
                }

                // Gain = reduction in variance
                let gain = (left_sum * left_sum / left_count)
                    + (right_sum * right_sum / right_count)
                    - (total_sum * total_sum / total_count);

                if gain > local_best.map_or(0.0, |(g, _)| g) {
                    local_best = Some((gain, b));
                }
            }

            local_best.map(|(gain, b)| (gain, feat_idx, b))
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((gain, best_feature, best_edge)) = best_split {
        if gain > 0.0 {
            let feature_bins = &binned.bins[best_feature];
            let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
                .iter()
                .partition(|&&i| (feature_bins[i] as usize) <= best_edge);

            // min_samples_leaf in the gain scan guarantees both sides are non-empty
            let left = NodeRef::Node(Box::new(build_tree(
                &left_idx,
                pseudo_residuals,
                residuals,
                config,
                depth + 1,
                binned,
            )));
            let right = NodeRef::Node(Box::new(build_tree(
                &right_idx,
                pseudo_residuals,
                residuals,
                config,
                depth + 1,
                binned,
            )));

            return TreeNode {
                feature_index: best_feature,
                threshold: binned.edges[best_feature][best_edge],
                left,
                right,
            };
        }
    }

    // Fallback to leaf
    make_leaf_node(leaf_value(indices, residuals, config.quantile))
}

/// Compute the leaf prediction value: quantile of residuals or mean.
fn leaf_value(indices: &[usize], residuals: &[f64], quantile: Option<f64>) -> f64 {
    let node_residuals: Vec<f64> = indices.iter().map(|&i| residuals[i]).collect();
    if let Some(q) = quantile {
        quantile_value(&node_residuals, q)
    } else {
        mean(&node_residuals)
    }
}

/// Make a leaf-only tree node.
///
/// Since `TreeNode` always has a split structure, we use a sentinel threshold
/// of `-1e308` (effectively negative infinity, but JSON-safe) so that all
/// inputs go to the left child, which holds the leaf value.
fn make_leaf_node(value: f64) -> TreeNode {
    let safe_val = if value.is_finite() { value } else { 0.0 };
    TreeNode {
        feature_index: 0,
        threshold: -1e308,
        left: NodeRef::Leaf(safe_val),
        right: NodeRef::Leaf(safe_val),
    }
}

/// Get percentile-based threshold candidates for a feature.
///
/// Sorts unique feature values and picks `n_bins` evenly spaced midpoints.
/// Returns an empty vec if the feature is constant. The returned thresholds
/// are strictly increasing and double as histogram bin edges.
fn percentile_thresholds(x: &[Vec<f64>], feat_idx: usize, n_bins: usize) -> Vec<f64> {
    let mut values: Vec<f64> = x.iter().map(|row| row[feat_idx]).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup();

    if values.len() <= 1 {
        return vec![];
    }

    let step = (values.len() as f64 / (n_bins + 1) as f64).max(1.0);
    let mut thresholds = Vec::with_capacity(n_bins);
    for i in 1..=n_bins {
        let idx = ((i as f64 * step) as usize).min(values.len() - 2);
        thresholds.push((values[idx] + values[idx + 1]) / 2.0);
    }
    thresholds.dedup();
    thresholds
}

/// Compute the mean of a slice.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute validation loss for the current ensemble.
///
/// Uses MSE for L2 loss, pinball loss for quantile regression.
fn compute_validation_loss(
    x_val: &[Vec<f64>],
    y_val: &[f64],
    trees: &[TreeNode],
    base_score: f64,
    learning_rate: f64,
    quantile: Option<f64>,
) -> f64 {
    let n = y_val.len();
    if n == 0 {
        return f64::MAX;
    }

    let mut total_loss = 0.0;
    for i in 0..n {
        let mut pred = base_score;
        for tree in trees {
            pred += learning_rate * traverse_node(tree, &x_val[i]);
        }
        let error = y_val[i] - pred;
        total_loss += if let Some(q) = quantile {
            // Pinball loss
            if error >= 0.0 {
                q * error
            } else {
                (q - 1.0) * error
            }
        } else {
            // MSE
            error * error
        };
    }

    total_loss / n as f64
}

/// Compute a quantile value from a slice using linear interpolation.
///
/// For example, `quantile_value(&[1.0, 2.0, 3.0, 4.0], 0.5)` returns 2.5.
pub(crate) fn quantile_value(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = (sorted.len() as f64 - 1.0) * q;
    let idx = pos.floor() as usize;
    let frac = pos - idx as f64;
    let upper_idx = (idx + 1).min(sorted.len() - 1);
    sorted[idx] * (1.0 - frac) + sorted[upper_idx] * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GBTConfig;

    /// Generate data: y = 2*x + 1 + small noise.
    fn linear_data(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let xi = i as f64 / n as f64 * 10.0;
            x.push(vec![xi]);
            // Deterministic "noise" based on index
            let noise = ((i * 7 + 3) % 11) as f64 / 55.0 - 0.1;
            y.push(2.0 * xi + 1.0 + noise);
        }
        (x, y)
    }

    #[test]
    fn test_train_learns_linear() {
        let (x, y) = linear_data(200);
        let config = GBTConfig {
            n_trees: 200,
            max_depth: 4,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile: None,
            early_stopping_rounds: None,
            n_bins: 255,
        };
        let model = train(&x, &y, &config, None);
        assert!(model.n_trees() > 0, "Model should have trees");

        // Check predictions at a few points
        for &xi in &[1.0, 3.0, 5.0, 7.0, 9.0] {
            let pred = model.predict(&[xi]);
            let expected = 2.0 * xi + 1.0;
            assert!(
                (pred - expected).abs() < 0.5,
                "At x={xi}, predicted {pred}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn test_quantile_p10_below_p90() {
        let (x, y) = linear_data(300);
        let config_p10 = GBTConfig {
            n_trees: 100,
            max_depth: 4,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile: Some(0.1),
            early_stopping_rounds: None,
            n_bins: 255,
        };
        let config_p90 = GBTConfig {
            quantile: Some(0.9),
            ..config_p10.clone()
        };

        let model_p10 = train(&x, &y, &config_p10, None);
        let model_p90 = train(&x, &y, &config_p90, None);

        // P10 should predict lower than P90 across the range
        for &xi in &[2.0, 5.0, 8.0] {
            let p10 = model_p10.predict(&[xi]);
            let p90 = model_p90.predict(&[xi]);
            assert!(
                p10 <= p90 + 0.01,
                "At x={xi}: P10={p10} should be <= P90={p90}"
            );
        }
    }

    #[test]
    fn test_early_stopping_works() {
        let (x, y) = linear_data(200);
        // Use first 80% as train, rest as validation
        let split = (x.len() as f64 * 0.8) as usize;
        let (x_train, x_val) = x.split_at(split);
        let (y_train, y_val) = y.split_at(split);

        let config = GBTConfig {
            n_trees: 500,
            max_depth: 4,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile: None,
            early_stopping_rounds: Some(3),
            n_bins: 255,
        };

        let model = train_with_validation(x_train, y_train, x_val, y_val, &config, None);
        // Early stopping should have kicked in before 500 trees
        assert!(
            model.n_trees() < 500,
            "Expected early stopping before 500 trees, got {} trees",
            model.n_trees()
        );
        assert!(model.n_trees() > 0, "Model should have at least 1 tree");
    }

    #[test]
    fn test_empty_data() {
        let config = GBTConfig::default();
        let model = train(&[], &[], &config, None);
        assert_eq!(model.n_trees(), 0);
        assert_eq!(model.base_score, 0.0);
    }

    #[test]
    fn test_feature_names_default() {
        let x = vec![vec![1.0, 2.0, 3.0]; 20];
        let y = vec![1.0; 20];
        let config = GBTConfig {
            n_trees: 1,
            min_samples_leaf: 1,
            ..GBTConfig::default()
        };
        let model = train(&x, &y, &config, None);
        assert_eq!(model.feature_names, vec!["f0", "f1", "f2"]);
    }

    #[test]
    fn test_feature_names_custom() {
        let x = vec![vec![1.0, 2.0]; 20];
        let y = vec![1.0; 20];
        let names = vec!["temperature".to_string(), "humidity".to_string()];
        let config = GBTConfig {
            n_trees: 1,
            min_samples_leaf: 1,
            ..GBTConfig::default()
        };
        let model = train(&x, &y, &config, Some(&names));
        assert_eq!(model.feature_names, names);
    }

    #[test]
    fn test_quantile_value_interpolation() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert!((quantile_value(&values, 0.0) - 1.0).abs() < 1e-10);
        assert!((quantile_value(&values, 0.5) - 2.5).abs() < 1e-10);
        assert!((quantile_value(&values, 1.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_value_empty() {
        assert_eq!(quantile_value(&[], 0.5), 0.0);
    }

    #[test]
    fn test_percentile_thresholds_constant() {
        let x = vec![vec![5.0]; 10];
        let thresholds = percentile_thresholds(&x, 0, 10);
        assert!(
            thresholds.is_empty(),
            "Constant feature should produce no thresholds"
        );
    }

    #[test]
    fn test_binning_roundtrip() {
        // Rows with bin index <= b must be exactly the rows with value <= edges[b]
        let x: Vec<Vec<f64>> = (0..100).map(|i| vec![(i % 17) as f64]).collect();
        let binned = bin_data(&x, 1, 8);
        let edges = &binned.edges[0];
        assert!(!edges.is_empty());
        for (i, row) in x.iter().enumerate() {
            for (b, &edge) in edges.iter().enumerate() {
                assert_eq!(
                    (binned.bins[0][i] as usize) <= b,
                    row[0] <= edge,
                    "row {i} value {} edge {edge}",
                    row[0]
                );
            }
        }
    }
}
