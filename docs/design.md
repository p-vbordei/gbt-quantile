# gbt-quantile — Design Spec

## Overview
Pure-Rust gradient-boosted tree library with quantile regression. First of its kind — no BLAS, no C++ bindings, no Python. Trains GBT models with pinball loss for probabilistic prediction intervals (P10-P90).

## Public API

### Core Training
```rust
use gbt_quantile::{GradientBoostedTree, GBTConfig};

let config = GBTConfig::default(); // 100 trees, depth 4, lr 0.05
let model = GradientBoostedTree::train(&x, &y, &config);

// With validation + early stopping
let model = GradientBoostedTree::train_with_validation(&x, &y, &x_val, &y_val, &config);
```

### Quantile Ensemble
```rust
use gbt_quantile::{QuantileEnsemble, GBTConfig};

let config = GBTConfig { n_trees: 100, max_depth: 4, learning_rate: 0.05, ..Default::default() };
let ensemble = QuantileEnsemble::train(&x, &y, &[0.1, 0.25, 0.5, 0.75, 0.9], &config);

let pred = ensemble.predict(&features);
// pred.p(0.5) → median prediction
// pred.values() → [(0.1, val), (0.25, val), ...]
```

### Prediction
```rust
let value = model.predict(&features);       // single sample
let batch = model.predict_batch(&matrix);   // batch of samples
```

### Serialization (JSON)
```rust
let json_str = model.to_json()?;
let model = GradientBoostedTree::from_json(&json_str)?;

let bytes = model.to_bytes()?;
let model = GradientBoostedTree::from_bytes(&bytes)?;
```

### Evaluation Metrics
```rust
use gbt_quantile::metrics;

let m = metrics::evaluate(&y_true, &y_pred);
// m.mae, m.rmse, m.r2, m.mape

let loss = metrics::pinball_loss(&y_true, &y_pred, 0.9);
```

### Data Utilities
```rust
use gbt_quantile::split::train_test_split;

let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.8, Some(42));
let importance = model.feature_importance(); // HashMap<usize, f64>
```

## Module Structure

```
src/
  lib.rs          — re-exports, crate docs
  config.rs       — GBTConfig
  tree.rs         — TreeNode, NodeRef, GradientBoostedTree (model + predict + serialize)
  trainer.rs      — train(), build_tree(), split finding
  ensemble.rs     — QuantileEnsemble, QuantilePrediction, monotonicity enforcement
  metrics.rs      — evaluate(), pinball_loss(), Metrics struct
  split.rs        — train_test_split()
```

## Types

### GBTConfig
```rust
pub struct GBTConfig {
    pub n_trees: usize,                    // default: 100
    pub max_depth: usize,                  // default: 4
    pub learning_rate: f64,                // default: 0.05
    pub min_samples_leaf: usize,           // default: 10
    pub quantile: Option<f64>,             // None = L2, Some(q) = pinball loss
    pub early_stopping_rounds: Option<usize>, // default: Some(5)
    pub n_bins: usize,                     // default: 10 (split candidates per feature)
}
```

### GradientBoostedTree
JSON-serializable model. Contains trees, base_score, learning_rate, feature metadata.
Schema versioned for forward compatibility.

### QuantileEnsemble
Wraps multiple GradientBoostedTree models (one per quantile). Enforces monotonicity on predictions (P10 <= P25 <= ... <= P90).

### QuantilePrediction
```rust
pub struct QuantilePrediction {
    values: Vec<(f64, f64)>,  // (quantile, prediction) pairs, sorted
}
impl QuantilePrediction {
    pub fn p(&self, quantile: f64) -> Option<f64>;
    pub fn values(&self) -> &[(f64, f64)];
    pub fn median(&self) -> f64;
    pub fn lower(&self) -> f64;  // lowest quantile
    pub fn upper(&self) -> f64;  // highest quantile
}
```

### Metrics
```rust
pub struct Metrics {
    pub mae: f64,
    pub rmse: f64,
    pub r2: f64,
    pub mape: f64,
    pub n_samples: usize,
}
```

## Algorithm

- **Split finding**: Percentile-based thresholds (n_bins candidates per feature)
- **Gain**: Variance reduction (sum²/count)
- **Quantile loss**: Pinball gradient — `if error >= 0 { q } else { q - 1 }`
- **Leaf values**: Mean (L2) or quantile (pinball) of residuals
- **Early stopping**: Validation loss checked every 5 rounds, stops after N checks without improvement
- **Base score**: mean(y) for L2, quantile(y, q) for pinball

## Dependencies
- `serde` (derive)
- `serde_json`
- `anyhow`

Zero numeric libraries. Pure `f64` + `Vec` operations.

## Testing
- Unit tests per module
- Integration: train → predict → serialize roundtrip
- Property: quantile monotonicity
- Correctness: known datasets (y = 2x + 1)
- Edge cases: empty data, single row, all-same target, NaN rejection
