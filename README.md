# gbt-quantile

Pure-Rust gradient-boosted trees with quantile regression. No BLAS, no C++ bindings, no Python.

The first Rust GBT crate with pinball loss for probabilistic prediction intervals.

## Features

- **Quantile regression** — train P10, P25, P50, P75, P90 models with pinball loss
- **Early stopping** — monitors validation loss, stops when overfitting
- **JSON serialization** — save and load models as human-readable JSON
- **Quantile ensemble** — train multiple quantile models in one call with monotonicity enforcement
- **Evaluation metrics** — MAE, RMSE, R², MAPE, pinball loss
- **High Performance** — global histogram thresholding and Rayon multi-threading
- **Zero dependencies** beyond serde and rayon — no BLAS, LAPACK, or C++ libraries
- **Schema versioned** — forward-compatible model format

## Quick Start

```rust
use gbt_quantile::{GBTConfig, GradientBoostedTree, QuantileEnsemble};
use gbt_quantile::trainer;

// Prepare data: x = feature matrix (row-major), y = targets
let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
let y = vec![2.1, 4.0, 6.1, 7.9, 10.0];

// Train a single model (L2 loss)
let config = GBTConfig::default();
let model = trainer::train(&x, &y, &config, None);
let prediction = model.predict(&[3.5]);

// Train a quantile ensemble (P10-P90)
let ensemble = QuantileEnsemble::train(&x, &y, &[0.1, 0.5, 0.9], &config, None);
let intervals = ensemble.predict(&[3.5]);
println!("P10={:.2}, P50={:.2}, P90={:.2}",
    intervals.lower(), intervals.median(), intervals.upper());

// Save and load
let json = model.to_json().unwrap();
let restored = GradientBoostedTree::from_json(&json).unwrap();
```

## Quantile Regression

Train separate models for different quantiles using pinball loss:

```rust
let config = GBTConfig {
    quantile: Some(0.1),  // P10 — conservative lower bound
    ..Default::default()
};
let p10_model = trainer::train(&x, &y, &config, None);

let config = GBTConfig {
    quantile: Some(0.9),  // P90 — optimistic upper bound
    ..Default::default()
};
let p90_model = trainer::train(&x, &y, &config, None);
```

Or use `QuantileEnsemble` to train all quantiles at once with automatic monotonicity enforcement (P10 <= P25 <= ... <= P90).

## Early Stopping

Pass validation data to prevent overfitting:

```rust
let model = trainer::train_with_validation(
    &x_train, &y_train,
    &x_val, &y_val,
    &config,
    None,
);
// Model automatically stops when validation loss plateaus
println!("Trained {} trees (max was {})", model.n_trees(), config.n_trees);
```

## Evaluation

```rust
use gbt_quantile::metrics;

let preds = model.predict_batch(&x_test);
let m = metrics::evaluate(&y_test, &preds).unwrap();
println!("MAE={:.3}, RMSE={:.3}, R²={:.3}", m.mae, m.rmse, m.r2);

let loss = metrics::pinball_loss(&y_test, &preds, 0.9).unwrap();
println!("Pinball loss (q=0.9): {:.4}", loss);
```

## Algorithm

- **Split finding**: Percentile-based thresholds (configurable bins per feature)
- **Gain**: Variance reduction
- **Quantile loss**: Pinball gradient — `if error >= 0 { q } else { q - 1 }`
- **Early stopping**: Validation loss checked every 5 rounds
- **Base score**: mean(y) for L2, quantile(y, q) for pinball

## License

MIT OR Apache-2.0
