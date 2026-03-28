//! # gbt-quantile
//!
//! Pure-Rust gradient-boosted trees with quantile regression.
//!
//! This crate provides a complete GBT implementation with:
//! - **Squared-error (L2) loss** for mean regression
//! - **Pinball loss** for quantile regression (P10, P25, P50, P75, P90, etc.)
//! - **Early stopping** with periodic validation checks
//! - **Quantile ensembles** that train multiple models and enforce monotonicity
//! - **JSON serialization** for model persistence and portability
//!
//! ## Quick start
//!
//! ```rust
//! use gbt_quantile::{GBTConfig, QuantileEnsemble};
//!
//! // Generate some data
//! let x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
//! let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 1.0).collect();
//!
//! // Train a quantile ensemble
//! let config = GBTConfig { n_trees: 50, ..GBTConfig::default() };
//! let quantiles = vec![0.1, 0.5, 0.9];
//! let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, None);
//!
//! // Predict
//! let pred = ensemble.predict(&[50.0]);
//! println!("P10={}, P50={}, P90={}",
//!     pred.p(0.1).unwrap(),
//!     pred.median(),
//!     pred.p(0.9).unwrap(),
//! );
//! ```
//!
//! ## Single model
//!
//! ```rust
//! use gbt_quantile::{GBTConfig, trainer};
//!
//! let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
//! let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let config = GBTConfig { n_trees: 50, min_samples_leaf: 1, ..GBTConfig::default() };
//! let model = trainer::train(&x, &y, &config, None);
//! let prediction = model.predict(&[3.0]);
//! ```

pub mod config;
pub mod ensemble;
pub mod metrics;
pub mod split;
pub mod trainer;
pub mod tree;

pub use config::GBTConfig;
pub use ensemble::{QuantileEnsemble, QuantilePrediction};
pub use metrics::Metrics;
pub use tree::GradientBoostedTree;
