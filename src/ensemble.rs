//! Quantile ensemble: train multiple GBT models at different quantiles
//! and predict with monotonicity enforcement.

use crate::config::GBTConfig;
use crate::trainer;
use crate::tree::GradientBoostedTree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A prediction from a quantile ensemble, containing one value per quantile.
///
/// Values are sorted by quantile and monotonicity is enforced:
/// lower quantiles always predict values less than or equal to higher quantiles.
#[derive(Debug, Clone)]
pub struct QuantilePrediction {
    /// (quantile, prediction) pairs, sorted by quantile.
    values: Vec<(f64, f64)>,
}

impl QuantilePrediction {
    /// Get the prediction for a specific quantile.
    ///
    /// Returns `None` if the quantile was not part of the ensemble.
    pub fn p(&self, quantile: f64) -> Option<f64> {
        self.values
            .iter()
            .find(|(q, _)| (*q - quantile).abs() < 1e-6)
            .map(|(_, v)| *v)
    }

    /// Get all (quantile, prediction) pairs.
    pub fn values(&self) -> &[(f64, f64)] {
        &self.values
    }

    /// Get the median prediction (P50).
    ///
    /// Falls back to the middle quantile if P50 is not present.
    pub fn median(&self) -> f64 {
        self.p(0.5).unwrap_or_else(|| {
            if self.values.is_empty() {
                0.0
            } else {
                self.values[self.values.len() / 2].1
            }
        })
    }

    /// Get the lowest quantile prediction (e.g. P10).
    pub fn lower(&self) -> f64 {
        self.values.first().map_or(0.0, |(_, v)| *v)
    }

    /// Get the highest quantile prediction (e.g. P90).
    pub fn upper(&self) -> f64 {
        self.values.last().map_or(0.0, |(_, v)| *v)
    }
}

/// An ensemble of GBT models, one per quantile.
///
/// Trains separate models for each quantile (e.g. P10, P25, P50, P75, P90)
/// and enforces monotonicity at prediction time: lower quantile predictions
/// are always less than or equal to higher quantile predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileEnsemble {
    /// (quantile, model) pairs, sorted by quantile.
    models: Vec<(f64, GradientBoostedTree)>,
}

impl QuantileEnsemble {
    /// Train a quantile ensemble on the provided data.
    ///
    /// Trains one GBT model per quantile in `quantiles`. Each model uses
    /// `config` as the base configuration, with the quantile field overridden.
    pub fn train(
        x: &[Vec<f64>],
        y: &[f64],
        quantiles: &[f64],
        config: &GBTConfig,
        feature_names: Option<&[String]>,
    ) -> Self {
        Self::train_with_validation(x, y, &[], &[], quantiles, config, feature_names)
    }

    /// Train with validation data for early stopping.
    pub fn train_with_validation(
        x: &[Vec<f64>],
        y: &[f64],
        x_val: &[Vec<f64>],
        y_val: &[f64],
        quantiles: &[f64],
        config: &GBTConfig,
        feature_names: Option<&[String]>,
    ) -> Self {
        let mut models: Vec<(f64, GradientBoostedTree)> = quantiles
            .par_iter()
            .map(|&q| {
                let q_config = GBTConfig {
                    quantile: Some(q),
                    ..config.clone()
                };
                let model =
                    trainer::train_with_validation(x, y, x_val, y_val, &q_config, feature_names);
                (q, model)
            })
            .collect();

        // Sort by quantile
        models.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Self { models }
    }

    /// Predict all quantiles for a single sample, with monotonicity enforcement.
    pub fn predict(&self, features: &[f64]) -> QuantilePrediction {
        let raw: Vec<(f64, f64)> = self
            .models
            .iter()
            .map(|(q, model)| (*q, model.predict(features)))
            .collect();

        QuantilePrediction {
            values: enforce_monotonicity(&raw),
        }
    }

    /// Predict all quantiles for a batch of samples.
    pub fn predict_batch(&self, matrix: &[Vec<f64>]) -> Vec<QuantilePrediction> {
        matrix.iter().map(|row| self.predict(row)).collect()
    }

    /// Serialize the ensemble to a JSON string.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize an ensemble from a JSON string.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Access the underlying (quantile, model) pairs.
    pub fn models(&self) -> &[(f64, GradientBoostedTree)] {
        &self.models
    }
}

/// Enforce monotonicity: ensure p_lower <= p_higher by sorting predictions.
///
/// True quantiles form a monotonically increasing CDF. If models cross and predict
/// out of order, sorting the predictions is a robust way to enforce monotonicity
/// without allowing a single low-quantile outlier to clamp all subsequent higher quantiles.
fn enforce_monotonicity(raw: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if raw.is_empty() {
        return vec![];
    }

    let mut values: Vec<f64> = raw.iter().map(|(_, v)| *v).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    raw.iter()
        .zip(values)
        .map(|(&(q, _), enforced_v)| (q, enforced_v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GBTConfig;

    fn sample_data(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let xi = i as f64 / n as f64 * 10.0;
            x.push(vec![xi]);
            // Add deterministic spread
            let noise = ((i * 13 + 7) % 19) as f64 / 10.0 - 0.95;
            y.push(3.0 * xi + 2.0 + noise);
        }
        (x, y)
    }

    #[test]
    fn test_ensemble_quantile_ordering() {
        let (x, y) = sample_data(300);
        let config = GBTConfig {
            n_trees: 100,
            max_depth: 4,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile: None, // overridden per quantile
            early_stopping_rounds: None,
            n_bins: 255,
        };

        let quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];
        let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, None);

        // Check ordering at multiple points
        for &xi in &[1.0, 5.0, 9.0] {
            let pred = ensemble.predict(&[xi]);
            let values = pred.values();

            for pair in values.windows(2) {
                assert!(
                    pair[0].1 <= pair[1].1 + 1e-10,
                    "At x={xi}: P{} ({}) should be <= P{} ({})",
                    (pair[0].0 * 100.0) as u32,
                    pair[0].1,
                    (pair[1].0 * 100.0) as u32,
                    pair[1].1,
                );
            }

            // lower <= median <= upper
            assert!(pred.lower() <= pred.median() + 1e-10);
            assert!(pred.median() <= pred.upper() + 1e-10);
        }
    }

    #[test]
    fn test_ensemble_serialization_roundtrip() {
        let (x, y) = sample_data(100);
        let config = GBTConfig {
            n_trees: 20,
            max_depth: 3,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile: None,
            early_stopping_rounds: None,
            n_bins: 255,
        };

        let quantiles = vec![0.1, 0.5, 0.9];
        let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, None);

        let json = ensemble.to_json().unwrap();
        let restored = QuantileEnsemble::from_json(&json).unwrap();

        assert_eq!(ensemble.models().len(), restored.models().len());

        // Predictions should be identical after roundtrip
        let test_features = vec![3.0];
        let orig_pred = ensemble.predict(&test_features);
        let rest_pred = restored.predict(&test_features);

        for (o, r) in orig_pred.values().iter().zip(rest_pred.values().iter()) {
            assert!(
                (o.0 - r.0).abs() < 1e-10,
                "Quantile mismatch: {} vs {}",
                o.0,
                r.0
            );
            assert!(
                (o.1 - r.1).abs() < 1e-10,
                "Prediction mismatch at q={}: {} vs {}",
                o.0,
                o.1,
                r.1
            );
        }
    }

    #[test]
    fn test_ensemble_p_lookup() {
        let (x, y) = sample_data(100);
        let config = GBTConfig {
            n_trees: 10,
            max_depth: 3,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile: None,
            early_stopping_rounds: None,
            n_bins: 255,
        };

        let quantiles = vec![0.1, 0.5, 0.9];
        let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, None);
        let pred = ensemble.predict(&[5.0]);

        assert!(pred.p(0.1).is_some());
        assert!(pred.p(0.5).is_some());
        assert!(pred.p(0.9).is_some());
        assert!(pred.p(0.42).is_none(), "Quantile 0.42 was not trained");
    }

    #[test]
    fn test_enforce_monotonicity() {
        // Simulate out-of-order raw predictions
        let raw = vec![(0.1, 5.0), (0.5, 3.0), (0.9, 7.0)];
        let enforced = enforce_monotonicity(&raw);
        // By sorting: [3.0, 5.0, 7.0]
        assert_eq!(enforced[0].1, 3.0);
        assert_eq!(enforced[1].1, 5.0);
        assert_eq!(enforced[2].1, 7.0);
    }

    #[test]
    fn test_predict_batch() {
        let (x, y) = sample_data(100);
        let config = GBTConfig {
            n_trees: 10,
            max_depth: 3,
            learning_rate: 0.1,
            min_samples_leaf: 5,
            quantile: None,
            early_stopping_rounds: None,
            n_bins: 255,
        };

        let quantiles = vec![0.1, 0.5, 0.9];
        let ensemble = QuantileEnsemble::train(&x, &y, &quantiles, &config, None);

        let batch = vec![vec![1.0], vec![5.0], vec![9.0]];
        let preds = ensemble.predict_batch(&batch);
        assert_eq!(preds.len(), 3);

        // Each prediction should have 3 quantiles
        for pred in &preds {
            assert_eq!(pred.values().len(), 3);
        }
    }
}
