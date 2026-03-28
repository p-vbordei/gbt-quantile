/// Configuration for gradient-boosted tree training.
#[derive(Debug, Clone)]
pub struct GBTConfig {
    /// Number of boosting rounds (trees to train).
    pub n_trees: usize,
    /// Maximum depth of each tree.
    pub max_depth: usize,
    /// Learning rate (shrinkage factor per tree).
    pub learning_rate: f64,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Target quantile for pinball loss. `None` = squared error (L2) loss.
    /// `Some(0.5)` = median regression, `Some(0.1)` = P10, `Some(0.9)` = P90.
    pub quantile: Option<f64>,
    /// Stop training if validation loss hasn't improved for this many checks.
    /// Checks happen every 5 boosting rounds. `None` = no early stopping.
    pub early_stopping_rounds: Option<usize>,
    /// Number of candidate split thresholds per feature (percentile-based).
    pub n_bins: usize,
}

impl Default for GBTConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: 4,
            learning_rate: 0.05,
            min_samples_leaf: 10,
            quantile: None,
            early_stopping_rounds: Some(5),
            n_bins: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GBTConfig::default();
        assert_eq!(config.n_trees, 100);
        assert_eq!(config.max_depth, 4);
        assert!((config.learning_rate - 0.05).abs() < 1e-10);
        assert_eq!(config.min_samples_leaf, 10);
        assert!(config.quantile.is_none());
        assert_eq!(config.early_stopping_rounds, Some(5));
        assert_eq!(config.n_bins, 10);
    }
}
