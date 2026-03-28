//! Tree model data structures: nodes, leaves, and the full GBT model.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single decision tree node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Index of the feature used for splitting.
    pub feature_index: usize,
    /// Threshold value: samples with `feature[feature_index] <= threshold` go left.
    pub threshold: f64,
    /// Left child (feature <= threshold).
    pub left: NodeRef,
    /// Right child (feature > threshold).
    pub right: NodeRef,
}

/// A reference to a child node: either a leaf value or a subtree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeRef {
    /// Terminal leaf with a prediction value.
    Leaf(f64),
    /// Internal node with further splits.
    Node(Box<TreeNode>),
}

/// A gradient-boosted tree ensemble model.
///
/// Predictions are computed as: `base_score + learning_rate * sum(tree_predictions)`.
/// Models are JSON-serializable with schema versioning for forward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostedTree {
    /// Schema version for forward/backward compatibility.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// The ensemble of decision trees.
    pub trees: Vec<TreeNode>,
    /// Initial prediction (bias / intercept).
    pub base_score: f64,
    /// Learning rate (shrinkage factor applied to each tree).
    pub learning_rate: f64,
    /// Feature names in order matching feature_index.
    pub feature_names: Vec<String>,
    /// Target quantile: `None` for L2/mean, `Some(q)` for quantile regression.
    pub quantile: Option<f64>,
    /// Output scale factor applied to the final prediction.
    #[serde(default = "default_output_scale")]
    pub output_scale: f64,
    /// Optional metadata about the model.
    pub metadata: Option<serde_json::Value>,
}

fn default_schema_version() -> u32 { 1 }
fn default_output_scale() -> f64 { 1.0 }

impl GradientBoostedTree {
    /// Predict a single sample.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let raw = self.trees.iter().fold(self.base_score, |acc, tree| {
            acc + self.learning_rate * traverse_node(tree, features)
        });
        raw * self.output_scale
    }

    /// Predict a batch of samples.
    pub fn predict_batch(&self, feature_matrix: &[Vec<f64>]) -> Vec<f64> {
        feature_matrix.iter().map(|row| self.predict(row)).collect()
    }

    /// Number of trees in the ensemble.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of features the model expects.
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Compute feature importance based on split frequency.
    /// Returns a map from feature index to normalized importance [0, 1].
    pub fn feature_importance(&self) -> HashMap<usize, f64> {
        let n = self.feature_names.len();
        let mut counts = vec![0.0_f64; n];
        for tree in &self.trees {
            count_splits(tree, &mut counts);
        }
        let total: f64 = counts.iter().sum();
        let mut importance = HashMap::new();
        for (i, &count) in counts.iter().enumerate() {
            let imp = if total > 0.0 { count / total } else { 1.0 / n as f64 };
            importance.insert(i, imp);
        }
        importance
    }

    /// Compute named feature importance.
    pub fn feature_importance_named(&self) -> HashMap<String, f64> {
        let indexed = self.feature_importance();
        indexed
            .into_iter()
            .map(|(i, v)| {
                let name = self.feature_names.get(i).cloned().unwrap_or_else(|| format!("f{i}"));
                (name, v)
            })
            .collect()
    }

    /// Serialize the model to JSON string.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a model from JSON string.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Serialize the model to bytes (UTF-8 JSON).
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize a model from bytes (UTF-8 JSON).
    pub fn from_bytes(data: &[u8]) -> anyhow::Result<Self> {
        Ok(serde_json::from_slice(data)?)
    }
}

/// Traverse a tree node to get the leaf prediction for a feature vector.
pub(crate) fn traverse_node(node: &TreeNode, features: &[f64]) -> f64 {
    let val = features.get(node.feature_index).copied().unwrap_or(0.0);
    let child = if val <= node.threshold { &node.left } else { &node.right };
    match child {
        NodeRef::Leaf(v) => *v,
        NodeRef::Node(next) => traverse_node(next, features),
    }
}

fn count_splits(node: &TreeNode, counts: &mut [f64]) {
    if node.feature_index < counts.len() {
        counts[node.feature_index] += 1.0;
    }
    if let NodeRef::Node(ref child) = node.left {
        count_splits(child, counts);
    }
    if let NodeRef::Node(ref child) = node.right {
        count_splits(child, counts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_model() -> GradientBoostedTree {
        let tree = TreeNode {
            feature_index: 0,
            threshold: 5.0,
            left: NodeRef::Leaf(1.0),
            right: NodeRef::Leaf(10.0),
        };
        GradientBoostedTree {
            schema_version: 1,
            trees: vec![tree],
            base_score: 0.0,
            learning_rate: 1.0,
            feature_names: vec!["x".to_string()],
            quantile: None,
            output_scale: 1.0,
            metadata: None,
        }
    }

    #[test]
    fn test_predict_left_right() {
        let model = simple_model();
        assert_eq!(model.predict(&[3.0]), 1.0);  // 3 <= 5 → left
        assert_eq!(model.predict(&[7.0]), 10.0); // 7 > 5 → right
        assert_eq!(model.predict(&[5.0]), 1.0);  // 5 <= 5 → left
    }

    #[test]
    fn test_predict_batch() {
        let model = simple_model();
        let preds = model.predict_batch(&[vec![3.0], vec![7.0]]);
        assert_eq!(preds, vec![1.0, 10.0]);
    }

    #[test]
    fn test_json_roundtrip() {
        let model = simple_model();
        let json = model.to_json().unwrap();
        let restored = GradientBoostedTree::from_json(&json).unwrap();
        assert_eq!(restored.predict(&[3.0]), model.predict(&[3.0]));
        assert_eq!(restored.n_trees(), 1);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let model = simple_model();
        let bytes = model.to_bytes().unwrap();
        let restored = GradientBoostedTree::from_bytes(&bytes).unwrap();
        assert_eq!(restored.predict(&[7.0]), model.predict(&[7.0]));
    }

    #[test]
    fn test_feature_importance() {
        let model = simple_model();
        let imp = model.feature_importance();
        assert_eq!(imp[&0], 1.0); // only feature 0 is used
    }
}
