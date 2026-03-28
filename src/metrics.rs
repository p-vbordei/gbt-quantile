//! Evaluation metrics for regression and quantile models.
//!
//! Provides standard regression metrics (MAE, RMSE, R², MAPE) and
//! pinball loss for quantile regression evaluation.

/// Summary of regression evaluation metrics.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Mean Absolute Error: mean(|y_true - y_pred|).
    pub mae: f64,
    /// Root Mean Squared Error: sqrt(mean((y_true - y_pred)²)).
    pub rmse: f64,
    /// Coefficient of determination (R²): 1 - SS_res / SS_tot.
    pub r2: f64,
    /// Mean Absolute Percentage Error: mean(|error / actual|),
    /// skipping actuals with absolute value < 0.01 to avoid division by near-zero.
    pub mape: f64,
    /// Number of samples used in the evaluation.
    pub n_samples: usize,
}

/// Compute regression metrics from true and predicted values.
///
/// Returns an error if the arrays have different lengths or are empty.
///
/// # Example
///
/// ```
/// use gbt_quantile::metrics;
///
/// let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y_pred = vec![1.1, 2.0, 2.9, 4.1, 5.0];
/// let m = metrics::evaluate(&y_true, &y_pred).unwrap();
/// assert!(m.mae < 0.1);
/// assert!(m.r2 > 0.99);
/// ```
pub fn evaluate(y_true: &[f64], y_pred: &[f64]) -> anyhow::Result<Metrics> {
    if y_true.len() != y_pred.len() {
        anyhow::bail!(
            "Length mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        );
    }
    if y_true.is_empty() {
        anyhow::bail!("Cannot evaluate empty arrays");
    }

    let n = y_true.len() as f64;

    // MAE
    let mae: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / n;

    // RMSE
    let mse: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / n;
    let rmse = mse.sqrt();

    // R²
    let y_mean = y_true.iter().sum::<f64>() / n;
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - y_mean).powi(2)).sum();
    let r2 = if ss_tot.abs() < 1e-12 {
        // All true values are identical; R² is undefined, return 0.0
        0.0
    } else {
        1.0 - ss_res / ss_tot
    };

    // MAPE — skip samples where |actual| < 0.01
    let mape_pairs: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, _)| t.abs() >= 0.01)
        .map(|(t, p)| ((t - p) / t).abs())
        .collect();
    let mape = if mape_pairs.is_empty() {
        0.0
    } else {
        mape_pairs.iter().sum::<f64>() / mape_pairs.len() as f64
    };

    Ok(Metrics {
        mae,
        rmse,
        r2,
        mape,
        n_samples: y_true.len(),
    })
}

/// Compute pinball (quantile) loss.
///
/// For a quantile `q`:
/// - If `error >= 0` (under-prediction): loss = `q * error`
/// - If `error < 0` (over-prediction): loss = `(q - 1) * error`
///
/// Returns the mean pinball loss across all samples.
///
/// # Example
///
/// ```
/// use gbt_quantile::metrics;
///
/// let y_true = vec![1.0, 2.0, 3.0];
/// let y_pred = vec![1.5, 1.5, 2.5];
/// let loss = metrics::pinball_loss(&y_true, &y_pred, 0.5).unwrap();
/// assert!(loss > 0.0);
/// ```
pub fn pinball_loss(y_true: &[f64], y_pred: &[f64], quantile: f64) -> anyhow::Result<f64> {
    if y_true.len() != y_pred.len() {
        anyhow::bail!(
            "Length mismatch: y_true has {} elements, y_pred has {}",
            y_true.len(),
            y_pred.len()
        );
    }
    if y_true.is_empty() {
        anyhow::bail!("Cannot compute pinball loss on empty arrays");
    }

    let n = y_true.len() as f64;
    let total: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let error = t - p;
            if error >= 0.0 {
                quantile * error
            } else {
                (quantile - 1.0) * error
            }
        })
        .sum();

    Ok(total / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_predictions() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = evaluate(&y, &y).unwrap();
        assert!((m.mae).abs() < 1e-10);
        assert!((m.rmse).abs() < 1e-10);
        assert!((m.r2 - 1.0).abs() < 1e-10);
        assert!((m.mape).abs() < 1e-10);
        assert_eq!(m.n_samples, 5);
    }

    #[test]
    fn test_known_mae_rmse() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 6.0]; // error = [0, 0, 3]

        let m = evaluate(&y_true, &y_pred).unwrap();
        // MAE = (0 + 0 + 3) / 3 = 1.0
        assert!((m.mae - 1.0).abs() < 1e-10);
        // RMSE = sqrt((0 + 0 + 9) / 3) = sqrt(3)
        assert!((m.rmse - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_r2_known() {
        // y_true = [1, 2, 3], mean = 2
        // y_pred = [1.5, 2.5, 2.5]
        // ss_res = 0.25 + 0.25 + 0.25 = 0.75
        // ss_tot = 1 + 0 + 1 = 2
        // R² = 1 - 0.75/2 = 0.625
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.5, 2.5, 2.5];
        let m = evaluate(&y_true, &y_pred).unwrap();
        assert!((m.r2 - 0.625).abs() < 1e-10, "R² = {}, expected 0.625", m.r2);
    }

    #[test]
    fn test_mape_skips_near_zero() {
        // Second value is near zero, should be skipped
        let y_true = vec![10.0, 0.001, 20.0];
        let y_pred = vec![11.0, 5.0, 22.0];
        let m = evaluate(&y_true, &y_pred).unwrap();
        // MAPE over [10, 20] only: (1/10 + 2/20) / 2 = (0.1 + 0.1) / 2 = 0.1
        assert!((m.mape - 0.1).abs() < 1e-10, "MAPE = {}, expected 0.1", m.mape);
    }

    #[test]
    fn test_length_mismatch() {
        let result = evaluate(&[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_arrays() {
        let result = evaluate(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pinball_loss_median() {
        // At q=0.5, pinball loss = 0.5 * MAE
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.5, 1.5, 2.5];
        let loss = pinball_loss(&y_true, &y_pred, 0.5).unwrap();

        let m = evaluate(&y_true, &y_pred).unwrap();
        assert!(
            (loss - m.mae * 0.5).abs() < 1e-10,
            "Pinball(0.5) = {loss}, MAE/2 = {}",
            m.mae * 0.5
        );
    }

    #[test]
    fn test_pinball_loss_asymmetry() {
        // Under-prediction: error = 1.0 > 0
        // At q=0.9: loss = 0.9 * 1.0 = 0.9
        // At q=0.1: loss = 0.1 * 1.0 = 0.1
        let y_true = vec![2.0];
        let y_pred = vec![1.0];

        let loss_90 = pinball_loss(&y_true, &y_pred, 0.9).unwrap();
        let loss_10 = pinball_loss(&y_true, &y_pred, 0.1).unwrap();

        assert!((loss_90 - 0.9).abs() < 1e-10);
        assert!((loss_10 - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_pinball_loss_over_prediction() {
        // Over-prediction: error = -1.0 < 0
        // At q=0.9: loss = (0.9 - 1.0) * (-1.0) = 0.1
        // At q=0.1: loss = (0.1 - 1.0) * (-1.0) = 0.9
        let y_true = vec![1.0];
        let y_pred = vec![2.0];

        let loss_90 = pinball_loss(&y_true, &y_pred, 0.9).unwrap();
        let loss_10 = pinball_loss(&y_true, &y_pred, 0.1).unwrap();

        assert!((loss_90 - 0.1).abs() < 1e-10);
        assert!((loss_10 - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_r2_constant_target() {
        let y_true = vec![5.0, 5.0, 5.0];
        let y_pred = vec![5.0, 5.0, 5.0];
        let m = evaluate(&y_true, &y_pred).unwrap();
        // ss_tot = 0, R² defined as 0.0 for constant target
        assert!((m.r2).abs() < 1e-10);
    }
}
