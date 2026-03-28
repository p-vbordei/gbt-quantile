//! Data splitting utilities for train/test partitioning.

/// Split data into train and test sets.
///
/// By default, performs a temporal split: the first `train_fraction` of rows
/// become the training set, the rest become the test set. This preserves
/// time ordering, which is important for time-series data.
///
/// If `seed` is provided, the data is shuffled first using a simple LCG
/// (linear congruential generator), producing a random split instead.
///
/// # Example
///
/// ```
/// use gbt_quantile::split::train_test_split;
///
/// let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
/// let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
/// let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.8, None);
/// assert_eq!(x_train.len(), 4);
/// assert_eq!(x_test.len(), 1);
/// ```
pub fn train_test_split(
    x: &[Vec<f64>],
    y: &[f64],
    train_fraction: f64,
    seed: Option<u64>,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len().min(y.len());
    if n == 0 {
        return (vec![], vec![], vec![], vec![]);
    }

    let fraction = train_fraction.clamp(0.0, 1.0);
    let train_size = ((n as f64) * fraction).round() as usize;
    let train_size = train_size.clamp(0, n);

    // Build index order: sequential or shuffled
    let indices: Vec<usize> = if let Some(seed) = seed {
        let mut idx: Vec<usize> = (0..n).collect();
        lcg_shuffle(&mut idx, seed);
        idx
    } else {
        (0..n).collect()
    };

    let mut x_train = Vec::with_capacity(train_size);
    let mut y_train = Vec::with_capacity(train_size);
    let mut x_test = Vec::with_capacity(n - train_size);
    let mut y_test = Vec::with_capacity(n - train_size);

    for (pos, &i) in indices.iter().enumerate() {
        if pos < train_size {
            x_train.push(x[i].clone());
            y_train.push(y[i]);
        } else {
            x_test.push(x[i].clone());
            y_test.push(y[i]);
        }
    }

    (x_train, y_train, x_test, y_test)
}

/// Fisher-Yates shuffle using a linear congruential generator for determinism.
fn lcg_shuffle(indices: &mut [usize], seed: u64) {
    let n = indices.len();
    if n <= 1 {
        return;
    }

    // LCG parameters (same as glibc)
    let mut state = seed;
    for i in (1..n).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sizes() {
        let x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.8, None);
        assert_eq!(x_train.len(), 80);
        assert_eq!(y_train.len(), 80);
        assert_eq!(x_test.len(), 20);
        assert_eq!(y_test.len(), 20);
    }

    #[test]
    fn test_temporal_split_preserves_order() {
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..10).map(|i| i as f64 * 10.0).collect();

        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.5, None);

        // First half should be train
        assert_eq!(x_train.len(), 5);
        for i in 0..5 {
            assert_eq!(x_train[i][0], i as f64);
            assert_eq!(y_train[i], i as f64 * 10.0);
        }

        // Second half should be test
        assert_eq!(x_test.len(), 5);
        for i in 0..5 {
            assert_eq!(x_test[i][0], (i + 5) as f64);
            assert_eq!(y_test[i], (i + 5) as f64 * 10.0);
        }
    }

    #[test]
    fn test_shuffled_split_is_deterministic() {
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..50).map(|i| i as f64).collect();

        let (x1, y1, _, _) = train_test_split(&x, &y, 0.8, Some(42));
        let (x2, y2, _, _) = train_test_split(&x, &y, 0.8, Some(42));

        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_shuffled_split_differs_from_temporal() {
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..50).map(|i| i as f64).collect();

        let (x_temporal, _, _, _) = train_test_split(&x, &y, 0.8, None);
        let (x_shuffled, _, _, _) = train_test_split(&x, &y, 0.8, Some(42));

        // With a shuffle, the training set should differ from a temporal split
        assert_ne!(x_temporal, x_shuffled, "Shuffled split should differ from temporal");
    }

    #[test]
    fn test_empty_data() {
        let (x_train, y_train, x_test, y_test) =
            train_test_split(&[], &[], 0.8, None);
        assert!(x_train.is_empty());
        assert!(y_train.is_empty());
        assert!(x_test.is_empty());
        assert!(y_test.is_empty());
    }

    #[test]
    fn test_all_train() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![10.0, 20.0];
        let (x_train, _, x_test, _) = train_test_split(&x, &y, 1.0, None);
        assert_eq!(x_train.len(), 2);
        assert_eq!(x_test.len(), 0);
    }

    #[test]
    fn test_all_test() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![10.0, 20.0];
        let (x_train, _, x_test, _) = train_test_split(&x, &y, 0.0, None);
        assert_eq!(x_train.len(), 0);
        assert_eq!(x_test.len(), 2);
    }

    #[test]
    fn test_no_data_loss() {
        let x: Vec<Vec<f64>> = (0..37).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..37).map(|i| i as f64).collect();

        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.7, Some(99));
        assert_eq!(
            x_train.len() + x_test.len(),
            37,
            "No rows should be lost"
        );
        assert_eq!(y_train.len() + y_test.len(), 37);
    }
}
