//! Adversarial performance and memory tracking benchmark.
//!
//! Validates:
//! 1. Training time on 105 days history (10,080 samples x 96 quarters/day) is well below 200 ms.
//! 2. Peak heap memory allocation during training is well below 15 MB.
//! 3. Monotonicity holds on full 105-day load profile.

use gbt_quantile::config::GBTConfig;
use gbt_quantile::ensemble::QuantileEnsemble;
use gbt_quantile::trainer::train_with_validation;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct TrackingAllocator;

static CURRENT_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let current = CURRENT_ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK_ALLOCATED.fetch_max(current, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT_ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            if new_size > layout.size() {
                let diff = new_size - layout.size();
                let current = CURRENT_ALLOCATED.fetch_add(diff, Ordering::SeqCst) + diff;
                PEAK_ALLOCATED.fetch_max(current, Ordering::SeqCst);
            } else {
                let diff = layout.size() - new_size;
                CURRENT_ALLOCATED.fetch_sub(diff, Ordering::SeqCst);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn generate_105_days_telemetry() -> (Vec<Vec<f64>>, Vec<f64>, Vec<String>) {
    let days = 105;
    let n = days * 96; // 10,080 quarters
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    let feature_names = vec![
        "hour".to_string(),
        "dow".to_string(),
        "is_weekend".to_string(),
        "is_holiday".to_string(),
        "lag_672".to_string(),
        "avg4".to_string(),
        "temp_c".to_string(),
    ];

    for i in 0..n {
        let quarter_of_day = (i % 96) as f64;
        let hour = (quarter_of_day / 4.0).floor();
        let day_idx = i / 96;
        let dow = (day_idx % 7) as f64;
        let is_weekend = if dow >= 5.0 { 1.0 } else { 0.0 };
        let is_holiday = if day_idx % 30 == 0 { 1.0 } else { 0.0 };

        // Synthetic daily & weekly load shape
        let diurnal = 50.0 + 35.0 * ((quarter_of_day / 96.0 * std::f64::consts::TAU) - 1.5).sin();
        let weekly = if is_weekend == 1.0 { 0.7 } else { 1.0 };
        let temp_c = 15.0 + 10.0 * (((day_idx as f64 / 30.0).sin() * 0.8) + ((quarter_of_day / 96.0 * std::f64::consts::TAU) - 2.0).sin() * 0.5);
        let thermal_response = (20.0 - temp_c).max(0.0) * 1.5 + (temp_c - 24.0).max(0.0) * 2.0;

        let lag_672 = diurnal * weekly * 0.95 + thermal_response;
        let avg4 = diurnal * weekly + thermal_response * 0.9;

        // Pseudo-random noise
        let noise = (((i * 2654435761) ^ (i >> 3)) % 1000) as f64 / 200.0 - 2.5;
        let load = (diurnal * weekly + thermal_response + noise).max(1.0);

        x.push(vec![
            hour,
            dow,
            is_weekend,
            is_holiday,
            lag_672,
            avg4,
            temp_c,
        ]);
        y.push(load.ln_1p());
    }

    (x, y, feature_names)
}

#[test]
fn test_105_days_training_latency_and_memory() {
    let (x, y, feature_names) = generate_105_days_telemetry();
    let n = x.len();
    assert_eq!(n, 10_080, "105 days must have exactly 10,080 intervals");

    // Hold out 1 week (672 rows) for calibration/validation, exactly matching consumption-core
    let cal_rows = 672;
    let train_n = n - cal_rows;
    let (x_train, y_train) = (&x[..train_n], &y[..train_n]);
    let (x_val, y_val) = (&x[train_n..], &y[train_n..]);

    let config = GBTConfig {
        n_trees: 80,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: Some(0.50),
        early_stopping_rounds: Some(10),
        n_bins: 255,
    };

    // Warm-up run to prime any thread pool or one-time runtime caches
    let _warmup = train_with_validation(x_train, y_train, x_val, y_val, &config, Some(&feature_names));

    // Reset peak tracker to current baseline allocation
    let baseline_bytes = CURRENT_ALLOCATED.load(Ordering::SeqCst);
    PEAK_ALLOCATED.store(baseline_bytes, Ordering::SeqCst);

    let start = Instant::now();
    let model = train_with_validation(x_train, y_train, x_val, y_val, &config, Some(&feature_names));
    let elapsed = start.elapsed();

    let peak_during_training = PEAK_ALLOCATED.load(Ordering::SeqCst);
    let delta_peak_bytes = peak_during_training.saturating_sub(baseline_bytes);
    let delta_peak_mb = delta_peak_bytes as f64 / (1024.0 * 1024.0);
    let total_peak_mb = peak_during_training as f64 / (1024.0 * 1024.0);

    println!("\n=======================================================");
    println!("105-DAY (10,080 INTERVALS) TRAINING BENCHMARK RESULTS:");
    println!("  Trees trained: {}", model.n_trees());
    println!("  Training latency: {:.2?}", elapsed);
    println!("  Latency (ms): {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Delta peak heap allocated: {:.2} MB ({} bytes)", delta_peak_mb, delta_peak_bytes);
    println!("  Total peak heap allocated: {:.2} MB ({} bytes)", total_peak_mb, peak_during_training);
    println!("=======================================================\n");

    let latency_ms = elapsed.as_secs_f64() * 1000.0;
    #[cfg(not(debug_assertions))]
    assert!(
        latency_ms < 200.0,
        "Training time in release mode must be well under 200 ms, got {:.2} ms",
        latency_ms
    );
    #[cfg(debug_assertions)]
    assert!(
        latency_ms < 1000.0,
        "Training time in debug mode must be under 1000 ms, got {:.2} ms",
        latency_ms
    );
    assert!(
        total_peak_mb < 15.0,
        "Total peak heap memory must be strictly under 15 MB, got {:.2} MB",
        total_peak_mb
    );
}

#[test]
fn test_105_days_multi_quantile_ensemble_latency_and_monotonicity() {
    let (x, y, feature_names) = generate_105_days_telemetry();
    let n = x.len();

    let config = GBTConfig {
        n_trees: 80,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: None,
        early_stopping_rounds: Some(10),
        n_bins: 255,
    };

    let cal_rows = 672;
    let train_n = n - cal_rows;
    let (x_train, y_train) = (&x[..train_n], &y[..train_n]);
    let (x_val, y_val) = (&x[train_n..], &y[train_n..]);

    let quantiles = vec![0.10, 0.50, 0.90];

    let baseline_bytes = CURRENT_ALLOCATED.load(Ordering::SeqCst);
    PEAK_ALLOCATED.store(baseline_bytes, Ordering::SeqCst);

    let start = Instant::now();
    let ensemble = QuantileEnsemble::train_with_validation(
        x_train,
        y_train,
        x_val,
        y_val,
        &quantiles,
        &config,
        Some(&feature_names),
    );
    let ensemble_elapsed = start.elapsed();

    let peak_during_ensemble = PEAK_ALLOCATED.load(Ordering::SeqCst);
    let ensemble_peak_mb = peak_during_ensemble as f64 / (1024.0 * 1024.0);
    let ensemble_latency_ms = ensemble_elapsed.as_secs_f64() * 1000.0;

    println!("\n=======================================================");
    println!("105-DAY 3-QUANTILE ENSEMBLE [P10, P50, P90] RESULTS:");
    println!("  Total ensemble training latency: {:.2} ms", ensemble_latency_ms);
    println!("  Ensemble peak heap memory: {:.2} MB", ensemble_peak_mb);
    println!("=======================================================\n");

    // Test monotonicity on 500 test points
    let mut violations = 0;
    for sample in x.iter().take(500) {
        let pred = ensemble.predict(sample);
        let vals = pred.values();
        for w in vals.windows(2) {
            if w[0].1 > w[1].1 + 1e-12 {
                violations += 1;
            }
        }
        assert!(pred.lower() <= pred.median() + 1e-12);
        assert!(pred.median() <= pred.upper() + 1e-12);
    }

    assert_eq!(violations, 0, "Ensemble must have zero monotonicity violations");
    assert!(
        ensemble_peak_mb < 15.0,
        "Ensemble peak memory must be under 15 MB, got {:.2} MB",
        ensemble_peak_mb
    );
}

#[test]
fn test_worst_case_no_early_stopping_latency() {
    // Adversarial worst case: 80 full trees trained with no early stopping
    let (x, y, feature_names) = generate_105_days_telemetry();
    let _n = x.len();

    let config = GBTConfig {
        n_trees: 80,
        max_depth: 4,
        learning_rate: 0.1,
        min_samples_leaf: 5,
        quantile: Some(0.50),
        early_stopping_rounds: None, // force full 80 trees
        n_bins: 255,
    };

    let start = Instant::now();
    let model = train_with_validation(&x, &y, &[], &[], &config, Some(&feature_names));
    let elapsed = start.elapsed();
    let latency_ms = elapsed.as_secs_f64() * 1000.0;

    println!("Full 80 trees (no early stopping) latency: {:.2} ms (trees={})", latency_ms, model.n_trees());
    assert_eq!(model.n_trees(), 80);
    #[cfg(not(debug_assertions))]
    assert!(
        latency_ms < 200.0,
        "Worst-case full 80 trees in release mode must be under 200 ms, got {:.2} ms",
        latency_ms
    );
    #[cfg(debug_assertions)]
    assert!(
        latency_ms < 1000.0,
        "Worst-case full 80 trees in debug mode must be under 1000 ms, got {:.2} ms",
        latency_ms
    );
}
