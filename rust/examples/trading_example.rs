//! Trading Example: Hierarchical VAE with Bybit BTCUSDT Data
//!
//! This example:
//! 1. Fetches BTCUSDT kline data from Bybit
//! 2. Detects market regimes (bull/bear/sideways)
//! 3. Trains a 3-level Hierarchical VAE
//! 4. Evaluates per-level KL (information partitioning)
//! 5. Generates multi-scale scenarios with fixed top-level latents
//! 6. Compares generated vs real regime statistics

use anyhow::Result;
use hierarchical_vae_trading::*;
use ndarray::{Array1, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<()> {
    println!("=== Hierarchical VAE Trading Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Fetch data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT 1h klines from Bybit...");
    let prices = match fetch_bybit_klines("BTCUSDT", "60", 500) {
        Ok(p) => {
            println!("Fetched {} candles from Bybit.\n", p.len());
            p
        }
        Err(e) => {
            println!(
                "Could not fetch from Bybit ({}), using synthetic data.\n",
                e
            );
            generate_synthetic_prices(500)
        }
    };

    // -----------------------------------------------------------------------
    // 2. Detect regimes
    // -----------------------------------------------------------------------
    let window = 10;
    let regimes = detect_regimes(&prices, window);

    let bull_count = regimes.iter().filter(|r| **r == MarketRegime::Bull).count();
    let bear_count = regimes.iter().filter(|r| **r == MarketRegime::Bear).count();
    let side_count = regimes
        .iter()
        .filter(|r| **r == MarketRegime::Sideways)
        .count();

    println!("Regime distribution:");
    println!(
        "  Bull:     {} ({:.1}%)",
        bull_count,
        100.0 * bull_count as f64 / regimes.len() as f64
    );
    println!(
        "  Bear:     {} ({:.1}%)",
        bear_count,
        100.0 * bear_count as f64 / regimes.len() as f64
    );
    println!(
        "  Sideways: {} ({:.1}%)\n",
        side_count,
        100.0 * side_count as f64 / regimes.len() as f64
    );

    // -----------------------------------------------------------------------
    // 3. Prepare training data
    // -----------------------------------------------------------------------
    let feature_window = 5;
    let data = prepare_training_data(&prices, feature_window);
    println!("Training samples: {}\n", data.len());

    if data.is_empty() {
        println!("Not enough data to train. Exiting.");
        return Ok(());
    }

    // -----------------------------------------------------------------------
    // 4. Create and train Hierarchical VAE
    // -----------------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(42);

    let config = HVAEConfig {
        input_dim: feature_window,
        latent_dims: vec![3, 2, 2], // bottom, middle, top
        hidden_dims: vec![16, 12, 8],
        free_bits: 0.1,
    };

    let mut hvae = HierarchicalVAE::new(config.clone(), &mut rng);

    println!(
        "Training 3-level HVAE (input={}, latent=[{},{},{}], hidden=[{},{},{}])...\n",
        config.input_dim,
        config.latent_dims[0],
        config.latent_dims[1],
        config.latent_dims[2],
        config.hidden_dims[0],
        config.hidden_dims[1],
        config.hidden_dims[2],
    );

    let train_config = TrainConfig {
        epochs: 30,
        lr: 0.001,
        warmup_fraction: 0.3,
    };

    let history = train_hvae(&mut hvae, &data, &train_config, &mut rng);
    println!(
        "\nFinal loss: {:.6}\n",
        history.loss.last().unwrap_or(&0.0)
    );

    // -----------------------------------------------------------------------
    // 5. Multi-scale evaluation
    // -----------------------------------------------------------------------
    println!("=== Multi-Scale Evaluation ===\n");

    let ms_metrics = evaluate_multi_scale(&hvae, &data, 200, &mut rng);

    println!("Per-level KL divergence (information per level):");
    for (l, kl) in ms_metrics.level_kl.iter().enumerate() {
        let label = match l {
            0 => "Bottom (intraday details)",
            1 => "Middle (daily patterns) ",
            _ => "Top    (macro regimes)  ",
        };
        println!("  Level {}: {} KL={:.6}", l, label, kl);
    }

    println!("\n=== Per-Regime Evaluation ===\n");
    for m in &ms_metrics.regime_metrics {
        println!("{}", m);
    }

    // -----------------------------------------------------------------------
    // 6. Multi-scale generation demo
    // -----------------------------------------------------------------------
    println!("\n=== Multi-Scale Generation ===\n");

    // Encode some samples to find typical top-level latents per regime
    println!("Top-level latent encodings by regime:");
    for regime_idx in 0..MarketRegime::COUNT {
        let regime = MarketRegime::from_index(regime_idx);
        let regime_data: Vec<&Array1<f64>> = data
            .iter()
            .filter(|(_, r)| *r == regime)
            .map(|(x, _)| x)
            .take(20)
            .collect();

        if regime_data.is_empty() {
            continue;
        }

        let mut top_latents_sum = Array1::<f64>::zeros(config.latent_dims[2]);
        for x in &regime_data {
            let top_z = hvae.encode_top_level(x, &mut rng);
            top_latents_sum = top_latents_sum + &top_z;
        }
        let mean_top = top_latents_sum / regime_data.len() as f64;
        println!(
            "  {:?}: mean top z = [{:.4}]",
            regime,
            mean_top
                .iter()
                .map(|v| format!("{:+.4}", v))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Generate scenarios with different fixed top-level latents
    println!("\n--- Scenario generation with fixed top-level ---\n");

    let top_latents = vec![
        ("Scenario A (positive top)", Array1::from(vec![1.0, 0.5])),
        ("Scenario B (negative top)", Array1::from(vec![-1.0, -0.5])),
        ("Scenario C (neutral top)", Array1::from(vec![0.0, 0.0])),
    ];

    for (label, top_z) in &top_latents {
        let samples = hvae.generate_multi_scale(top_z, 500, &mut rng);
        let mean = samples.mean_axis(Axis(0)).unwrap();
        let var = samples.var_axis(Axis(0), 0.0);

        println!("{}:", label);
        println!(
            "  Mean: [{}]",
            mean.iter()
                .map(|v| format!("{:+.4}", v))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "  Var:  [{}]",
            var.iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // -----------------------------------------------------------------------
    // 7. Show sample generated windows
    // -----------------------------------------------------------------------
    println!("\n=== Sample Generated Windows (free generation) ===\n");

    let free_samples = hvae.generate(5, None, &mut rng);
    for i in 0..free_samples.nrows() {
        let row: Vec<String> = free_samples
            .row(i)
            .iter()
            .map(|v| format!("{:+.4}", v))
            .collect();
        println!("  [{}]", row.join(", "));
    }

    println!("\nDone!");
    Ok(())
}

/// Generate synthetic price data for offline testing.
fn generate_synthetic_prices(n: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(123);
    let mut prices = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        let drift = if i < n / 3 {
            0.002 // bull
        } else if i < 2 * n / 3 {
            -0.003 // bear
        } else {
            0.0001 // sideways
        };

        let vol = if i < n / 3 {
            0.01
        } else if i < 2 * n / 3 {
            0.025
        } else {
            0.008
        };

        let r = drift + vol * rng.gen_range(-1.0..1.0);
        price *= 1.0 + r;
        prices.push(price);
    }

    prices
}
