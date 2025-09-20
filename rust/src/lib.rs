//! Hierarchical VAE (HVAE) for Trading
//!
//! Implements a multi-level Hierarchical Variational Autoencoder that learns
//! multi-scale latent representations of financial time series. The hierarchy
//! naturally decomposes market dynamics into macro regimes (top level),
//! daily patterns (middle level), and intraday details (bottom level).

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Market Regime
// ---------------------------------------------------------------------------

/// Market regime labels used for evaluation and conditioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
}

impl MarketRegime {
    pub const COUNT: usize = 3;

    /// One-hot encode the regime.
    pub fn one_hot(&self) -> Array1<f64> {
        let mut v = Array1::zeros(Self::COUNT);
        v[self.index()] = 1.0;
        v
    }

    pub fn index(&self) -> usize {
        match self {
            MarketRegime::Bull => 0,
            MarketRegime::Bear => 1,
            MarketRegime::Sideways => 2,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => MarketRegime::Bull,
            1 => MarketRegime::Bear,
            _ => MarketRegime::Sideways,
        }
    }
}

// ---------------------------------------------------------------------------
// Regime detection
// ---------------------------------------------------------------------------

/// Detect market regimes from a price series using rolling statistics.
pub fn detect_regimes(prices: &[f64], window: usize) -> Vec<MarketRegime> {
    if prices.len() < 2 {
        return vec![];
    }

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    if returns.len() < window {
        return returns.iter().map(|_| MarketRegime::Sideways).collect();
    }

    let global_vol = {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        (returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64)
            .sqrt()
    };

    let mut regimes = Vec::with_capacity(returns.len());

    for i in 0..returns.len() {
        if i < window {
            regimes.push(MarketRegime::Sideways);
            continue;
        }
        let slice = &returns[i - window..i];
        let roll_mean = slice.iter().sum::<f64>() / window as f64;
        let roll_vol = (slice
            .iter()
            .map(|r| (r - roll_mean).powi(2))
            .sum::<f64>()
            / window as f64)
            .sqrt();

        let regime = if roll_mean > 0.0 && roll_vol <= global_vol {
            MarketRegime::Bull
        } else if roll_mean < 0.0 && roll_vol > global_vol {
            MarketRegime::Bear
        } else {
            MarketRegime::Sideways
        };
        regimes.push(regime);
    }

    regimes
}

// ---------------------------------------------------------------------------
// Simple linear layer helper
// ---------------------------------------------------------------------------

/// A dense (fully-connected) layer: y = relu(x * W + b) or y = x * W + b.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Array2<f64>,
    pub bias: Array1<f64>,
}

impl Linear {
    pub fn new_random(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / input_dim as f64).sqrt();
        let weight = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.dot(&self.weight) + &self.bias
    }

    pub fn forward_relu(&self, x: &Array1<f64>) -> Array1<f64> {
        let out = self.forward(x);
        out.mapv(|v| v.max(0.0))
    }

    /// Simple SGD update: w -= lr * grad.
    pub fn update(&mut self, weight_grad: &Array2<f64>, bias_grad: &Array1<f64>, lr: f64) {
        self.weight.scaled_add(-lr, weight_grad);
        self.bias.scaled_add(-lr, bias_grad);
    }
}

// ---------------------------------------------------------------------------
// Level Module — one level of the hierarchy
// ---------------------------------------------------------------------------

/// Represents one level in the hierarchical VAE.
///
/// Each level has:
/// - A bottom-up encoder that extracts features from the input or previous level
/// - A top-down prior that maps parent latent → (mu_prior, logvar_prior)
/// - A posterior that merges bottom-up features with top-down context
///   to produce (mu_q, logvar_q)
#[derive(Debug, Clone)]
pub struct HVAELevel {
    /// Bottom-up feature extraction layer
    pub bottom_up: Linear,
    /// Maps parent latent to prior mu
    pub prior_mu: Linear,
    /// Maps parent latent to prior log-variance
    pub prior_logvar: Linear,
    /// Maps bottom-up features to posterior mu contribution
    pub posterior_mu: Linear,
    /// Maps bottom-up features to posterior log-variance contribution
    pub posterior_logvar: Linear,
    /// Latent dimensionality at this level
    pub latent_dim: usize,
    /// Whether this is the top level (uses standard normal prior)
    pub is_top: bool,
}

impl HVAELevel {
    /// Create a new HVAE level.
    ///
    /// * `input_dim` – input feature dimension for this level's bottom-up pass
    /// * `hidden_dim` – hidden layer dimension for bottom-up features
    /// * `latent_dim` – latent dimensionality at this level
    /// * `parent_latent_dim` – latent dim of parent level (0 for top level)
    /// * `is_top` – whether this is the top level
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        latent_dim: usize,
        parent_latent_dim: usize,
        is_top: bool,
        rng: &mut impl Rng,
    ) -> Self {
        let prior_input = if is_top { 1 } else { parent_latent_dim };

        Self {
            bottom_up: Linear::new_random(input_dim, hidden_dim, rng),
            prior_mu: Linear::new_random(prior_input, latent_dim, rng),
            prior_logvar: Linear::new_random(prior_input, latent_dim, rng),
            posterior_mu: Linear::new_random(hidden_dim, latent_dim, rng),
            posterior_logvar: Linear::new_random(hidden_dim, latent_dim, rng),
            latent_dim,
            is_top,
        }
    }

    /// Compute bottom-up features from input.
    pub fn bottom_up_features(&self, x: &Array1<f64>) -> Array1<f64> {
        self.bottom_up.forward_relu(x)
    }

    /// Compute prior distribution parameters p(z_l | z_{l+1}).
    /// For the top level, returns standard normal (0, 0) parameters.
    pub fn prior(&self, parent_z: Option<&Array1<f64>>) -> (Array1<f64>, Array1<f64>) {
        if self.is_top {
            // Standard normal prior for top level
            (
                Array1::zeros(self.latent_dim),
                Array1::zeros(self.latent_dim),
            )
        } else {
            let parent = parent_z.expect("Non-top level requires parent latent");
            let mu = self.prior_mu.forward(parent);
            let logvar = self.prior_logvar.forward(parent);
            (mu, logvar)
        }
    }

    /// Compute posterior distribution using precision-weighted merge.
    ///
    /// Bottom-up provides (mu_bu, logvar_bu) from data.
    /// Top-down provides (mu_td, logvar_td) from prior.
    /// The merge follows the Ladder VAE approach:
    ///
    /// mu_q = (mu_bu * var_td + mu_td * var_bu) / (var_bu + var_td)
    /// var_q = (var_bu * var_td) / (var_bu + var_td)
    pub fn posterior(
        &self,
        bottom_up_features: &Array1<f64>,
        prior_mu: &Array1<f64>,
        prior_logvar: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let mu_bu = self.posterior_mu.forward(bottom_up_features);
        let logvar_bu = self.posterior_logvar.forward(bottom_up_features);

        // Precision-weighted merge
        let var_bu = logvar_bu.mapv(f64::exp);
        let var_td = prior_logvar.mapv(f64::exp);

        let var_sum = &var_bu + &var_td;
        let mu_q = (&mu_bu * &var_td + prior_mu * &var_bu) / &var_sum;
        let var_q = (&var_bu * &var_td) / &var_sum;
        let logvar_q = var_q.mapv(|v| v.max(1e-10).ln());

        (mu_q, logvar_q)
    }
}

// ---------------------------------------------------------------------------
// Hierarchical VAE
// ---------------------------------------------------------------------------

/// Configuration for the Hierarchical VAE.
#[derive(Debug, Clone)]
pub struct HVAEConfig {
    /// Input data dimensionality
    pub input_dim: usize,
    /// Latent dimensions for each level (bottom to top)
    pub latent_dims: Vec<usize>,
    /// Hidden layer dimensions for each level
    pub hidden_dims: Vec<usize>,
    /// Free bits per level (minimum KL per dimension)
    pub free_bits: f64,
}

impl Default for HVAEConfig {
    fn default() -> Self {
        Self {
            input_dim: 5,
            latent_dims: vec![3, 2, 2],  // bottom, middle, top
            hidden_dims: vec![16, 12, 8],
            free_bits: 0.1,
        }
    }
}

/// Complete Hierarchical VAE for trading data.
///
/// Organizes latent space into multiple levels:
/// - Level 0 (bottom): fine-grained details (intraday patterns)
/// - Level 1 (middle): medium-scale patterns (daily momentum)
/// - Level 2 (top): coarse-grained structure (macro regimes)
pub struct HierarchicalVAE {
    /// Levels of the hierarchy, from bottom (0) to top (L-1)
    pub levels: Vec<HVAELevel>,
    /// Decoder: maps bottom-level latent z_0 → reconstructed x
    pub decoder_hidden: Linear,
    pub decoder_output: Linear,
    /// Configuration
    pub config: HVAEConfig,
}

impl HierarchicalVAE {
    /// Create a new Hierarchical VAE.
    pub fn new(config: HVAEConfig, rng: &mut impl Rng) -> Self {
        let num_levels = config.latent_dims.len();
        assert!(num_levels >= 2, "Need at least 2 levels for hierarchy");

        let mut levels = Vec::with_capacity(num_levels);

        for l in 0..num_levels {
            let is_top = l == num_levels - 1;
            let input_dim = if l == 0 {
                config.input_dim
            } else {
                config.hidden_dims[l - 1]
            };
            let parent_latent_dim = if is_top {
                0
            } else {
                config.latent_dims[l + 1]
            };

            levels.push(HVAELevel::new(
                input_dim,
                config.hidden_dims[l],
                config.latent_dims[l],
                parent_latent_dim,
                is_top,
                rng,
            ));
        }

        let decoder_hidden =
            Linear::new_random(config.latent_dims[0], config.hidden_dims[0], rng);
        let decoder_output =
            Linear::new_random(config.hidden_dims[0], config.input_dim, rng);

        Self {
            levels,
            decoder_hidden,
            decoder_output,
            config,
        }
    }

    /// Reparameterization trick: sample z = mu + sigma * eps.
    pub fn reparameterize(
        mu: &Array1<f64>,
        logvar: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        let std = logvar.mapv(|lv| (lv * 0.5).exp());
        let eps = Array1::from_shape_fn(mu.len(), |_| {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        });
        mu + &(std * &eps)
    }

    /// Full forward pass through the hierarchy.
    ///
    /// Returns `HVAEOutput` containing reconstruction, per-level latents,
    /// and per-level KL divergences.
    pub fn forward(&self, x: &Array1<f64>, rng: &mut impl Rng) -> HVAEOutput {
        let num_levels = self.levels.len();

        // --- Bottom-up pass: extract features at each level ---
        let mut bu_features = Vec::with_capacity(num_levels);
        let mut current_input = x.clone();

        for level in &self.levels {
            let features = level.bottom_up_features(&current_input);
            current_input = features.clone();
            bu_features.push(features);
        }

        // --- Top-down pass: sample latents from top to bottom ---
        let mut latent_samples = vec![Array1::zeros(0); num_levels];
        let mut kl_per_level = vec![0.0_f64; num_levels];
        let mut mu_per_level = Vec::with_capacity(num_levels);
        let mut logvar_per_level = Vec::with_capacity(num_levels);

        // Start from the top level
        for l in (0..num_levels).rev() {
            let level = &self.levels[l];
            let parent_z = if l == num_levels - 1 {
                None
            } else {
                Some(&latent_samples[l + 1])
            };

            // Prior
            let (mu_prior, logvar_prior) = level.prior(parent_z);

            // Posterior (precision-weighted merge)
            let (mu_q, logvar_q) =
                level.posterior(&bu_features[l], &mu_prior, &logvar_prior);

            // Sample
            let z = Self::reparameterize(&mu_q, &logvar_q, rng);

            // KL divergence
            let kl = kl_divergence(&mu_q, &logvar_q, &mu_prior, &logvar_prior);

            // Apply free bits
            let kl_effective = kl.max(self.config.free_bits * level.latent_dim as f64);

            latent_samples[l] = z;
            kl_per_level[l] = kl_effective;
            mu_per_level.push(mu_q);
            logvar_per_level.push(logvar_q);
        }

        // Reverse mu/logvar so they are bottom-to-top
        mu_per_level.reverse();
        logvar_per_level.reverse();

        // --- Decode from bottom-level latent ---
        let h = self.decoder_hidden.forward_relu(&latent_samples[0]);
        let reconstruction = self.decoder_output.forward(&h);

        // Reconstruction loss (MSE)
        let recon_loss =
            (x - &reconstruction).mapv(|v| v.powi(2)).sum() / x.len() as f64;

        HVAEOutput {
            reconstruction,
            latent_samples,
            kl_per_level,
            recon_loss,
            mu_per_level,
            logvar_per_level,
        }
    }

    /// Generate samples by sampling from the hierarchical prior.
    ///
    /// Optionally fix top-level latent for regime-controlled generation.
    pub fn generate(
        &self,
        num_samples: usize,
        fixed_top_z: Option<&Array1<f64>>,
        rng: &mut impl Rng,
    ) -> Array2<f64> {
        let num_levels = self.levels.len();
        let mut samples = Array2::zeros((num_samples, self.config.input_dim));

        for s in 0..num_samples {
            let mut latent_samples = vec![Array1::zeros(0); num_levels];

            // Top-down generation through the hierarchy
            for l in (0..num_levels).rev() {
                let level = &self.levels[l];

                if l == num_levels - 1 {
                    // Top level
                    if let Some(fixed_z) = fixed_top_z {
                        latent_samples[l] = fixed_z.clone();
                    } else {
                        let (mu, logvar) = level.prior(None);
                        latent_samples[l] = Self::reparameterize(&mu, &logvar, rng);
                    }
                } else {
                    // Lower levels: sample from conditional prior
                    let parent_z = &latent_samples[l + 1];
                    let (mu, logvar) = level.prior(Some(parent_z));
                    latent_samples[l] = Self::reparameterize(&mu, &logvar, rng);
                }
            }

            // Decode from bottom-level latent
            let h = self.decoder_hidden.forward_relu(&latent_samples[0]);
            let x = self.decoder_output.forward(&h);
            samples.row_mut(s).assign(&x);
        }

        samples
    }

    /// Generate samples with a fixed top-level latent and varying lower levels.
    /// This produces regime-consistent but diverse scenarios.
    pub fn generate_multi_scale(
        &self,
        top_z: &Array1<f64>,
        num_samples: usize,
        rng: &mut impl Rng,
    ) -> Array2<f64> {
        self.generate(num_samples, Some(top_z), rng)
    }

    /// Encode input data and return the top-level latent representation.
    /// Useful for extracting macro regime features.
    pub fn encode_top_level(
        &self,
        x: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        let output = self.forward(x, rng);
        let top_idx = self.levels.len() - 1;
        output.latent_samples[top_idx].clone()
    }
}

// ---------------------------------------------------------------------------
// HVAE Output
// ---------------------------------------------------------------------------

/// Output of a hierarchical VAE forward pass.
pub struct HVAEOutput {
    /// Reconstructed input
    pub reconstruction: Array1<f64>,
    /// Latent samples at each level (bottom to top)
    pub latent_samples: Vec<Array1<f64>>,
    /// KL divergence at each level (with free bits applied)
    pub kl_per_level: Vec<f64>,
    /// Reconstruction loss (MSE)
    pub recon_loss: f64,
    /// Posterior means at each level
    pub mu_per_level: Vec<Array1<f64>>,
    /// Posterior log-variances at each level
    pub logvar_per_level: Vec<Array1<f64>>,
}

impl HVAEOutput {
    /// Total ELBO loss = reconstruction + beta * sum(KL per level).
    pub fn total_loss(&self, beta: f64) -> f64 {
        let total_kl: f64 = self.kl_per_level.iter().sum();
        self.recon_loss + beta * total_kl
    }
}

// ---------------------------------------------------------------------------
// KL Divergence
// ---------------------------------------------------------------------------

/// KL divergence between two diagonal Gaussians.
///
/// KL(q || p) = 0.5 * sum(log(var_p/var_q) - 1 + var_q/var_p + (mu_q - mu_p)^2/var_p)
pub fn kl_divergence(
    mu_q: &Array1<f64>,
    logvar_q: &Array1<f64>,
    mu_p: &Array1<f64>,
    logvar_p: &Array1<f64>,
) -> f64 {
    0.5 * (logvar_p - logvar_q
        + (logvar_q.mapv(f64::exp) + (mu_q - mu_p).mapv(|v| v.powi(2)))
            / logvar_p.mapv(f64::exp)
        - 1.0)
        .sum()
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// Training configuration for the Hierarchical VAE.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub lr: f64,
    /// Fraction of training to use for KL warmup (0.0 to 1.0)
    pub warmup_fraction: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 30,
            lr: 0.001,
            warmup_fraction: 0.3,
        }
    }
}

/// Train the Hierarchical VAE using finite-difference gradient estimation.
///
/// Uses KL warmup annealing to prevent posterior collapse.
/// This is a simplified training loop suitable for demonstration.
pub fn train_hvae(
    hvae: &mut HierarchicalVAE,
    data: &[(Array1<f64>, MarketRegime)],
    config: &TrainConfig,
    rng: &mut impl Rng,
) -> TrainHistory {
    let mut history = TrainHistory {
        loss: Vec::with_capacity(config.epochs),
        recon_loss: Vec::with_capacity(config.epochs),
        kl_per_level: Vec::with_capacity(config.epochs),
    };

    let warmup_epochs = (config.epochs as f64 * config.warmup_fraction) as usize;

    for epoch in 0..config.epochs {
        // KL warmup: linearly increase beta from 0 to 1
        let beta = if warmup_epochs > 0 && epoch < warmup_epochs {
            epoch as f64 / warmup_epochs as f64
        } else {
            1.0
        };

        let mut total_loss = 0.0;
        let mut total_recon = 0.0;
        let mut total_kl_levels = vec![0.0_f64; hvae.levels.len()];

        for (x, _regime) in data.iter() {
            let output = hvae.forward(x, rng);
            let loss = output.total_loss(beta);
            total_loss += loss;
            total_recon += output.recon_loss;
            for (l, kl) in output.kl_per_level.iter().enumerate() {
                total_kl_levels[l] += kl;
            }

            // --- Numerical gradient for decoder layers ---
            let eps = 1e-4;

            // Update decoder output weights
            let (rows, cols) = hvae.decoder_output.weight.dim();
            let mut w_grad = Array2::zeros((rows, cols));
            for r in 0..rows {
                for col in 0..cols {
                    hvae.decoder_output.weight[[r, col]] += eps;
                    let out_p = hvae.forward(x, rng);
                    let loss_p = out_p.total_loss(beta);
                    hvae.decoder_output.weight[[r, col]] -= eps;
                    w_grad[[r, col]] = (loss_p - loss) / eps;
                }
            }
            let mut b_grad = Array1::zeros(cols);
            for col in 0..cols {
                hvae.decoder_output.bias[col] += eps;
                let out_p = hvae.forward(x, rng);
                let loss_p = out_p.total_loss(beta);
                hvae.decoder_output.bias[col] -= eps;
                b_grad[col] = (loss_p - loss) / eps;
            }
            hvae.decoder_output.update(&w_grad, &b_grad, config.lr);

            // Update decoder hidden weights
            let (rows, cols) = hvae.decoder_hidden.weight.dim();
            let mut w_grad = Array2::zeros((rows, cols));
            for r in 0..rows {
                for col in 0..cols {
                    hvae.decoder_hidden.weight[[r, col]] += eps;
                    let out_p = hvae.forward(x, rng);
                    let loss_p = out_p.total_loss(beta);
                    hvae.decoder_hidden.weight[[r, col]] -= eps;
                    w_grad[[r, col]] = (loss_p - loss) / eps;
                }
            }
            let mut b_grad = Array1::zeros(cols);
            for col in 0..cols {
                hvae.decoder_hidden.bias[col] += eps;
                let out_p = hvae.forward(x, rng);
                let loss_p = out_p.total_loss(beta);
                hvae.decoder_hidden.bias[col] -= eps;
                b_grad[col] = (loss_p - loss) / eps;
            }
            hvae.decoder_hidden.update(&w_grad, &b_grad, config.lr);
        }

        let n = data.len() as f64;
        let avg_loss = total_loss / n;
        let avg_recon = total_recon / n;
        let avg_kl: Vec<f64> = total_kl_levels.iter().map(|k| k / n).collect();

        history.loss.push(avg_loss);
        history.recon_loss.push(avg_recon);
        history.kl_per_level.push(avg_kl.clone());

        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            let kl_str: Vec<String> = avg_kl.iter().map(|k| format!("{:.4}", k)).collect();
            println!(
                "Epoch {}/{}: loss={:.6} recon={:.6} kl=[{}] beta={:.3}",
                epoch + 1,
                config.epochs,
                avg_loss,
                avg_recon,
                kl_str.join(", "),
                beta,
            );
        }
    }

    history
}

/// Training history tracking loss components per epoch.
pub struct TrainHistory {
    /// Total loss per epoch
    pub loss: Vec<f64>,
    /// Reconstruction loss per epoch
    pub recon_loss: Vec<f64>,
    /// KL per level per epoch
    pub kl_per_level: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Quality Metrics
// ---------------------------------------------------------------------------

/// Per-regime quality metrics comparing generated vs real data.
pub struct RegimeMetrics {
    pub regime: MarketRegime,
    pub real_mean: f64,
    pub real_std: f64,
    pub gen_mean: f64,
    pub gen_std: f64,
    pub mean_error: f64,
    pub std_error: f64,
    pub wasserstein_approx: f64,
}

impl std::fmt::Display for RegimeMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}: real(mean={:.6}, std={:.6}) gen(mean={:.6}, std={:.6}) \
             err(mean={:.6}, std={:.6}) wasserstein~{:.6}",
            self.regime,
            self.real_mean,
            self.real_std,
            self.gen_mean,
            self.gen_std,
            self.mean_error,
            self.std_error,
            self.wasserstein_approx
        )
    }
}

/// Multi-scale quality metrics.
pub struct MultiScaleMetrics {
    /// Per-level average KL divergence (how active each level is)
    pub level_kl: Vec<f64>,
    /// Per-regime metrics
    pub regime_metrics: Vec<RegimeMetrics>,
}

/// Evaluate generated samples against real data for each regime.
pub fn evaluate_per_regime(
    hvae: &HierarchicalVAE,
    data: &[(Array1<f64>, MarketRegime)],
    samples_per_regime: usize,
    rng: &mut impl Rng,
) -> Vec<RegimeMetrics> {
    let mut metrics = Vec::new();

    for regime_idx in 0..MarketRegime::COUNT {
        let regime = MarketRegime::from_index(regime_idx);

        let real_values: Vec<f64> = data
            .iter()
            .filter(|(_, r)| *r == regime)
            .flat_map(|(x, _)| x.iter().cloned())
            .collect();

        if real_values.is_empty() {
            continue;
        }

        let real_mean = real_values.iter().sum::<f64>() / real_values.len() as f64;
        let real_std = (real_values
            .iter()
            .map(|v| (v - real_mean).powi(2))
            .sum::<f64>()
            / real_values.len() as f64)
            .sqrt();

        // Generate samples (no fixed top-level, free generation)
        let generated = hvae.generate(samples_per_regime, None, rng);
        let gen_values: Vec<f64> = generated.iter().cloned().collect();

        let gen_mean = gen_values.iter().sum::<f64>() / gen_values.len() as f64;
        let gen_std = (gen_values
            .iter()
            .map(|v| (v - gen_mean).powi(2))
            .sum::<f64>()
            / gen_values.len() as f64)
            .sqrt();

        // Approximate Wasserstein distance
        let mut real_sorted = real_values.clone();
        let mut gen_sorted = gen_values;
        real_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        gen_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n_quantiles = 100;
        let wasserstein: f64 = (0..n_quantiles)
            .map(|i| {
                let q = i as f64 / n_quantiles as f64;
                let ri = (q * (real_sorted.len() - 1) as f64) as usize;
                let gi = (q * (gen_sorted.len() - 1) as f64) as usize;
                (real_sorted[ri] - gen_sorted[gi]).abs()
            })
            .sum::<f64>()
            / n_quantiles as f64;

        metrics.push(RegimeMetrics {
            regime,
            real_mean,
            real_std,
            gen_mean,
            gen_std,
            mean_error: (real_mean - gen_mean).abs(),
            std_error: (real_std - gen_std).abs(),
            wasserstein_approx: wasserstein,
        });
    }

    metrics
}

/// Evaluate multi-scale quality: per-level KL and per-regime stats.
pub fn evaluate_multi_scale(
    hvae: &HierarchicalVAE,
    data: &[(Array1<f64>, MarketRegime)],
    samples_per_regime: usize,
    rng: &mut impl Rng,
) -> MultiScaleMetrics {
    let num_levels = hvae.levels.len();
    let mut level_kl_sums = vec![0.0_f64; num_levels];

    for (x, _) in data.iter() {
        let output = hvae.forward(x, rng);
        for (l, kl) in output.kl_per_level.iter().enumerate() {
            level_kl_sums[l] += kl;
        }
    }

    let n = data.len() as f64;
    let level_kl: Vec<f64> = level_kl_sums.iter().map(|k| k / n).collect();

    let regime_metrics = evaluate_per_regime(hvae, data, samples_per_regime, rng);

    MultiScaleMetrics {
        level_kl,
        regime_metrics,
    }
}

// ---------------------------------------------------------------------------
// Data preparation helpers
// ---------------------------------------------------------------------------

/// Convert prices + regimes into windowed training samples.
pub fn prepare_training_data(
    prices: &[f64],
    window: usize,
) -> Vec<(Array1<f64>, MarketRegime)> {
    if prices.len() < 2 {
        return vec![];
    }

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let regimes = detect_regimes(prices, window);

    // Normalize returns
    let mean_r = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_r = (returns
        .iter()
        .map(|r| (r - mean_r).powi(2))
        .sum::<f64>()
        / returns.len() as f64)
        .sqrt()
        .max(1e-8);
    let norm_returns: Vec<f64> = returns.iter().map(|r| (r - mean_r) / std_r).collect();

    let mut data = Vec::new();
    for i in window..norm_returns.len() {
        let x = Array1::from(norm_returns[i - window..i].to_vec());
        data.push((x, regimes[i]));
    }
    data
}

// ---------------------------------------------------------------------------
// Bybit API Integration
// ---------------------------------------------------------------------------

/// Response structure for Bybit kline endpoint.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// Fetch OHLCV kline data from Bybit.
///
/// * `symbol` – e.g. "BTCUSDT"
/// * `interval` – e.g. "60" for 1h, "D" for daily
/// * `limit` – number of candles (max 1000)
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<f64>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("Accept", "application/json")
        .send()
        .map_err(|e| anyhow!("HTTP request failed: {}", e))?
        .json()
        .map_err(|e| anyhow!("JSON parse failed: {}", e))?;

    if resp.ret_code != 0 {
        return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
    }

    let mut prices: Vec<f64> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 5 {
                row[4].parse::<f64>().ok()
            } else {
                None
            }
        })
        .collect();

    prices.reverse(); // oldest first
    Ok(prices)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Axis;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn default_config() -> HVAEConfig {
        HVAEConfig {
            input_dim: 5,
            latent_dims: vec![3, 2, 2],
            hidden_dims: vec![16, 12, 8],
            free_bits: 0.1,
        }
    }

    #[test]
    fn test_regime_one_hot() {
        assert_eq!(
            MarketRegime::Bull.one_hot(),
            Array1::from(vec![1.0, 0.0, 0.0])
        );
        assert_eq!(
            MarketRegime::Bear.one_hot(),
            Array1::from(vec![0.0, 1.0, 0.0])
        );
        assert_eq!(
            MarketRegime::Sideways.one_hot(),
            Array1::from(vec![0.0, 0.0, 1.0])
        );
    }

    #[test]
    fn test_regime_from_index_roundtrip() {
        for i in 0..MarketRegime::COUNT {
            let r = MarketRegime::from_index(i);
            assert_eq!(r.index(), i);
        }
    }

    #[test]
    fn test_detect_regimes_basic() {
        let mut prices = Vec::new();
        let mut p: f64 = 100.0;

        // Bull phase
        for _ in 0..40 {
            p *= 1.005;
            prices.push(p);
        }
        // Transition
        for i in 0..40 {
            let noise = if i % 2 == 0 { 1.008 } else { 0.992 };
            p *= noise;
            prices.push(p);
        }
        // Bear phase
        for i in 0..40 {
            let shock = if i % 3 == 0 { 0.96 } else { 0.985 };
            p *= shock;
            prices.push(p);
        }

        let regimes = detect_regimes(&prices, 10);
        assert_eq!(regimes.len(), prices.len() - 1);

        let bull_count = regimes.iter().filter(|r| **r == MarketRegime::Bull).count();
        let bear_count = regimes.iter().filter(|r| **r == MarketRegime::Bear).count();
        let side_count = regimes
            .iter()
            .filter(|r| **r == MarketRegime::Sideways)
            .count();

        let types_detected = [bull_count > 0, bear_count > 0, side_count > 0]
            .iter()
            .filter(|&&x| x)
            .count();
        assert!(
            types_detected >= 2,
            "Expected at least 2 regime types, got {} (bull={}, bear={}, side={})",
            types_detected,
            bull_count,
            bear_count,
            side_count
        );
    }

    #[test]
    fn test_hvae_creation() {
        let mut rng = make_rng();
        let config = default_config();
        let hvae = HierarchicalVAE::new(config, &mut rng);

        assert_eq!(hvae.levels.len(), 3);
        assert!(hvae.levels[2].is_top);
        assert!(!hvae.levels[0].is_top);
        assert!(!hvae.levels[1].is_top);
    }

    #[test]
    fn test_hvae_forward_dimensions() {
        let mut rng = make_rng();
        let config = default_config();
        let hvae = HierarchicalVAE::new(config.clone(), &mut rng);

        let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1]);
        let output = hvae.forward(&x, &mut rng);

        assert_eq!(output.reconstruction.len(), config.input_dim);
        assert_eq!(output.latent_samples.len(), 3);
        assert_eq!(output.latent_samples[0].len(), 3); // bottom
        assert_eq!(output.latent_samples[1].len(), 2); // middle
        assert_eq!(output.latent_samples[2].len(), 2); // top
        assert_eq!(output.kl_per_level.len(), 3);
    }

    #[test]
    fn test_hvae_loss_components() {
        let mut rng = make_rng();
        let config = default_config();
        let hvae = HierarchicalVAE::new(config, &mut rng);

        let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1]);
        let output = hvae.forward(&x, &mut rng);

        assert!(output.recon_loss >= 0.0, "Recon loss should be non-negative");
        for (l, kl) in output.kl_per_level.iter().enumerate() {
            assert!(*kl >= 0.0, "KL at level {} should be non-negative", l);
        }
        let total = output.total_loss(1.0);
        assert!(total >= 0.0, "Total loss should be non-negative");
    }

    #[test]
    fn test_hvae_generate_shape() {
        let mut rng = make_rng();
        let config = default_config();
        let hvae = HierarchicalVAE::new(config.clone(), &mut rng);

        let samples = hvae.generate(100, None, &mut rng);
        assert_eq!(samples.dim(), (100, config.input_dim));
    }

    #[test]
    fn test_hvae_generate_with_fixed_top() {
        let mut rng = make_rng();
        let config = default_config();
        let hvae = HierarchicalVAE::new(config.clone(), &mut rng);

        let fixed_top = Array1::from(vec![1.0, -0.5]);
        let samples = hvae.generate_multi_scale(&fixed_top, 50, &mut rng);
        assert_eq!(samples.dim(), (50, config.input_dim));
    }

    #[test]
    fn test_different_top_latents_produce_different_outputs() {
        let mut rng = make_rng();
        let config = default_config();
        let hvae = HierarchicalVAE::new(config, &mut rng);

        let top_a = Array1::from(vec![2.0, 0.0]);
        let top_b = Array1::from(vec![-2.0, 0.0]);

        let samples_a = hvae.generate_multi_scale(&top_a, 500, &mut rng);
        let samples_b = hvae.generate_multi_scale(&top_b, 500, &mut rng);

        let mean_a = samples_a.mean_axis(Axis(0)).unwrap();
        let mean_b = samples_b.mean_axis(Axis(0)).unwrap();

        let diff = (&mean_a - &mean_b).mapv(|v| v.abs()).sum();
        assert!(
            diff > 1e-6,
            "Different top latents should produce different outputs"
        );
    }

    #[test]
    fn test_kl_divergence_same_distributions() {
        let mu = Array1::from(vec![0.0, 0.0]);
        let logvar = Array1::from(vec![0.0, 0.0]);
        let kl = kl_divergence(&mu, &logvar, &mu, &logvar);
        assert!(
            kl.abs() < 1e-10,
            "KL should be 0 for identical distributions"
        );
    }

    #[test]
    fn test_kl_divergence_different_distributions() {
        let mu_q = Array1::from(vec![1.0, 2.0]);
        let logvar_q = Array1::from(vec![0.5, 0.5]);
        let mu_p = Array1::from(vec![0.0, 0.0]);
        let logvar_p = Array1::from(vec![0.0, 0.0]);

        let kl = kl_divergence(&mu_q, &logvar_q, &mu_p, &logvar_p);
        assert!(
            kl > 0.0,
            "KL should be positive for different distributions"
        );
    }

    #[test]
    fn test_reparameterize() {
        let mut rng = make_rng();
        let mu = Array1::from(vec![0.0, 0.0, 0.0]);
        let logvar = Array1::from(vec![0.0, 0.0, 0.0]);

        let z = HierarchicalVAE::reparameterize(&mu, &logvar, &mut rng);
        assert_eq!(z.len(), 3);
        for &v in z.iter() {
            assert!(v.abs() < 10.0, "Sample too extreme: {}", v);
        }
    }

    #[test]
    fn test_prepare_training_data() {
        let mut prices = Vec::new();
        let mut p: f64 = 100.0;
        for _ in 0..100 {
            p *= 1.0 + 0.001 * p.sin();
            prices.push(p);
        }

        let window = 5;
        let data = prepare_training_data(&prices, window);
        assert!(!data.is_empty());

        for (x, _regime) in &data {
            assert_eq!(x.len(), window);
        }
    }

    #[test]
    fn test_level_prior_top() {
        let mut rng = make_rng();
        let level = HVAELevel::new(5, 16, 3, 0, true, &mut rng);

        let (mu, logvar) = level.prior(None);
        assert_eq!(mu.len(), 3);
        assert_eq!(logvar.len(), 3);
        // Top level uses standard normal: mu=0, logvar=0
        assert!(mu.iter().all(|v| *v == 0.0));
        assert!(logvar.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_level_prior_non_top() {
        let mut rng = make_rng();
        let level = HVAELevel::new(16, 12, 2, 3, false, &mut rng);

        let parent_z = Array1::from(vec![0.5, -0.3, 0.8]);
        let (mu, logvar) = level.prior(Some(&parent_z));
        assert_eq!(mu.len(), 2);
        assert_eq!(logvar.len(), 2);
    }

    #[test]
    fn test_level_posterior() {
        let mut rng = make_rng();
        let level = HVAELevel::new(5, 16, 3, 0, true, &mut rng);

        let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1]);
        let bu_features = level.bottom_up_features(&x);

        let prior_mu = Array1::zeros(3);
        let prior_logvar = Array1::zeros(3);

        let (mu_q, logvar_q) = level.posterior(&bu_features, &prior_mu, &prior_logvar);
        assert_eq!(mu_q.len(), 3);
        assert_eq!(logvar_q.len(), 3);
    }

    #[test]
    fn test_encode_top_level() {
        let mut rng = make_rng();
        let config = default_config();
        let hvae = HierarchicalVAE::new(config, &mut rng);

        let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1]);
        let top_z = hvae.encode_top_level(&x, &mut rng);
        assert_eq!(top_z.len(), 2); // top latent dim = 2
    }

    #[test]
    fn test_multi_scale_evaluation() {
        let mut rng = make_rng();
        let config = default_config();
        let hvae = HierarchicalVAE::new(config, &mut rng);

        // Create simple data
        let data: Vec<(Array1<f64>, MarketRegime)> = (0..20)
            .map(|i| {
                let regime = MarketRegime::from_index(i % 3);
                let x = Array1::from(vec![0.1 * i as f64; 5]);
                (x, regime)
            })
            .collect();

        let metrics = evaluate_multi_scale(&hvae, &data, 50, &mut rng);
        assert_eq!(metrics.level_kl.len(), 3);
        assert!(!metrics.regime_metrics.is_empty());
    }
}
