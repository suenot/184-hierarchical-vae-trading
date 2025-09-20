# Chapter 237: Hierarchical VAE Trading

## 1. Introduction — Multi-Scale Latent Representations for Financial Markets

Financial markets operate simultaneously across multiple time scales. Intraday tick dynamics are nested within daily trend patterns, which themselves unfold within weekly and monthly regime cycles. A single-level Variational Autoencoder (VAE) compresses all of this multi-scale structure into a single flat latent vector, forcing the model to encode tick-level noise and macro regime information into the same representation. The result is a latent space that lacks interpretable structure and struggles to generate data that is simultaneously realistic at both fine and coarse granularities.

A **Hierarchical Variational Autoencoder (HVAE)** addresses this limitation by organizing the latent space into multiple levels, each capturing a different scale of variation. The top-level latent variables encode slow-moving, global structure (market regimes, trend direction, volatility cycles), while lower-level latent variables capture progressively finer details (intraday patterns, microstructure noise, local mean-reversion). This decomposition mirrors the natural multi-scale structure of financial time series and enables controlled generation at each level independently.

The key insight is that hierarchical latent structure enables **disentangled multi-scale generation**. A trader can fix the top-level latent to represent a "high-volatility bear market regime" while sampling different lower-level latents to generate diverse intraday patterns consistent with that regime. This is impossible with a flat VAE, where all scales are entangled in a single latent vector.

In this chapter, we develop a complete Hierarchical VAE implementation in Rust, integrate it with live market data from Bybit, and demonstrate how hierarchical latent structure improves both the quality of generated financial scenarios and the interpretability of the learned representations.

## 2. Mathematical Foundation

### The Hierarchical Generative Model

A standard VAE defines a single-level generative model:

```
p(x) = integral p(x|z) p(z) dz
```

A Hierarchical VAE with L levels defines a chain of latent variables z_1, z_2, ..., z_L where z_L is the top-level (most abstract) and z_1 is the bottom-level (most detailed):

```
p(x, z_1, ..., z_L) = p(z_L) * prod_{l=1}^{L-1} p(z_l | z_{l+1}) * p(x | z_1)
```

The generative process proceeds top-down:
1. Sample the top-level latent: `z_L ~ p(z_L) = N(0, I)`
2. For each subsequent level: `z_l ~ p(z_l | z_{l+1}) = N(mu_l(z_{l+1}), sigma_l(z_{l+1}))`
3. Generate the observation: `x ~ p(x | z_1)`

Each conditional `p(z_l | z_{l+1})` is parameterized by a neural network that maps the parent latent to the mean and variance of the child distribution. This creates a cascade where abstract information flows from the top level down to the finest level.

### The Hierarchical ELBO

The evidence lower bound for the hierarchical model decomposes as:

```
ELBO = E_q[log p(x|z_1)] - KL(q(z_L|x) || p(z_L)) - sum_{l=1}^{L-1} E_q[KL(q(z_l|z_{l+1},x) || p(z_l|z_{l+1}))]
```

This consists of:
- **Reconstruction term:** How well the bottom-level latent z_1 reconstructs the input
- **Top-level KL:** Regularizes the top-level posterior toward the standard normal prior
- **Conditional KL terms:** At each intermediate level, regularizes the encoder posterior toward the generative conditional prior

The conditional KL terms are the key innovation. At each level l, the encoder `q(z_l | z_{l+1}, x)` has access to both the parent latent and the original data, while the prior `p(z_l | z_{l+1})` only has access to the parent latent. The KL between them measures how much additional information from x the encoder uses at level l — this is exactly the information that level l encodes about the data beyond what is already captured by higher levels.

### The Inference Network (Bottom-Up + Top-Down)

The encoder in a hierarchical VAE uses a bidirectional architecture:

**Bottom-up pass:** A deterministic network processes the input x to produce features at each level:

```
h_l = f_l(h_{l-1}),  h_0 = x
```

**Top-down pass:** Starting from the top, each level combines the bottom-up features with the top-down latent to produce the posterior:

```
q(z_l | z_{l+1}, x) = N(mu_q(h_l, z_{l+1}), sigma_q(h_l, z_{l+1}))
```

This bidirectional structure ensures that each level of the posterior has access to both local features (from the bottom-up pass) and global context (from the top-down latent chain).

### The Ladder VAE Connection

The Ladder VAE (Sonderby et al., 2016) introduced a practical approach to hierarchical VAE training by combining the bottom-up and top-down computations via a precision-weighted merge:

```
mu_q = (mu_bu * sigma_td^2 + mu_td * sigma_bu^2) / (sigma_bu^2 + sigma_td^2)
sigma_q^2 = (sigma_bu^2 * sigma_td^2) / (sigma_bu^2 + sigma_td^2)
```

where `mu_bu, sigma_bu` come from the bottom-up pass and `mu_td, sigma_td` come from the top-down prior. This precision-weighted merge is optimal under Gaussian assumptions and prevents the well-known "posterior collapse" problem where higher levels are ignored during training.

### KL Annealing for Hierarchical Training

Training hierarchical VAEs is notoriously difficult because of posterior collapse — the tendency for the model to ignore higher latent levels and encode everything in z_1. Two strategies address this:

**Warmup annealing:** Gradually increase the weight of KL terms from 0 to 1 over training:

```
L = E_q[log p(x|z_1)] - beta(t) * sum_l KL_l
```

where `beta(t)` linearly increases from 0 to 1 during the first fraction of training.

**Per-level free bits:** Allow each latent dimension a minimum number of nats before incurring KL penalty:

```
KL_l_effective = max(KL_l, lambda)
```

This ensures that every level encodes at least lambda nats of information, preventing collapse.

## 3. Multi-Scale Financial Interpretation

### Level Assignment in Financial Data

A three-level HVAE for financial time series naturally maps to:

| Level | Time Scale | What It Captures | Financial Interpretation |
|-------|-----------|-------------------|------------------------|
| z_3 (top) | Weeks–Months | Macro regime, trend direction | Bull/bear regime, volatility cycle |
| z_2 (middle) | Days | Daily patterns, momentum | Sector rotation, mean-reversion speed |
| z_1 (bottom) | Intraday | Microstructure, noise | Bid-ask dynamics, tick patterns |

This decomposition is not manually imposed — the hierarchical structure naturally learns to partition information across scales because it minimizes the total description length. The top level encodes the most compressible, slowly-varying information, while lower levels capture the residual detail.

### Why Hierarchy Matters for Trading

**Scenario generation with scale control:** Fix z_3 to a "crisis regime" and sample z_2 and z_1 to generate diverse crisis scenarios that share the same macro characteristics but differ in daily and intraday details. This is far more useful for stress testing than flat VAE generation.

**Multi-scale risk decomposition:** The KL contribution at each level quantifies how much risk originates from each time scale. A trading strategy that is sensitive to z_3 perturbations is regime-dependent; one sensitive to z_1 perturbations is exposed to microstructure risk.

**Hierarchical anomaly detection:** An observation that is anomalous at the top level (unusual macro regime) requires different handling than one that is anomalous only at the bottom level (unusual tick pattern in a normal regime).

## 4. Trading Applications

### Multi-Scale Scenario Generation

The primary application is generating synthetic market data with independent control over each scale:

1. **Regime-consistent stress tests:** Fix the top-level latent to encode a specific regime (learned from historical crisis periods), then generate thousands of scenarios by sampling lower levels. Each scenario is consistent at the macro level but diverse at the daily/intraday level.

2. **Intraday pattern augmentation:** Fix the top and middle levels to represent a specific day type, then sample the bottom level to generate diverse intraday paths. This augments limited intraday datasets for training execution algorithms.

3. **Progressive refinement:** Generate a coarse scenario at the top level, inspect it, then progressively add detail by sampling lower levels. This enables interactive scenario exploration.

### Hierarchical Portfolio Risk Assessment

Decompose portfolio risk by latent level:

- **z_3 sensitivity:** How much does portfolio value change when the top-level latent shifts? This measures regime risk — the risk that the market environment changes fundamentally.
- **z_2 sensitivity:** Measures exposure to daily pattern changes — trend reversals, momentum shifts.
- **z_1 sensitivity:** Measures exposure to microstructure risk — execution slippage, intraday volatility spikes.

This decomposition helps allocate risk budgets and design hedging strategies that target specific scales.

### Multi-Resolution Forecasting

Use the hierarchical structure for forecasting at multiple horizons simultaneously:

- Top-level latent: Predict next month's regime probability
- Middle-level latent: Predict next week's return distribution
- Bottom-level latent: Predict tomorrow's intraday volatility pattern

Each level's forecast is conditioned on the higher-level predictions, ensuring consistency across horizons.

## 5. HVAE vs Flat VAE — The Multi-Scale Advantage

### Information Partitioning

A flat VAE with latent dimension d must encode all scales of variation into d dimensions with no structural guidance. In practice, the first few principal components dominate, and fine-grained structure is lost.

A hierarchical VAE with the same total latent dimension (distributed across levels) naturally partitions information by scale. This is more parameter-efficient because each level's decoder only needs to model the residual not captured by higher levels.

### Quantitative Improvements

In financial data experiments, hierarchical VAEs typically provide:

1. **Better log-likelihood:** 5-15% improvement in ELBO compared to flat VAEs with the same total latent dimension, because the hierarchical prior is more flexible.

2. **More realistic multi-scale statistics:** Generated data matches real data better at multiple time scales simultaneously — daily autocorrelations, weekly momentum patterns, and monthly regime statistics all improve.

3. **Sharper generated samples:** The bottom level captures fine detail that flat VAEs smooth over, producing generated returns with more realistic kurtosis and intraday patterns.

4. **Disentangled representations:** Latent traversals at different levels produce scale-appropriate changes — moving along z_3 changes the regime while keeping intraday patterns fixed.

### The Cost of Hierarchy

Hierarchical VAEs are harder to train (posterior collapse risk), require more careful hyperparameter tuning (per-level KL weights, annealing schedules), and have higher computational cost per forward pass. These costs are justified when the data has clear multi-scale structure — which financial time series almost always do.

## 6. Implementation Walkthrough with Rust

Our Rust implementation provides a complete three-level HVAE system:

### Core Components

The **HierarchicalEncoder** processes input data through a bottom-up network that produces features at each level. Each level has its own hidden layer that extracts features at the corresponding scale.

The **LevelPrior** at each level parameterizes the top-down conditional prior `p(z_l | z_{l+1})`. The top level uses a standard normal prior. Each subsequent level's prior is conditioned on the latent sample from the level above.

The **LevelPosterior** at each level combines bottom-up features with top-down context to produce the approximate posterior `q(z_l | z_{l+1}, x)`. This is where the precision-weighted merge occurs.

The **HierarchicalDecoder** takes the bottom-level latent z_1 and reconstructs the input. It mirrors the encoder architecture with progressive upscaling.

### Training Strategy

Our implementation uses KL warmup annealing to prevent posterior collapse:

1. For the first 30% of training, the KL weight `beta` linearly increases from 0.0 to 1.0
2. Per-level free bits (lambda = 0.1 nats) ensure all levels remain active
3. The learning rate is reduced for top-level parameters to prevent them from dominating early training

### Regime Detection and Multi-Scale Features

The implementation includes regime detection from price data and constructs multi-scale features:
- Window of normalized returns as input features
- Rolling statistics at multiple time scales for regime detection
- Automatic assignment of regime labels for evaluation

### Quality Metrics

We evaluate the hierarchical VAE using:
- **Per-level KL:** Verifies that each level is active (encoding information)
- **Multi-scale distributional match:** Wasserstein distance at different aggregation levels
- **Generated sample statistics:** Mean, variance, and kurtosis compared to real data per regime

## 7. Bybit Data Integration

Our implementation fetches real market data from the Bybit API, specifically BTCUSDT kline data:

1. **API endpoint:** We use `https://api.bybit.com/v5/market/kline` to fetch historical OHLCV data.

2. **Multi-scale feature construction:** Raw prices are converted to log returns. These returns serve as input to the HVAE, where the hierarchical structure automatically learns to decompose them into multi-scale components.

3. **Regime labeling:** The fetched price data is processed through our regime detector to produce condition labels for evaluation. Each time window is classified as bull, bear, or sideways.

4. **Evaluation pipeline:** After training, we generate scenarios at each level independently, comparing statistics against real data to verify that each level captures the appropriate scale of variation.

## 8. Key Takeaways

1. **Hierarchical VAEs organize the latent space into multiple levels**, each capturing a different scale of variation — from macro regimes at the top to microstructure details at the bottom.

2. **The hierarchical ELBO decomposes into per-level KL terms**, each measuring how much information that level encodes about the data beyond what higher levels capture.

3. **Financial markets are inherently multi-scale**, making hierarchical VAEs a natural fit. Daily trends, weekly momentum, and monthly regimes are captured by different latent levels.

4. **Multi-scale scenario generation** allows fixing top-level latents (regime) while sampling lower levels, producing regime-consistent but diverse scenarios for stress testing and risk assessment.

5. **Posterior collapse is the main training challenge** for hierarchical VAEs. KL warmup annealing and per-level free bits are essential techniques to keep all levels active during training.

6. **The Ladder VAE approach** uses precision-weighted merging of bottom-up and top-down information, providing a principled and effective way to train deep hierarchical models.

7. **Rust implementation** provides the computational efficiency needed for real-time multi-scale scenario generation, with Bybit integration enabling a complete pipeline from live data to hierarchical generation.

8. **Hierarchical risk decomposition** quantifies how much risk originates from each time scale, enabling targeted hedging strategies and more granular risk management than flat models allow.

## References

1. **Ladder Variational Autoencoders** — Sonderby et al. (2016). URL: https://arxiv.org/abs/1602.02282
2. **NVAE: A Deep Hierarchical Variational Autoencoder** — Vahdat & Koltun (2020). URL: https://arxiv.org/abs/2007.03898
3. **Variational Lossy Autoencoder** — Chen et al. (2016). URL: https://arxiv.org/abs/1611.02731
4. **Importance Weighted Autoencoders** — Burda et al. (2015). URL: https://arxiv.org/abs/1509.00519
