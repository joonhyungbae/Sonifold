# Temporal autocorrelation of β₀ time series

Overlapping STFT windows (hop=512, window=2048) induce temporal autocorrelation.
Effective sample size: n_eff = n × (1 − r₁) / (1 + r₁).

## Per-condition results

### Sphere × A3
- n_frames = 858, n_effective = 169.5
- Lag-1 autocorrelation r₁ = 0.6701
- mean(β₀) = 75.5408, std(β₀) = 8.9617
- Naive SE = 0.305947, Corrected SE = 0.688393
- Shapiro-Wilk W = 0.9831, p = 1.4744e-05

### Torus × A3
- n_frames = 858, n_effective = 49.2
- Lag-1 autocorrelation r₁ = 0.8916
- mean(β₀) = 51.1935, std(β₀) = 13.1723
- Naive SE = 0.449695, Corrected SE = 1.878436
- Shapiro-Wilk W = 0.9021, p = 2.2120e-17

### Double torus × A3
- n_frames = 858, n_effective = 42.5
- Lag-1 autocorrelation r₁ = 0.9057
- mean(β₀) = 40.3124, std(β₀) = 14.5290
- Naive SE = 0.496013, Corrected SE = 2.229615
- Shapiro-Wilk W = 0.9395, p = 2.1954e-13

### Sphere × A7
- n_frames = 858, n_effective = 45.6
- Lag-1 autocorrelation r₁ = 0.8991
- mean(β₀) = 40.0793, std(β₀) = 11.0325
- Naive SE = 0.376643, Corrected SE = 1.634015
- Shapiro-Wilk W = 0.8557, p = 5.1400e-21

### Torus × A7
- n_frames = 858, n_effective = 46.2
- Lag-1 autocorrelation r₁ = 0.8977
- mean(β₀) = 4.3007, std(β₀) = 4.5305
- Naive SE = 0.154669, Corrected SE = 0.666321
- Shapiro-Wilk W = 0.8377, p = 3.4672e-22
