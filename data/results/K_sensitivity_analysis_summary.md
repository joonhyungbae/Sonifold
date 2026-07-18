# K=50 vs K=100 vs K=200 results analysis summary

## 1. Data sources

- **K=50, 100, 200**: `results_genus_K100.csv`, `results_genus_K200.csv`, `K_sensitivity_genus_comparison.csv`  
  (Same pipeline: 10s audio, STFT frame-wise β₀ mean, direct mapping, EIGEN_TOL=1e-4, FRAME_STRIDE=1)

---

## 2. β₀ by K for A5 (white noise), genus order

| mesh              | genus | β₀ K=50 | β₀ K=100 | β₀ K=200 |
|-------------------|-------|---------|----------|----------|
| sphere_genus0     | 0     | 127.2   | 155.7    | 199.0    |
| torus_genus1      | 1     | **65.8**| **87.6** | **107.5**|
| double_torus_genus2 | 2   | 74.2    | 98.1     | 103.1    |
| triple_torus_genus3 | 3   | 96.9    | 109.0    | 144.1    |
| quad_torus_genus4 | 4     | 105.1   | 120.3    | 148.3    |
| penta_torus_genus5| 5     | 112.1   | 117.7    | 157.9    |
| hex_torus_genus6  | 6     | 86.9    | 145.4    | 162.6    |

---

## 3. Rank order (β₀ descending, A5)

- **K=50**:  sphere > penta > quad > triple > hex > double > **torus**
- **K=100**: sphere > hex > quad > penta > triple > double > **torus**
- **K=200**: sphere > hex > penta > quad > triple > torus > double

**Interpretation**: For all K, **torus (genus-1) is always lowest or next to lowest**. The exact rank changes somewhat with K (e.g. hex moves up with higher K), but the **qualitative pattern “torus has low β₀” holds at K=50, 100, and 200**.

---

## 4. Genus–β₀ monotonicity (A5)

- **K=50**:  β₀ does not increase monotonically with genus (e.g. genus 6 β₀ < genus 5).
- **K=100**: Not monotonic (genus 5 < genus 4, etc.).
- **K=200**: Not monotonic (genus 2 < genus 1, etc.).

**Conclusion**: **At all K, the genus–β₀ relation is non-monotonic.** The phenomenon that “β₀ is not determined monotonically by genus alone” (Conjecture 4.1) also appears at K=100 and K=200.

---

## 5. Torus anomaly (A1/A2)

| stimulus | K   | torus β₀ | double_torus β₀ | sphere β₀ |
|----------|-----|----------|------------------|-----------|
| A1       | 100 | **0.0**  | 0.0              | 58.0      |
| A1       | 200 | **1.0**  | 1.0              | 81.4      |
| A2       | 100 | **0.12** | 0.09             | 56.1      |
| A2       | 200 | **2.36** | 0.80             | 38.4      |

**Interpretation**: For A1 (440Hz pure tone) and A2 (chord), **torus (and double torus) β₀ at K=100, 200 remains near 0 or very small.** The **anomaly “torus has extremely low β₀ for low-frequency simple stimuli” persists** when more eigenmodes are used.

---

## 6. Paper / verification summary

1. **K sensitivity**: The **non-monotonicity** and **torus anomaly** seen at K=50 are reproduced at K=100 and K=200.  
   → Can be used as evidence that **qualitative conclusions are unchanged with more eigenmodes**, countering “K=50 truncation artifact” concerns.

2. **Rank order**: Detailed ranks change with K, but **torus is always in the lower part**.  
   → The statement “genus-1 has especially low β₀” is supported at K=50 as well as K=100 and K=200.

3. **Numerical scale**: Mean β₀ generally increases with K (more eigenmodes).  
   → For the paper, it is safer to focus on **relative genus order, non-monotonicity, and torus anomaly** rather than absolute values.

---

## 7. One-line summary

**The “genus–β₀ non-monotonicity” and “extremely low β₀ for torus (and double torus) under A1/A2” observed at K=50 also appear at K=100 and K=200, and can be used as verification of Conjecture 4.1 with respect to K-sensitivity.**
