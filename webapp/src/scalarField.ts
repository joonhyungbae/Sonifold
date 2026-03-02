/**
 * Scalar field: f(v) = Σ c_k · φ_k(v) = coefficients @ eigenvectors
 */
export function computeScalarField(
  eigenvectors: number[][],
  coefficients: number[]
): number[] {
  const N = Math.min(eigenvectors.length, coefficients.length);
  const V = eigenvectors[0]?.length ?? 0;
  const out: number[] = new Array(V).fill(0);
  for (let v = 0; v < V; v++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += coefficients[i] * (eigenvectors[i][v] ?? 0);
    out[v] = s;
  }
  return out;
}
