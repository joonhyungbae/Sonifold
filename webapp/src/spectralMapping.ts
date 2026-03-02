const N_FFT_BINS = 1024;
const N = 50;

function l2Norm(a: number[]): number[] {
  const n = Math.sqrt(a.reduce((s, x) => s + x * x, 0)) || 1;
  return a.map((x) => x / n);
}

export function mapDirect(fftMagnitudes: Float32Array | number[]): number[] {
  const m = Array.from(fftMagnitudes).slice(0, N_FFT_BINS);
  while (m.length < N_FFT_BINS) m.push(0);
  const step = N_FFT_BINS / N;
  const coef: number[] = [];
  for (let i = 0; i < N; i++) {
    let s = 0;
    const start = Math.floor(i * step);
    const end = Math.floor((i + 1) * step);
    for (let j = start; j < end && j < m.length; j++) s += m[j];
    coef.push(s / (end - start || 1));
  }
  return l2Norm(coef);
}

function melBandEdges(n: number): number[] {
  const low = 0;
  const high = 2595 * Math.log10(1 + (N_FFT_BINS - 1) / 700);
  const pts: number[] = [];
  for (let i = 0; i <= n; i++) {
    const mel = low + (high - low) * (i / n);
    pts.push(700 * (Math.pow(10, mel / 2595) - 1));
  }
  const scale = (N_FFT_BINS - 1) / (pts[n] || 1);
  return pts.map((p) => Math.min(p * scale, N_FFT_BINS));
}

export function mapMel(fftMagnitudes: Float32Array | number[]): number[] {
  const m = Array.from(fftMagnitudes).slice(0, N_FFT_BINS);
  while (m.length < N_FFT_BINS) m.push(0);
  const edges = melBandEdges(N);
  const coef: number[] = [];
  for (let i = 0; i < N; i++) {
    const lo = Math.floor(edges[i]);
    const hi = Math.min(Math.floor(edges[i + 1]), N_FFT_BINS);
    let s = 0;
    let count = 0;
    for (let j = lo; j < hi; j++) {
      s += m[j];
      count++;
    }
    coef.push(count ? s / count : 0);
  }
  return l2Norm(coef);
}

/** Paper: "Each coefficient is the total energy (sum of squared magnitudes) in the corresponding mel band." */
export function mapEnergy(
  fftMagnitudes: Float32Array | number[],
  _eigenvalues?: number[]
): number[] {
  const m = Array.from(fftMagnitudes).slice(0, N_FFT_BINS);
  while (m.length < N_FFT_BINS) m.push(0);
  const edges = melBandEdges(N);
  const coef: number[] = [];
  for (let i = 0; i < N; i++) {
    const lo = Math.floor(edges[i]);
    const hi = Math.min(Math.floor(edges[i + 1]), N_FFT_BINS);
    let energy = 0;
    for (let j = lo; j < hi && j < m.length; j++) {
      const x = m[j];
      energy += x * x;
    }
    coef.push(energy);
  }
  return l2Norm(coef);
}

export type MappingStrategy = "direct" | "mel" | "energy";

export function mapFftToCoefficients(
  fftMagnitudes: Float32Array | number[],
  strategy: MappingStrategy,
  eigenvalues?: number[]
): number[] {
  if (strategy === "direct") return mapDirect(fftMagnitudes);
  if (strategy === "mel") return mapMel(fftMagnitudes);
  return mapEnergy(fftMagnitudes, eigenvalues);
}
