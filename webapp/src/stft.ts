/**
 * STFT for file playback: paper params (Hann 2048, hop 512, FFT 2048 → 1024 bins).
 * Used to compute coefficients per playback time so the scalar field updates ~86 Hz.
 */

const N_FFT = 2048;
const N_BINS = 1024;

/** Radix-2 complex FFT in place. n must be power of 2. */
function fftRadix2(real: Float32Array, imag: Float32Array, inverse: boolean): void {
  const n = real.length;
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let k = n >> 1;
    while (j >= k) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const theta = (sign * Math.PI * 2) / len;
    const wlenRe = Math.cos(theta);
    const wlenIm = Math.sin(theta);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < half; j++) {
        const uRe = real[i + j];
        const uIm = imag[i + j];
        const tRe = real[i + j + half] * wRe - imag[i + j + half] * wIm;
        const tIm = real[i + j + half] * wIm + imag[i + j + half] * wRe;
        real[i + j] = uRe + tRe;
        imag[i + j] = uIm + tIm;
        real[i + j + half] = uRe - tRe;
        imag[i + j + half] = uIm - tIm;
        const nextWRe = wRe * wlenRe - wIm * wlenIm;
        const nextWIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextWRe;
        wIm = nextWIm;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] /= n;
    }
  }
}

/** Hann window of length N_FFT. */
function hannWindow(): Float32Array {
  const w = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N_FFT - 1)));
  }
  return w;
}

let cachedHann: Float32Array | null = null;
function getHann(): Float32Array {
  if (!cachedHann) cachedHann = hannWindow();
  return cachedHann;
}

/**
 * Returns 1024 magnitude bins for the STFT frame at timeSec.
 * Frame is centered at timeSec (start sample = timeSec * sampleRate - N_FFT/2, clamped).
 * Zero-pads if not enough samples.
 */
export function getMagnitudesAtTime(
  channelData: Float32Array,
  sampleRate: number,
  timeSec: number
): Float32Array {
  const startSample = Math.max(0, Math.floor(timeSec * sampleRate) - (N_FFT >> 1));
  const real = new Float32Array(N_FFT);
  const imag = new Float32Array(N_FFT);
  const hann = getHann();
  for (let i = 0; i < N_FFT; i++) {
    const src = startSample + i;
    const s = src >= 0 && src < channelData.length ? channelData[src] : 0;
    real[i] = s * hann[i];
    imag[i] = 0;
  }
  fftRadix2(real, imag, false);
  const mag = new Float32Array(N_BINS);
  for (let k = 0; k < N_BINS; k++) {
    mag[k] = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
  }
  return mag;
}
