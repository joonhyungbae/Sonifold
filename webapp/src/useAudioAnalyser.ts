import { useEffect, useRef, useState } from "react";

const FFT_SIZE = 1024;

export default function useAudioAnalyser(enabled: boolean) {
  const [fftData, setFftData] = useState<Float32Array>(new Float32Array(FFT_SIZE));
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number>(0);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataRef = useRef<Float32Array>(new Float32Array(FFT_SIZE));

  useEffect(() => {
    if (!enabled) {
      setError(null);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (analyserRef.current) analyserRef.current = null;
      cancelAnimationFrame(rafRef.current);
      return;
    }

    setError(null);
    const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = FFT_SIZE * 2;
    analyser.smoothingTimeConstant = 0.7;
    analyserRef.current = analyser;

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        streamRef.current = stream;
        const src = audioCtx.createMediaStreamSource(stream);
        src.connect(analyser);
        const buffer = new Float32Array(FFT_SIZE);
        const minIntervalMs = 1000 / 60;
        let lastPush = 0;

        function tick() {
          if (!analyserRef.current) return;
          analyserRef.current.getFloatFrequencyData(buffer);
          dataRef.current = buffer.slice(0);
          const now = performance.now();
          if (now - lastPush >= minIntervalMs) {
            lastPush = now;
            setFftData(buffer.slice(0));
          }
          rafRef.current = requestAnimationFrame(tick);
        }
        rafRef.current = requestAnimationFrame(tick);
      })
      .catch(() => {
        setError("Microphone access denied or unavailable");
      });

    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      analyserRef.current = null;
      cancelAnimationFrame(rafRef.current);
      audioCtx.close();
    };
  }, [enabled]);

  return { fftData, error };
}
