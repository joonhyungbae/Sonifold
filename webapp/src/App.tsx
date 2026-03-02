import { useState, useEffect, useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { MeshScene } from "./MeshScene";
import { AppBarDesktop, AppBarMobile } from "./AppBar";
import useAudioAnalyser from "./useAudioAnalyser";
import { mapFftToCoefficients, type MappingStrategy } from "./spectralMapping";
import { getMagnitudesAtTime } from "./stft";
import { computeScalarField } from "./scalarField";
import { computeTopologyMetrics, type TopologyMetrics } from "./topologyMetrics";
import type { MeshData } from "./types";

interface ExperimentSummary {
  h1: { rho: number; p: number };
  h2: { p: number; high_mean: number; low_mean: number };
  h3: { r2: number; p: number };
}
interface ExperimentRow {
  mesh: string;
  audio: string;
  strategy: string;
  beta0: number;
  beta1: number;
  chi: number;
  A_ratio: number;
  S: number;
}
interface ExperimentData {
  summary: ExperimentSummary;
  rows: ExperimentRow[];
}

function formatNum(x: number): string {
  return Number.isFinite(x) ? x.toFixed(3) : "—";
}

function formatTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

const PRECOMPUTE_FPS = 60;
const PRECOMPUTE_CHUNK = 90;

const MOBILE_BREAKPOINT = 768;

function getBaseCameraPosition(meshName: string): [number, number, number] {
  return meshName === "torus"
    ? [6.2, 4.13, 6.2]
    : meshName === "double_torus"
      ? [4.2, 2.8, 4.2]
      : meshName === "cube"
        ? [1.3, 0.87, 1.3]
        : meshName === "ellipsoid"
          ? [3.3, 2.2, 3.3]
          : meshName === "flat_plate"
            ? [0.85, 0.57, 0.85]
            : meshName === "tetrahedron"
              ? [1.2, 0.8, 1.2]
              : [1.65, 1.1, 1.65];
}

function CameraDistance({
  cameraPosition,
}: {
  cameraPosition: [number, number, number];
}) {
  const { camera } = useThree();
  useEffect(() => {
    const [x, y, z] = cameraPosition;
    camera.position.set(x, y, z);
  }, [cameraPosition, camera]);
  return null;
}

const MESH_OPTIONS: { id: string; label: string }[] = [
  { id: "sphere", label: "Sphere" },
  { id: "torus", label: "Torus" },
  { id: "cube", label: "Cube" },
  { id: "ellipsoid", label: "Ellipsoid" },
  { id: "double_torus", label: "Double torus" },
  { id: "flat_plate", label: "Flat plate" },
  { id: "tetrahedron", label: "Tetrahedron" },
  { id: "octahedron", label: "Octahedron" },
  { id: "icosahedron", label: "Icosahedron" },
];

export default function App() {
  const [meshName, setMeshName] = useState("sphere");
  const [isMobile, setIsMobile] = useState(
    typeof window !== "undefined" ? window.innerWidth <= MOBILE_BREAKPOINT : false
  );
  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth <= MOBILE_BREAKPOINT);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const cameraPosition = useMemo(() => {
    const [x, y, z] = getBaseCameraPosition(meshName);
    const scale = isMobile ? 2 : 1;
    return [x * scale, y * scale, z * scale] as [number, number, number];
  }, [meshName, isMobile]);
  const [meshData, setMeshData] = useState<MeshData | null>(null);
  const [meshLoading, setMeshLoading] = useState(false);
  const [meshError, setMeshError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<MappingStrategy>("energy");
  const [micOn, setMicOn] = useState(false);
  const [highlightNodal] = useState(true);
  const [contourLevels, setContourLevels] = useState(16);
  const [coefficients, setCoefficients] = useState<number[]>(() =>
    Array(50).fill(0)
  );
  const AUDIO_STIMULI = ["mic", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "upload"] as const;
  type AudioStimulus = (typeof AUDIO_STIMULI)[number];
  const [audioStimulus, setAudioStimulus] = useState<AudioStimulus>("A4");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedObjectUrl, setUploadedObjectUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const decodedBufferRef = useRef<AudioBuffer | null>(null);
  const lastStftPlayTimeRef = useRef<number>(-1);
  const playRafRef = useRef<number>(0);
  const meshDataRef = useRef<MeshData | null>(null);
  const strategyRef = useRef<MappingStrategy>("energy");
  const precomputedFramesRef = useRef<Float32Array | null>(null);
  const precomputedNumFramesRef = useRef(0);
  const precomputedScalarRef = useRef<Float32Array | null>(null);
  const precomputedUniformsRef = useRef<Float32Array | null>(null);
  const precomputedNvRef = useRef(0);
  const displayScalarRef = useRef<Float32Array | null>(null);
  const displayUniformsRef = useRef<{ minS: number; range: number; maxAbs: number }>({ minS: 0, range: 1, maxAbs: 1 });
  const [precomputeStatus, setPrecomputeStatus] = useState<"idle" | "running" | "done">("idle");
  const [decodedBufferReady, setDecodedBufferReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playTime, setPlayTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [audioLoadError, setAudioLoadError] = useState<string | null>(null);

  const { fftData, error: micError } = useAudioAnalyser(audioStimulus === "mic");

  useEffect(() => {
    if (audioStimulus === "mic") setMicOn(true);
  }, [audioStimulus]);

  useEffect(() => {
    let cancelled = false;
    setMeshLoading(true);
    setMeshError(null);
    fetch(`data/eigen/${meshName}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Not found"))))
      .then((data: MeshData) => {
        if (!cancelled) {
          setMeshData(data);
          setMeshError(null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setMeshData(null);
          setMeshError("Failed to load mesh");
        }
      })
      .finally(() => {
        if (!cancelled) setMeshLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [meshName]);

  meshDataRef.current = meshData;
  strategyRef.current = strategy;

  const SMOOTH_ALPHA = 0.35;

  useEffect(() => {
    if (audioStimulus === "mic") return;
    const precomputedScalar = precomputedScalarRef.current;
    const precomputedUniforms = precomputedUniformsRef.current;
    const numFrames = precomputedNumFramesRef.current;
    const nV = precomputedNvRef.current;
    if (!isPlaying || !precomputedScalar || !precomputedUniforms || numFrames === 0 || nV === 0) return;
    const displayScalar = displayScalarRef.current;
    if (!displayScalar || displayScalar.length !== nV) return;
    const scalar = precomputedScalar;
    const uniforms = precomputedUniforms;
    const display = displayScalar;
    let running = true;
    const SMOOTH = SMOOTH_ALPHA;
    function tick() {
      if (!running || !audioRef.current) {
        playRafRef.current = requestAnimationFrame(tick);
        return;
      }
      const t = audioRef.current.currentTime;
      const frameF = t * PRECOMPUTE_FPS;
      const i0 = Math.min(Math.floor(frameF), numFrames - 1);
      const i1 = Math.min(i0 + 1, numFrames - 1);
      const frac = Math.max(0, Math.min(1, frameF - i0));
      const off0 = i0 * nV;
      const off1 = i1 * nV;
      const u0 = i0 * 3;
      const u1 = i1 * 3;
      for (let v = 0; v < nV; v++) {
        const raw = (1 - frac) * scalar[off0 + v] + frac * scalar[off1 + v];
        display[v] = SMOOTH * raw + (1 - SMOOTH) * display[v];
      }
      displayUniformsRef.current.minS = (1 - frac) * uniforms[u0] + frac * uniforms[u1];
      displayUniformsRef.current.range = (1 - frac) * uniforms[u0 + 1] + frac * uniforms[u1 + 1];
      displayUniformsRef.current.maxAbs = (1 - frac) * uniforms[u0 + 2] + frac * uniforms[u1 + 2];
      playRafRef.current = requestAnimationFrame(tick);
    }
    playRafRef.current = requestAnimationFrame(tick);
    return () => {
      running = false;
      cancelAnimationFrame(playRafRef.current);
    };
  }, [audioStimulus, isPlaying, precomputeStatus]);

  useEffect(() => {
    if (audioStimulus !== "mic") return;
    const ev = meshData?.eigenvalues;
    const coef = mapFftToCoefficients(fftData, strategy, ev);
    setCoefficients(coef);
  }, [audioStimulus, fftData, strategy, meshData?.eigenvalues]);

  useEffect(() => {
    if (audioStimulus === "mic") return;
    setIsPlaying(false);
    setPlayTime(0);
    setDuration(0);
    setAudioLoadError(null);
    decodedBufferRef.current = null;
    lastStftPlayTimeRef.current = -1;
    precomputedFramesRef.current = null;
    precomputedNumFramesRef.current = 0;
    precomputedScalarRef.current = null;
    precomputedUniformsRef.current = null;
    precomputedNvRef.current = 0;
    displayScalarRef.current = null;
    setPrecomputeStatus("idle");
    setDecodedBufferReady(false);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [audioStimulus]);

  useEffect(() => {
    if (audioStimulus === "mic") return;
    const buf = decodedBufferRef.current;
    const mesh = meshData;
    if (!buf || !mesh?.eigenvalues || !mesh?.eigenvectors?.length) {
      precomputedFramesRef.current = null;
      precomputedNumFramesRef.current = 0;
      precomputedScalarRef.current = null;
      precomputedUniformsRef.current = null;
      precomputedNvRef.current = 0;
      displayScalarRef.current = null;
      setPrecomputeStatus("idle");
      setIsPlaying(false);
      return;
    }
    setIsPlaying(false);
    const durationSec = buf.duration;
    const numFrames = Math.ceil(durationSec * PRECOMPUTE_FPS) || 1;
    const nV = mesh.vertices.length;
    const arr = new Float32Array(numFrames * 50);
    const scalarArr = new Float32Array(numFrames * nV);
    const uniformsArr = new Float32Array(numFrames * 3);
    const ch = buf.getChannelData(0);
    const sr = buf.sampleRate;
    const ev = mesh.eigenvalues;
    const eigenvectors = mesh.eigenvectors;
    setPrecomputeStatus("running");
    let frameIndex = 0;
    let cancelled = false;
    function doChunk() {
      if (cancelled) return;
      const end = Math.min(frameIndex + PRECOMPUTE_CHUNK, numFrames);
      for (; frameIndex < end; frameIndex++) {
        const t = frameIndex / PRECOMPUTE_FPS;
        const mag = getMagnitudesAtTime(ch, sr, t);
        const coef = mapFftToCoefficients(mag, strategy, ev);
        const base = frameIndex * 50;
        for (let k = 0; k < 50; k++) arr[base + k] = coef[k];
        const scalar = computeScalarField(eigenvectors, coef);
        let minS = Infinity, maxS = -Infinity;
        for (let v = 0; v < nV; v++) {
          const s = scalar[v];
          scalarArr[frameIndex * nV + v] = s;
          if (s < minS) minS = s;
          if (s > maxS) maxS = s;
        }
        const range = Math.max(maxS - minS, 1e-6);
        let maxAbs = 0;
        for (let v = 0; v < nV; v++) {
          const a = Math.abs(scalar[v]);
          if (a > maxAbs) maxAbs = a;
        }
        maxAbs = Math.max(maxAbs, 1e-10);
        uniformsArr[frameIndex * 3] = minS;
        uniformsArr[frameIndex * 3 + 1] = range;
        uniformsArr[frameIndex * 3 + 2] = maxAbs;
      }
      if (frameIndex < numFrames) {
        requestAnimationFrame(doChunk);
      } else {
        if (!cancelled) {
          precomputedFramesRef.current = arr;
          precomputedNumFramesRef.current = numFrames;
          precomputedScalarRef.current = scalarArr;
          precomputedUniformsRef.current = uniformsArr;
          precomputedNvRef.current = nV;
          displayScalarRef.current = new Float32Array(nV);
          displayScalarRef.current.set(scalarArr.subarray(0, nV));
          displayUniformsRef.current = {
            minS: uniformsArr[0],
            range: uniformsArr[1],
            maxAbs: uniformsArr[2],
          };
          setPrecomputeStatus("done");
          setCoefficients(Array.from(arr.subarray(0, 50)));
        }
      }
    }
    requestAnimationFrame(doChunk);
    return () => {
      cancelled = true;
      precomputedFramesRef.current = null;
      precomputedNumFramesRef.current = 0;
      precomputedScalarRef.current = null;
      precomputedUniformsRef.current = null;
      precomputedNvRef.current = 0;
      displayScalarRef.current = null;
      setPrecomputeStatus("idle");
      setIsPlaying(false);
    };
  }, [audioStimulus, decodedBufferReady, meshData?.eigenvalues, meshData?.eigenvectors, meshData?.vertices, strategy]);

  useEffect(() => {
    if (audioStimulus === "mic" || audioStimulus === "upload") return;
    let cancelled = false;
    fetch(`audio/${audioStimulus}.wav`)
      .then((r) => r.arrayBuffer())
      .then((ab) => {
        if (cancelled) return;
        return new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)().decodeAudioData(ab);
      })
      .then((buffer) => {
        if (!cancelled && buffer) {
          decodedBufferRef.current = buffer;
          setDecodedBufferReady(true);
        }
      })
      .catch(() => {
        if (!cancelled) decodedBufferRef.current = null;
      });
    return () => {
      cancelled = true;
      decodedBufferRef.current = null;
      setDecodedBufferReady(false);
    };
  }, [audioStimulus]);

  useEffect(() => {
    if (audioStimulus !== "upload") return;
    setPlayTime(0);
    setDuration(0);
  }, [audioStimulus, uploadedObjectUrl]);

  useEffect(() => {
    if (audioStimulus !== "upload" || !uploadedFile) return;
    let cancelled = false;
    const ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    uploadedFile.arrayBuffer().then((ab) => {
      if (cancelled) return;
      return ctx.decodeAudioData(ab);
    }).then((buffer) => {
      if (!cancelled && buffer) {
        decodedBufferRef.current = buffer;
        setDecodedBufferReady(true);
      }
    }).catch(() => {
      if (!cancelled) decodedBufferRef.current = null;
    });
    return () => {
      cancelled = true;
      decodedBufferRef.current = null;
      setDecodedBufferReady(false);
    };
  }, [audioStimulus, uploadedFile]);

  useEffect(() => {
    if (audioStimulus !== "upload") {
      if (uploadedObjectUrl) {
        URL.revokeObjectURL(uploadedObjectUrl);
        setUploadedObjectUrl(null);
      }
      setUploadedFile(null);
    }
  }, [audioStimulus]);

  useEffect(() => {
    if (audioStimulus === "mic") return;
    if (isPlaying) return;
    const precomputedScalar = precomputedScalarRef.current;
    const precomputedUniforms = precomputedUniformsRef.current;
    const precomputed = precomputedFramesRef.current;
    const numFrames = precomputedNumFramesRef.current;
    const nV = precomputedNvRef.current;
    if (precomputedScalar && precomputedUniforms && numFrames > 0 && nV > 0) {
      const frameIndex = Math.min(Math.max(0, Math.floor(playTime * PRECOMPUTE_FPS)), numFrames - 1);
      const displayScalar = displayScalarRef.current;
      if (displayScalar && displayScalar.length === nV) {
        displayScalar.set(precomputedScalar.subarray(frameIndex * nV, frameIndex * nV + nV));
        displayUniformsRef.current.minS = precomputedUniforms[frameIndex * 3];
        displayUniformsRef.current.range = precomputedUniforms[frameIndex * 3 + 1];
        displayUniformsRef.current.maxAbs = precomputedUniforms[frameIndex * 3 + 2];
      }
      if (precomputed) {
        const start = frameIndex * 50;
        setCoefficients(Array.from(precomputed.subarray(start, start + 50)));
      }
      return;
    }
    if (precomputed && numFrames > 0) {
      const frameIndex = Math.min(Math.floor(playTime * PRECOMPUTE_FPS), numFrames - 1);
      const start = Math.max(0, frameIndex) * 50;
      setCoefficients(Array.from(precomputed.subarray(start, start + 50)));
      return;
    }
    const buf = decodedBufferRef.current;
    if (buf && meshData?.eigenvalues) {
      const ch = buf.getChannelData(0);
      const mag = getMagnitudesAtTime(ch, buf.sampleRate, playTime);
      const coef = mapFftToCoefficients(mag, strategy, meshData.eigenvalues);
      setCoefficients(coef);
      return;
    }
    if (audioStimulus === "upload") return;
    lastStftPlayTimeRef.current = -1;
    let cancelled = false;
    fetch(`data/coefficients/${meshName}_${audioStimulus}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Not found"))))
      .then((data: Record<string, number[]>) => {
        if (!cancelled && data[strategy]) setCoefficients(data[strategy]);
      })
      .catch(() => {
        if (!cancelled) setCoefficients(Array(50).fill(0));
      });
    return () => { cancelled = true; };
  }, [audioStimulus, meshName, strategy, playTime, decodedBufferReady, meshData?.eigenvalues, isPlaying]);

  const scalar = useMemo(() => {
    if (!meshData?.eigenvectors?.length || !coefficients.length) return [];
    return computeScalarField(meshData.eigenvectors, coefficients);
  }, [meshData?.eigenvectors, coefficients]);

  const metrics: TopologyMetrics | null = useMemo(() => {
    if (!meshData?.vertices?.length || scalar.length !== meshData.vertices.length) return null;
    return computeTopologyMetrics(meshData.vertices, meshData.faces, scalar);
  }, [meshData?.vertices, meshData?.faces, scalar]);

  const [experimentData, setExperimentData] = useState<ExperimentData | null | "loading">("loading");
  useEffect(() => {
    let cancelled = false;
    fetch("data/experiment.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Not found"))))
      .then((data: ExperimentData) => {
        if (!cancelled) setExperimentData(data);
      })
      .catch(() => {
        if (!cancelled) setExperimentData(null);
      });
    return () => { cancelled = true; };
  }, []);

  const experimentRowsForMesh = useMemo(() => {
    if (!experimentData || experimentData === "loading" || !experimentData.rows) return [];
    return experimentData.rows.filter((r) => r.mesh === meshName);
  }, [experimentData, meshName]);

  const AUDIO_IDS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"];
  const [experimentView, setExperimentView] = useState<"mesh" | "audio">("mesh");
  const [selectedAudio, setSelectedAudio] = useState("A1");
  const experimentRowsForAudio = useMemo(() => {
    if (!experimentData || experimentData === "loading" || !experimentData.rows) return [];
    return experimentData.rows.filter((r) => r.audio === selectedAudio);
  }, [experimentData, selectedAudio]);

  const experimentTableRows = experimentView === "mesh" ? experimentRowsForMesh : experimentRowsForAudio;
  const experimentTableCaption =
    experimentView === "mesh"
      ? `This mesh (${meshName}) × audio × strategy`
      : `This audio (${selectedAudio}) × mesh × strategy`;

  const matchingExperimentRow = useMemo(() => {
    if (!experimentData || experimentData === "loading" || !experimentData.rows) return null;
    return experimentData.rows.find(
      (r) => r.mesh === meshName && r.audio === audioStimulus && r.strategy === strategy
    ) ?? null;
  }, [experimentData, meshName, audioStimulus, strategy]);
  const symmetryIndexS = matchingExperimentRow != null ? matchingExperimentRow.S : null;

  const appBarProps = {
    meshLoading,
    meshError,
    micOn,
    micError,
    meshName,
    setMeshName,
    meshOptions: MESH_OPTIONS,
    audioStimulus,
    setAudioStimulus: (v: string) => setAudioStimulus(v as AudioStimulus),
    audioStimuli: AUDIO_STIMULI,
    setMicOn,
    strategy,
    setStrategy,
    contourLevels,
    setContourLevels,
    onUploadFileChange: (file: File | null) => {
      setUploadedFile(file);
      setUploadedObjectUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return file ? URL.createObjectURL(file) : null;
      });
    },
    uploadedFileName: uploadedFile?.name ?? null,
  };

  return (
    <>
      {isMobile ? (
        <AppBarMobile {...appBarProps} />
      ) : (
        <AppBarDesktop {...appBarProps} />
      )}

      {audioStimulus !== "mic" && (audioStimulus !== "upload" || uploadedObjectUrl) && (
        <audio
          key={audioStimulus === "upload" ? uploadedObjectUrl ?? "upload-pending" : audioStimulus}
          ref={audioRef}
          src={audioStimulus === "upload" ? uploadedObjectUrl! : `audio/${audioStimulus}.wav`}
          preload="auto"
          onLoadedMetadata={() => {
            const el = audioRef.current;
            if (!el) return;
            const d = el.duration;
            if (Number.isFinite(d) && d >= 0) {
              setDuration(d);
              setPlayTime(0);
            }
            setAudioLoadError(null);
          }}
          onDurationChange={() => {
            const el = audioRef.current;
            if (!el) return;
            const d = el.duration;
            if (Number.isFinite(d) && d >= 0) setDuration(d);
          }}
          onError={() => setAudioLoadError("Audio file not found. Run: python -m experiment.export_for_web")}
          onTimeUpdate={() => {
            const el = audioRef.current;
            if (el) setPlayTime(el.currentTime);
          }}
          onEnded={() => setIsPlaying(false)}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        />
      )}
      <aside className="panel panel--sidebar" aria-label="Sidebar">
        <section className="sidebar-section" aria-label="Topology metrics">
          <h2 className="sidebar-section__title">Topology metrics</h2>
          {metrics ? (
            <table className="metrics-table" aria-label="Topology metrics">
              <tbody>
                <tr>
                  <td>β₀</td>
                  <td>{metrics.beta0}</td>
                </tr>
                <tr>
                  <td>β₁</td>
                  <td>{metrics.beta1}</td>
                </tr>
                <tr>
                  <td>χ</td>
                  <td>{metrics.chi}</td>
                </tr>
                <tr>
                  <td>A_ratio</td>
                  <td>{metrics.A_ratio.toFixed(4)}</td>
                </tr>
                <tr>
                  <td>S</td>
                  <td>{symmetryIndexS != null ? symmetryIndexS.toFixed(4) : "—"}</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <span className="metrics-empty">—</span>
          )}
        </section>
        <section className="sidebar-section" aria-label="Experiment results">
          <h2 className="sidebar-section__title">Experiment results</h2>
          {experimentData === "loading" && (
            <span className="panel-message panel-message--loading">Loading…</span>
          )}
          {experimentData === null && (
            <p className="experiment-unavailable">
              Not available. Run <code>./step4.sh</code> then{" "}
              <code>python -m experiment.export_for_web</code> and reload.
            </p>
          )}
          {experimentData && experimentData !== "loading" && (
            <>
              <div className="experiment-summary">
                <div><strong>H1</strong> (spectral gap vs bandwidth): ρ={formatNum(experimentData.summary.h1.rho)}, p={formatNum(experimentData.summary.h1.p)}</div>
                <div><strong>H2</strong> (symmetric vs asymmetric S): p={formatNum(experimentData.summary.h2.p)}</div>
                <div><strong>H3</strong> (Weyl vs β₀): R²={formatNum(experimentData.summary.h3.r2)}, p={formatNum(experimentData.summary.h3.p)}</div>
              </div>
              <div className="experiment-view-controls">
                <span className="experiment-view-label">View:</span>
                <button
                  type="button"
                  className={"experiment-view-btn" + (experimentView === "mesh" ? " experiment-view-btn--active" : "")}
                  onClick={() => setExperimentView("mesh")}
                  aria-pressed={experimentView === "mesh"}
                >
                  By mesh
                </button>
                <button
                  type="button"
                  className={"experiment-view-btn" + (experimentView === "audio" ? " experiment-view-btn--active" : "")}
                  onClick={() => setExperimentView("audio")}
                  aria-pressed={experimentView === "audio"}
                >
                  By audio
                </button>
                {experimentView === "audio" && (
                  <select
                    className="select experiment-audio-select"
                    value={selectedAudio}
                    onChange={(e) => setSelectedAudio(e.target.value)}
                    aria-label="Select audio"
                  >
                    {AUDIO_IDS.map((a) => (
                      <option key={a} value={a}>{a}</option>
                    ))}
                  </select>
                )}
              </div>
              {experimentTableRows.length > 0 && (
                <div className="experiment-table-wrap">
                  <div className="experiment-table-caption">{experimentTableCaption}</div>
                  <table className="metrics-table experiment-table">
                    <thead>
                      <tr>
                        {experimentView === "mesh" ? <th>audio</th> : <th>mesh</th>}
                        <th>strategy</th>
                        <th>β₀</th>
                        <th>A_ratio</th>
                        <th>S</th>
                      </tr>
                    </thead>
                    <tbody>
                      {experimentTableRows.map((r, i) => (
                        <tr key={experimentView === "mesh" ? `${r.audio}-${r.strategy}-${i}` : `${r.mesh}-${r.strategy}-${i}`}>
                          <td>{experimentView === "mesh" ? r.audio : r.mesh}</td>
                          <td>{r.strategy}</td>
                          <td>{r.beta0}</td>
                          <td>{r.A_ratio.toFixed(4)}</td>
                          <td>{r.S.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </section>
      </aside>

      <aside className="panel panel--player" aria-label="Audio player">
        <div className="player-bar">
          {audioStimulus !== "mic" && (audioStimulus !== "upload" || uploadedObjectUrl) && (
            <div className="player-controls">
              {audioLoadError && (
                <span className="player-bar__error" role="alert">{audioLoadError}</span>
              )}
              <button
                type="button"
                className="player-btn player-btn--icon"
                disabled={precomputeStatus !== "done"}
                onClick={() => {
                  if (precomputeStatus !== "done") return;
                  const el = audioRef.current;
                  if (!el) return;
                  if (isPlaying) {
                    el.pause();
                    return;
                  }
                  el.play().catch(() => setAudioLoadError("Playback failed. Try reloading or run: python -m experiment.export_for_web"));
                }}
                aria-label={isPlaying ? "Pause" : "Play"}
                title={precomputeStatus === "running" ? "Preparing…" : (isPlaying ? "Pause" : "Play")}
              >
                {isPlaying ? (
                  <span className="player-btn__icon" aria-hidden>⏸</span>
                ) : (
                  <span className="player-btn__icon player-btn__icon--play" aria-hidden>▶</span>
                )}
              </button>
              <span className="player-time player-time--current" aria-live="polite">
                {formatTime(playTime)}
              </span>
              <div className="player-progress-wrap">
                <div
                  className="player-progress-fill"
                  style={{ width: duration ? `${(playTime / duration) * 100}%` : "0%" }}
                />
                <input
                  type="range"
                  className="player-progress"
                  min={0}
                  max={duration || 1}
                  step={0.01}
                  value={playTime}
                  onChange={(e) => {
                    const t = Number(e.target.value);
                    setPlayTime(t);
                    if (audioRef.current) audioRef.current.currentTime = t;
                  }}
                  aria-label="Seek"
                />
              </div>
              <span className="player-time player-time--total">
                {formatTime(duration)}
              </span>
            </div>
          )}
          {audioStimulus === "mic" && (
            <span className="player-bar__live">Live mic</span>
          )}
          {audioStimulus === "upload" && !uploadedObjectUrl && (
            <span className="player-bar__upload-hint">Select an audio file using the Upload button above.</span>
          )}
        </div>
      </aside>

      {precomputeStatus === "running" && (
        <div className="preparing-modal" role="dialog" aria-modal="true" aria-live="polite" aria-label="Preparing audio">
          <div className="preparing-modal__backdrop" aria-hidden />
          <div className="preparing-modal__card">
            <div className="preparing-modal__spinner" aria-hidden />
            <p className="preparing-modal__text">Preparing…</p>
            <p className="preparing-modal__hint">Computing mesh response to audio</p>
          </div>
        </div>
      )}

      <Canvas
        camera={{ position: cameraPosition, fov: 55 }}
        gl={{
          antialias: true,
          pixelRatio: Math.min(2, typeof window !== "undefined" ? window.devicePixelRatio : 1),
        }}
        style={{ width: "100%", height: "100%" }}
      >
        <color attach="background" args={["#0a0a0f"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        <CameraDistance cameraPosition={cameraPosition} />
        <MeshScene
          meshName={meshName}
          meshData={meshData}
          coefficients={coefficients}
          highlightNodal={highlightNodal}
          contourLevels={contourLevels}
          usePrecomputedScalar={audioStimulus !== "mic" && !!precomputedScalarRef.current}
          displayScalarRef={displayScalarRef}
          displayUniformsRef={displayUniformsRef}
        />
        <OrbitControls makeDefault />
      </Canvas>
    </>
  );
}
