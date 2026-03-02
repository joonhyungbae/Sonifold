import { useRef, useMemo, useLayoutEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { MeshData } from "./types";

/** Paper: ε = 0.05 × max|f| for nodal set. Applied in fragment via uMaxAbs. */
const NODAL_THRESHOLD_RATIO = 0.05;

const vertexShader = `
  attribute float scalar;
  uniform float uMinS;
  uniform float uRange;
  varying float vNorm;
  varying vec3 vNormal;
  void main() {
    vNorm = (uRange > 0.0) ? ((scalar - uMinS) / uRange * 2.0 - 1.0) : 0.0;
    vNormal = normalMatrix * normal;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = `
  precision highp float;
  uniform float uHighlightNodal;
  uniform float uContourCount;
  uniform float uMinS;
  uniform float uRange;
  uniform float uMaxAbs;
  uniform vec3 uLightDir;
  varying float vNorm;
  varying vec3 vNormal;
  void main() {
    float scalarVal = uMinS + (vNorm + 1.0) * 0.5 * uRange;
    float eps = ${NODAL_THRESHOLD_RATIO.toFixed(4)} * max(uMaxAbs, 1e-10);
    float nodal = step(abs(scalarVal), eps) * uHighlightNodal * step(1e-6, uMaxAbs);
    float contourW = 0.006;
    float contour = 0.0;
    for (float i = 0.0; i < 16.0; i += 1.0) {
      if (i >= uContourCount) break;
      float level = -1.0 + (2.0 * i + 1.0) / uContourCount;
      float d = abs(vNorm - level);
      if (d < contourW) { contour = 1.0; break; }
    }
    vec3 col;
    if (nodal > 0.5) {
      col = vec3(0.0, 0.0, 0.0);
    } else if (contour > 0.5) {
      col = vec3(0.95, 0.9, 0.6);
    } else if (vNorm <= 0.0) {
      float t = 1.0 + vNorm;
      col = mix(vec3(0.15, 0.35, 0.95), vec3(1.0, 1.0, 1.0), t);
    } else {
      float t = vNorm;
      col = mix(vec3(1.0, 1.0, 1.0), vec3(0.95, 0.25, 0.2), t);
    }
    vec3 n = normalize(vNormal);
    float diff = max(0.0, dot(n, uLightDir));
    diff = 0.5 + 0.5 * diff;
    gl_FragColor = vec4(col * diff, 1.0);
  }
`;

interface MeshSceneProps {
  meshName?: string;
  meshData: MeshData | null;
  coefficients: number[];
  highlightNodal: boolean;
  /** Number of contour levels (evenly spaced). 0 = no contours. */
  contourLevels?: number;
  /** When true, read scalar/uniforms from display refs only (no coefficient→scalar compute). */
  usePrecomputedScalar?: boolean;
  displayScalarRef?: React.MutableRefObject<Float32Array | null>;
  displayUniformsRef?: React.MutableRefObject<{ minS: number; range: number; maxAbs: number } | null>;
}

const worldLightDir = new THREE.Vector3(0.5, 0.6, 0.6).normalize();

function runAfterNextFrame(fn: () => void): void {
  requestAnimationFrame(() => {
    setTimeout(fn, 0);
  });
}

interface ScalarResult {
  minS: number;
  range: number;
  maxAbs: number;
}

const BLEND_RATE = 42;

export function MeshScene({
  meshName = "",
  meshData,
  coefficients,
  highlightNodal,
  contourLevels = 7,
  usePrecomputedScalar = false,
  displayScalarRef,
  displayUniformsRef,
}: MeshSceneProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const scalarRef = useRef<Float32Array | null>(null);
  const scalarAttrRef = useRef<THREE.BufferAttribute | null>(null);
  const prevScalarRef = useRef<Float32Array | null>(null);
  const currentScalarRef = useRef<Float32Array | null>(null);
  const prevResultRef = useRef<ScalarResult | null>(null);
  const currentResultRef = useRef<ScalarResult | null>(null);
  const lastAppliedResultRef = useRef<ScalarResult>({ minS: 0, range: 1, maxAbs: 1 });
  const blendAlphaRef = useRef(1);
  const prevCoefficientsRef = useRef<number[] | null>(null);
  const pendingResultRef = useRef<ScalarResult | null>(null);
  const lightDirView = useRef(new THREE.Vector3());
  const uniformsRef = useRef({
    uMinS: { value: 0 },
    uRange: { value: 1 },
    uMaxAbs: { value: 1 },
    uHighlightNodal: { value: 1 },
    uContourCount: { value: 7 },
    uLightDir: { value: new THREE.Vector3(0.5, 0.6, 0.6).normalize() },
  });
  const { camera } = useThree();

  const { geometry } = useMemo(() => {
    if (!meshData) return { geometry: null };
    const V = meshData.vertices;
    const F = meshData.faces;
    const nV = V.length;
    const positions = new Float32Array(nV * 3);
    for (let i = 0; i < nV; i++) {
      positions[i * 3] = V[i][0];
      positions[i * 3 + 1] = V[i][1];
      positions[i * 3 + 2] = V[i][2];
    }
    const indices: number[] = [];
    for (const face of F) indices.push(face[0], face[1], face[2]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setIndex(indices);
    geo.setAttribute("scalar", new THREE.BufferAttribute(new Float32Array(nV), 1));
    geo.computeVertexNormals();
    return { geometry: geo };
  }, [meshData]);

  useLayoutEffect(() => {
    if (!geometry || !meshData) return;
    const nV = meshData.vertices.length;
    const attr = geometry.getAttribute("scalar") as THREE.BufferAttribute;
    scalarAttrRef.current = attr;
    scalarRef.current = new Float32Array(nV);
    prevScalarRef.current = new Float32Array(nV);
    currentScalarRef.current = new Float32Array(nV);
    prevResultRef.current = null;
    prevCoefficientsRef.current = null;
    blendAlphaRef.current = 1;
  }, [geometry, meshData]);

  useFrame((_, delta) => {
    if (!meshData || !scalarAttrRef.current) return;
    const attr = scalarAttrRef.current;
    const nV = attr.array.length;

    lightDirView.current.copy(worldLightDir).transformDirection(camera.matrixWorldInverse);
    uniformsRef.current.uLightDir.value.copy(lightDirView.current);
    uniformsRef.current.uHighlightNodal.value = highlightNodal ? 1 : 0;
    uniformsRef.current.uContourCount.value = Math.max(0, Math.min(16, contourLevels));

    if (usePrecomputedScalar && displayScalarRef?.current && displayUniformsRef?.current && displayScalarRef.current.length === nV) {
      (attr.array as Float32Array).set(displayScalarRef.current);
      attr.needsUpdate = true;
      const u = displayUniformsRef.current;
      uniformsRef.current.uMinS.value = u.minS;
      uniformsRef.current.uRange.value = u.range;
      uniformsRef.current.uMaxAbs.value = u.maxAbs;
      return;
    }

    if (!scalarRef.current) return;
    const pending = pendingResultRef.current;
    if (pending) {
      const prevScalar = prevScalarRef.current!;
      const currentScalar = currentScalarRef.current!;
      prevScalar.set(attr.array as Float32Array);
      currentScalar.set(scalarRef.current!);
      if (prevResultRef.current === null) {
        attr.array.set(scalarRef.current!);
        attr.needsUpdate = true;
        uniformsRef.current.uMinS.value = pending.minS;
        uniformsRef.current.uRange.value = pending.range;
        uniformsRef.current.uMaxAbs.value = pending.maxAbs;
        lastAppliedResultRef.current = { ...pending };
        prevResultRef.current = { ...pending };
        currentResultRef.current = { ...pending };
        blendAlphaRef.current = 1;
      } else {
        prevResultRef.current = { ...lastAppliedResultRef.current };
        currentResultRef.current = { ...pending };
        lastAppliedResultRef.current = { ...pending };
        blendAlphaRef.current = 0;
      }
      pendingResultRef.current = null;
    }

    const alpha = blendAlphaRef.current;
    if (alpha < 1 && prevScalarRef.current && currentResultRef.current) {
      const prevS = prevScalarRef.current;
      const currS = currentScalarRef.current!;
      const prevR = prevResultRef.current!;
      const currR = currentResultRef.current;
      blendAlphaRef.current = Math.min(1, alpha + delta * BLEND_RATE);
      const t = blendAlphaRef.current;
      const arr = attr.array as Float32Array;
      for (let v = 0; v < nV; v++) arr[v] = prevS[v] + t * (currS[v] - prevS[v]);
      uniformsRef.current.uMinS.value = prevR.minS + t * (currR.minS - prevR.minS);
      uniformsRef.current.uRange.value = prevR.range + t * (currR.range - prevR.range);
      uniformsRef.current.uMaxAbs.value = prevR.maxAbs + t * (currR.maxAbs - prevR.maxAbs);
      attr.needsUpdate = true;
    }

    if (!coefficients.length) return;
    if (prevCoefficientsRef.current === coefficients) return;

    prevCoefficientsRef.current = coefficients;
    const ev = meshData.eigenvectors;
    const N = Math.min(coefficients.length, ev.length);
    const scalar = scalarRef.current!;
    const coeffs = coefficients;

    runAfterNextFrame(() => {
      if (!scalarAttrRef.current || !scalarRef.current) return;
      let minS = Infinity,
        maxS = -Infinity;
      for (let v = 0; v < nV; v++) {
        let s = 0;
        for (let i = 0; i < N; i++) s += coeffs[i] * (ev[i][v] ?? 0);
        scalar[v] = s;
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
      pendingResultRef.current = { minS, range, maxAbs };
    });
  });

  if (!geometry) {
    return (
      <mesh>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color="#333" wireframe />
      </mesh>
    );
  }

  const content = (
    <mesh ref={meshRef} geometry={geometry}>
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniformsRef.current}
        side={THREE.DoubleSide}
      />
    </mesh>
  );

  if (meshName === "flat_plate") {
    return (
      <group position={[-0.5, 0, 0.5]} rotation={[-Math.PI / 2, 0, 0]}>
        {content}
      </group>
    );
  }
  return content;
}
