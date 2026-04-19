/**
 * Nodal surface topology metrics: β₀, β₁, χ, A_ratio
 * Nodal set: |f(v)| < ε, ε = thresholdRatio * max|f|
 */

export interface TopologyMetrics {
  beta0: number;
  beta1: number;
  chi: number;
  A_ratio: number;
}

function meshEdges(F: number[][]): Set<string> {
  const out = new Set<string>();
  for (const [a, b, c] of F) {
    out.add([Math.min(a, b), Math.max(a, b)].join(","));
    out.add([Math.min(b, c), Math.max(b, c)].join(","));
    out.add([Math.min(c, a), Math.max(c, a)].join(","));
  }
  return out;
}

function adjacencyFromEdges(edgeSet: Set<string>): Map<number, number[]> {
  const adj = new Map<number, number[]>();
  for (const key of edgeSet) {
    const [i, j] = key.split(",").map(Number);
    if (!adj.has(i)) adj.set(i, []);
    adj.get(i)!.push(j);
    if (!adj.has(j)) adj.set(j, []);
    adj.get(j)!.push(i);
  }
  return adj;
}

function countComponents(adj: Map<number, number[]>, nodal: Set<number>): number {
  const visited = new Set<number>();
  let count = 0;
  const queue: number[] = [];
  for (const v of nodal) {
    if (visited.has(v)) continue;
    count++;
    queue.length = 0;
    queue.push(v);
    visited.add(v);
    while (queue.length > 0) {
      const u = queue.shift()!;
      for (const w of adj.get(u) ?? []) {
        if (nodal.has(w) && !visited.has(w)) {
          visited.add(w);
          queue.push(w);
        }
      }
    }
  }
  return count;
}

function triArea(a: number[], b: number[], c: number[]): number {
  const bx = b[0] - a[0], by = b[1] - a[1], bz = b[2] - a[2];
  const cx = c[0] - a[0], cy = c[1] - a[1], cz = c[2] - a[2];
  const nx = by * cz - bz * cy;
  const ny = bz * cx - bx * cz;
  const nz = bx * cy - by * cx;
  return 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
}

function meshTotalArea(V: number[][], F: number[][]): number {
  let total = 0;
  for (const [i, j, k] of F) {
    total += triArea(V[i], V[j], V[k]);
  }
  return total;
}

function nodalFaceArea(V: number[][], F: number[][], nodal: Set<number>): number {
  let total = 0;
  for (const [i, j, k] of F) {
    if (nodal.has(i) || nodal.has(j) || nodal.has(k)) {
      total += triArea(V[i], V[j], V[k]);
    }
  }
  return total;
}

export function computeTopologyMetrics(
  vertices: number[][],
  faces: number[][],
  scalarValues: number[],
  thresholdRatio = 0.05
): TopologyMetrics {
  const nV = vertices.length;
  if (scalarValues.length !== nV) {
    return { beta0: 0, beta1: 0, chi: 0, A_ratio: 0 };
  }
  const maxAbs = Math.max(...scalarValues.map((f) => Math.abs(f)), 1e-20);
  const eps = thresholdRatio * maxAbs;
  const nodal = new Set<number>();
  for (let v = 0; v < nV; v++) {
    if (Math.abs(scalarValues[v]) < eps) nodal.add(v);
  }
  if (nodal.size === 0) {
    return { beta0: 0, beta1: 0, chi: 0, A_ratio: 0 };
  }
  const edgeSet = meshEdges(faces);
  const nodalEdges = new Set<string>();
  for (const key of edgeSet) {
    const [i, j] = key.split(",").map(Number);
    if (nodal.has(i) && nodal.has(j)) nodalEdges.add(key);
  }
  const adj = adjacencyFromEdges(nodalEdges);
  const beta0 = countComponents(adj, nodal);
  const V_nodal = nodal.size;
  const E_nodal = nodalEdges.size;
  const chi = V_nodal - E_nodal;
  const beta1 = Math.max(0, E_nodal - V_nodal + beta0);
  const totalArea = meshTotalArea(vertices, faces);
  const nodalArea = nodalFaceArea(vertices, faces, nodal);
  const A_ratio = totalArea > 1e-20 ? nodalArea / totalArea : 0;
  return { beta0, beta1, chi, A_ratio };
}
