import * as THREE from "three";

// Procedural 8x8 checker texture generated in UV space.  Each texture
// repeat covers a [0,1]^2 UV square; wrapping repeats it over the sheet.
export function makeCheckerTexture() {
  const n = 256;
  const cells = 8;
  const c = document.createElement("canvas");
  c.width = c.height = n;
  const ctx = c.getContext("2d");
  const step = n / cells;
  for (let i = 0; i < cells; i++) {
    for (let j = 0; j < cells; j++) {
      ctx.fillStyle = (i + j) % 2 === 0 ? "#f4f4f4" : "#3b3b44";
      ctx.fillRect(i * step, j * step, step, step);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 4;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// Normalise LSCM UVs (anchors pinned at (0,0) and (1,0)) to [0,1]^2 while
// keeping a uniform aspect scale so the checker shows real distortion.
export function normaliseUVs(uv) {
  const minX = Math.min(...uv.map((p) => p[0]));
  const maxX = Math.max(...uv.map((p) => p[0]));
  const minY = Math.min(...uv.map((p) => p[1]));
  const maxY = Math.max(...uv.map((p) => p[1]));
  const s = Math.max(maxX - minX, maxY - minY) || 1;
  return uv.map(([x, y]) => [(x - minX) / s, (y - minY) / s]);
}
