import { useEffect, useRef } from "react";
import { normaliseUVs } from "./checker.js";

// Hit test a point against a triangle (2D), returns barycentric if inside.
function pointInTriangle(px, py, ax, ay, bx, by, cx, cy) {
  const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}

export default function UVView({
  split,
  solution,
  selectedFace,
  anchors,
  mode,
  onSelectFace,
  onPickAnchor,
}) {
  const canvasRef = useRef(null);
  const normRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const pad = 40;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#15171c";
    ctx.fillRect(0, 0, W, H);

    // preview embedding before a solve exists (anchor picking layout)
    let coords;
    if (solution) {
      coords = normaliseUVs(solution.uv);
    } else if (split?.valid && split.preview_uv) {
      coords = split.preview_uv; // already roughly in [-1,1]
    } else {
      return;
    }
    normRef.current = coords;

    // map coords (roughly [0,1] after normalise, [-1.4,1.4] preview) to px
    const lo = solution ? 0 : -1.4;
    const hi = solution ? 1 : 1.4;
    const span = hi - lo;
    const toX = (x) => pad + ((x - lo) / span) * (W - 2 * pad);
    const toY = (y) => H - pad - ((y - lo) / span) * (H - 2 * pad);

    const cornerUv = split.corner_uv;
    const flipped = new Set(solution?.flipped_faces ?? []);
    const deg = new Set(solution?.degenerate_uv_faces ?? []);

    // checker background clipped to the UV sheet (solved view only)
    if (solution) {
      const sheetPath = new Path2D();
      for (let f = 0; f < cornerUv.length; f++) {
        const tri = cornerUv[f];
        const p = tri.map((u) => coords[u]);
        sheetPath.moveTo(toX(p[0][0]), toY(p[0][1]));
        sheetPath.lineTo(toX(p[1][0]), toY(p[1][1]));
        sheetPath.lineTo(toX(p[2][0]), toY(p[2][1]));
        sheetPath.closePath();
      }
      ctx.save();
      ctx.clip(sheetPath);
      ctx.fillStyle = "#e9e9ef";
      ctx.fillRect(0, 0, W, H);
      const cells = 8;
      const step = (toX(1) - toX(0)) / cells;
      const top = toY(1);
      for (let i = 0; i < cells; i++) {
        for (let j = 0; j < cells; j++) {
          if ((i + j) % 2 === 0) {
            ctx.fillStyle = "#3d3d47";
            ctx.fillRect(toX(0) + i * step, top + j * step, step, step);
          }
        }
      }
      ctx.restore();
    }

    for (let f = 0; f < cornerUv.length; f++) {
      const tri = cornerUv[f];
      const p = tri.map((u) => coords[u]);
      const path = new Path2D();
      path.moveTo(toX(p[0][0]), toY(p[0][1]));
      path.lineTo(toX(p[1][0]), toY(p[1][1]));
      path.lineTo(toX(p[2][0]), toY(p[2][1]));
      path.closePath();

      if (!solution) {
        ctx.fillStyle = "#3a4150";
        ctx.fill(path);
      }
      ctx.strokeStyle = f === selectedFace ? "#ffe14d" : "#20232b";
      ctx.lineWidth = f === selectedFace ? 2.5 : 1;
      ctx.stroke(path);

      if (flipped.has(f)) {
        ctx.fillStyle = "rgba(255,60,60,0.45)";
        ctx.fill(path);
      }
      if (deg.has(f)) {
        ctx.fillStyle = "rgba(255,160,0,0.4)";
        ctx.fill(path);
      }
    }

    // seam edges on the split mesh: boundary half-edges that correspond to
    // user-selected seams (drawn red over the checker)
    const seamOrigSet = new Set(
      split.seam_edges.map(([a, b]) => `${Math.min(a, b)}-${Math.max(a, b)}`),
    );
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#ff3b3b";
    const seenHalf = new Set();
    for (const tri of cornerUv) {
      for (let j = 0; j < 3; j++) {
        const a = tri[j];
        const b = tri[(j + 1) % 3];
        const oa = split.orig_of_uv[a];
        const ob = split.orig_of_uv[b];
        const key = `${Math.min(oa, ob)}-${Math.max(oa, ob)}`;
        const hkey = `${Math.min(a, b)}-${Math.max(a, b)}`;
        if (seamOrigSet.has(key) && !seenHalf.has(hkey)) {
          seenHalf.add(hkey);
          const p0 = coords[a];
          const p1 = coords[b];
          ctx.beginPath();
          ctx.moveTo(toX(p0[0]), toY(p0[1]));
          ctx.lineTo(toX(p1[0]), toY(p1[1]));
          ctx.stroke();
        }
      }
    }

    // anchor points
    if (anchors) {
      anchors.forEach((u, k) => {
        if (u == null || !coords[u]) return;
        const [x, y] = coords[u];
        ctx.beginPath();
        ctx.arc(toX(x), toY(y), 8, 0, Math.PI * 2);
        ctx.fillStyle = k === 0 ? "#2ecc71" : "#3498ff";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px sans-serif";
        ctx.fillText(k === 0 ? "(0,0)" : "(1,0)", toX(x) + 10, toY(y) - 8);
      });
    }

    // highlight boundary vertices available as anchors
    if (mode === "anchor" && split.valid) {
      for (const u of split.boundary_uvs) {
        if (anchors && anchors.includes(u)) continue;
        const [x, y] = coords[u];
        ctx.beginPath();
        ctx.arc(toX(x), toY(y), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.65)";
        ctx.fill();
      }
    }
  }, [split, solution, selectedFace, anchors, mode]);

  function onClick(e) {
    const canvas = canvasRef.current;
    const coords = normRef.current;
    if (!coords || !split?.valid) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const W = canvas.width;
    const H = canvas.height;
    const pad = 40;
    const lo = solution ? 0 : -1.4;
    const hi = solution ? 1 : 1.4;
    const span = hi - lo;
    const x = lo + ((px - pad) / (W - 2 * pad)) * span;
    const y = lo + ((H - pad - py) / (H - 2 * pad)) * span;

    if (mode === "anchor") {
      let best = null;
      for (const u of split.boundary_uvs) {
        const [ux, uy] = coords[u];
        const d = Math.hypot(ux - x, uy - y);
        if (!best || d < best.d) best = { d, u };
      }
      if (best && best.d < 0.06 * span) {
        onPickAnchor(best.u);
        return;
      }
    }

    for (let f = 0; f < split.corner_uv.length; f++) {
      const tri = split.corner_uv[f];
      const p = tri.map((u) => coords[u]);
      if (pointInTriangle(x, y, ...p[0], ...p[1], ...p[2])) {
        onSelectFace(f);
        return;
      }
    }
    onSelectFace(null);
  }

  return (
    <canvas
      ref={canvasRef}
      width={560}
      height={560}
      onClick={onClick}
      style={{
        width: "100%",
        height: "100%",
        cursor: mode === "anchor" ? "crosshair" : "pointer",
      }}
    />
  );
}
