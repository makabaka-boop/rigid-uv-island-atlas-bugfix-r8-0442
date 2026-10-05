const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const PX = 500;
// Color per island band; identity (not coordinate) drives the coloring.
const COLORS = ["#1464a0", "#a0531f", "#2f7d32", "#7a2d8a", "#8a7d2d", "#8a2d4f", "#2d8a82", "#555"];

function el(name, attrs = {}) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

async function loadSample() {
  const r = await fetch("/api/atlas/sample");
  if (!r.ok) return;
  $("request").value = JSON.stringify(await r.json(), null, 2);
}

function cornersByFace(corners) {
  const faces = new Map();
  for (const c of corners) {
    if (!faces.has(c.face)) faces.set(c.face, [null, null, null]);
    faces.get(c.face)[c.corner] = c;
  }
  return faces;
}

function signedArea2(a, b, c) {
  return (b.u - a.u) * (c.v - a.v) - (b.v - a.v) * (c.u - a.u);
}

// Independent client-side checks of the promised invariants.
function verifyAtlas(atlas) {
  const problems = [];
  const p = atlas.padding;
  for (const island of atlas.islands) {
    if (island.scale !== atlas.scale)
      problems.push(`${island.id}: per-island scale ${island.scale} != common scale`);
    const faces = cornersByFace(island.corners);
    for (const [face, tri] of faces) {
      if (tri.some((x) => !x)) {
        problems.push(`${island.id}: face ${face} missing corners`);
        continue;
      }
      if (tri.some((x) => x.island_id !== island.id))
        problems.push(`${island.id}: face ${face} labelled with another island id`);
      const area = signedArea2(tri[0], tri[1], tri[2]);
      if (area <= 0) problems.push(`${island.id}: face ${face} flipped/degenerate (area2=${area})`);
      for (const c of tri) {
        if (c.u < -1e-9 || c.u > 1 + 1e-9 || c.v < -1e-9 || c.v > 1 + 1e-9)
          problems.push(`${island.id}: face ${face} leaves [0,1] page`);
      }
    }
    const us = island.corners.map((c) => c.u);
    const vs = island.corners.map((c) => c.v);
    if (Math.min(...us) < p - 1e-9 || Math.max(...us) > 1 - p + 1e-9 ||
        Math.min(...vs) < p - 1e-9 || Math.max(...vs) > 1 - p + 1e-9)
      problems.push(`${island.id}: violates the padding margin`);
  }
  return problems;
}

function drawIsland(island, color, index) {
  const g = el("g");
  const faces = cornersByFace(island.corners);
  for (const [face, tri] of faces) {
    const poly = el("polygon", {
      points: tri.map((c) => `${c.u * PX},${(1 - c.v) * PX}`).join(" "),
      fill: color, "fill-opacity": 0.08, stroke: color, "stroke-width": 1,
    });
    const tip = `${island.id} · 原面 ${face} · ` +
      tri.map((c) => `角${c.corner}:原顶点${c.orig_vertex}/UV顶点${c.uv_vertex}`).join(" ");
    poly.appendChild(el("title")).textContent = tip;
    poly.dataset.island = island.id;
    poly.dataset.face = face;
    g.appendChild(poly);
  }
  // First corner of face 0: shows that face/corner identity survives the
  // packing (and that a proper rotation never reverses the corner order).
  const f0 = faces.get(0);
  if (f0) {
    const c0 = f0[0], c1 = f0[1];
    g.appendChild(el("circle", {
      cx: c0.u * PX, cy: (1 - c0.v) * PX, r: 3.5, fill: color,
    })).appendChild(el("title")).textContent =
      `${island.id} 面0 角0 原顶点${c0.orig_vertex} UV顶点${c0.uv_vertex}`;
    g.appendChild(el("line", {
      x1: c0.u * PX, y1: (1 - c0.v) * PX, x2: c1.u * PX, y2: (1 - c1.v) * PX,
      stroke: color, "stroke-width": 2,
    }));
  }
  const vmin = island.corners.reduce((m, c) =>
    (c.v < m.v ? c : m), island.corners[0]);
  const label = el("text", {
    x: vmin.u * PX, y: (1 - vmin.v) * PX + 13,
    "font-size": 11, fill: color,
  });
  label.textContent = `${index}: ${island.id} (rot ${island.rotation_deg}°)`;
  g.appendChild(label);
  return g;
}

function cornerCSV(atlas) {
  const head = "island_id,face,corner,orig_vertex,uv_vertex,u,v,angle_error_deg\n";
  return head + atlas.islands.flatMap((it) => it.corners.map((c) =>
    [c.island_id, c.face, c.corner, c.orig_vertex, c.uv_vertex, c.u, c.v,
     c.angle_error_deg].join(","))).join("\n");
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.textContent = "下载 " + name;
  a.download = name;
  a.href = URL.createObjectURL(new Blob([text], { type }));
  $("downloads").append(a, document.createTextNode("  "));
}

async function run() {
  const view = $("view"), output = $("output"), status = $("status");
  view.replaceChildren();
  $("downloads").replaceChildren();
  status.textContent = "";
  try {
    const r = await fetch("/api/atlas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: $("request").value,
    });
    const data = await r.json();
    output.textContent = JSON.stringify(data, null, 2);
    if (!r.ok || !data.ok) {
      status.textContent = "整份图集被拒绝（不导出任何岛）：" + (data.error || r.status);
      status.style.color = "#a02020";
      return;
    }
    const atlas = data.atlas;
    atlas.islands.forEach((it, i) => view.appendChild(drawIsland(it, COLORS[i % COLORS.length], i)));

    const problems = verifyAtlas(atlas);
    status.textContent = problems.length
      ? "校验发现问题：" + problems.join("；")
      : `校验通过：${atlas.islands.length} 个岛共用缩放 ${atlas.scale.toFixed(6)}，` +
        "无翻转面，岛 id 与原面/面角/顶点身份一致，全部位于 padding 内。";
    status.style.color = problems.length ? "#a02020" : "#2f7d32";

    download("atlas.json", JSON.stringify(data), "application/json");
    download("corners.csv", cornerCSV(atlas), "text/csv");
  } catch (e) {
    output.textContent = e.message;
    status.textContent = "请求解析失败：" + e.message;
  }
}

$("loadSample").onclick = loadSample;
$("run").onclick = run;
loadSample();
