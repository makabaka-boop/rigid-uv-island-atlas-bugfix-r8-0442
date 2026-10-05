const SVG = "http://www.w3.org/2000/svg";
const COLORS = ["#1464a0", "#a04a14", "#2e7d32", "#7b1fa2",
  "#c62828", "#00838f", "#ad7900", "#4e342e"];

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.append(node);
  return node;
}

let lastData = null;

async function loadSample() {
  const r = await fetch("/api/atlas/sample");
  if (!r.ok) throw new Error(`sample HTTP ${r.status}`);
  $("request").value = JSON.stringify(await r.json(), null, 2);
}

function render(data) {
  lastData = data;
  const view = $("view");
  view.replaceChildren();
  const legend = $("legend");
  legend.replaceChildren();

  const p = data.atlas.padding;
  // unified outer-margin rectangle
  el("rect", {
    x: p * 500, y: p * 500,
    width: (1 - 2 * p) * 500, height: (1 - 2 * p) * 500,
    fill: "none", stroke: "#bbb", "stroke-dasharray": "4 3",
  }, view);

  data.atlas.islands.forEach((island, idx) => {
    const color = COLORS[idx % COLORS.length];
    const b = island.bounds;
    // bounding box actually occupied by this island's own faces
    el("rect", {
      x: b.u_min * 500, y: (1 - b.v_max) * 500,
      width: (b.u_max - b.u_min) * 500,
      height: (b.v_max - b.v_min) * 500,
      fill: color, "fill-opacity": 0.05,
      stroke: color, "stroke-dasharray": "2 3",
    }, view);
    el("text", {
      x: ((b.u_min + b.u_max) / 2) * 500,
      y: (1 - b.v_max) * 500 - 4,
      "text-anchor": "middle", "font-size": 14, fill: color,
    }, view).textContent = `${island.id} (rot ${island.rotation_deg}°)`;

    const corners = island.corners;
    for (let k = 0; k < corners.length; k += 3) {
      const face = el("polygon", {
        points: corners.slice(k, k + 3)
          .map((q) => `${q.u * 500},${(1 - q.v) * 500}`).join(" "),
        fill: "none", stroke: color, "stroke-width": 1.4,
      }, view);
      const row0 = corners[k];
      face.title = `${island.id} face=${row0.face}\n` +
        corners.slice(k, k + 3).map((q) =>
          `corner${q.corner}: orig=${q.orig_vertex} uv=${q.uv_vertex} ` +
          `(${q.u.toFixed(4)},${q.v.toFixed(4)})`).join("\n");
      face.addEventListener("click", () => {
        $("pick").textContent =
          `island_id : ${island.id}\nface      : ${row0.face}\n` +
          corners.slice(k, k + 3).map((q) =>
            `corner ${q.corner}: orig_vertex=${q.orig_vertex} ` +
            `uv_vertex=${q.uv_vertex}`).join("\n");
      });
      view.append(face);
    }

    const li = document.createElement("li");
    li.style.color = color;
    li.textContent = `${island.id} — band ${island.band}, ` +
      `${corners.length / 3} faces, rotation ${island.rotation_deg}°`;
    legend.append(li);
  });

  $("status").textContent =
    `common scale = ${data.atlas.scale.toFixed(5)}, padding = ${p}`;

  const a = $("download");
  a.hidden = false;
  a.href = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
}

async function run() {
  const output = $("output");
  output.textContent = "";
  $("status").textContent = "running…";
  try {
    const r = await fetch("/api/atlas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: $("request").value,
    });
    const data = await r.json();
    output.textContent = JSON.stringify(data, null, 2);
    if (!r.ok || !data.ok) {
      $("status").textContent = "rejected: " + (data.error || r.status);
      return;
    }
    render(data);
  } catch (e) {
    $("status").textContent = "error: " + e.message;
  }
}

$("sample").onclick = () => loadSample().catch((e) => {
  $("status").textContent = e.message;
});
$("run").onclick = run;
