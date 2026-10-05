import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchSample, prepareMesh, solveMesh } from "./api.js";
import ThreeView from "./ThreeView.jsx";
import UVView from "./UVView.jsx";

const SAMPLES = [
  { id: "plane", label: "平面网格（已是圆盘）" },
  { id: "cube_cut", label: "立方体 + 对偶树切缝（合法）" },
  { id: "plane_cut", label: "平面 + 边界切入缝（合法）" },
  { id: "cube_closed", label: "封闭立方体（无切缝，非法）" },
  { id: "octahedron_closed", label: "封闭八面体（见证）" },
];

const WITNESS_LABEL = {
  closed_surface: "封闭曲面：完全没有边界，无法映到圆盘",
  multiple_boundary_loops: "边界环多于一个：切口未把表面开成一个圆盘",
  disconnected: "切后不连通：切缝把面片分成了多个组件",
  non_disk_characteristic: "单个边界环但欧拉示性数 ≠ 1：仍有把手/未开顶拓扑",
  invalid_input: "输入不合法",
  open_boundary_chain: "边界半边未闭合：表面不是闭合的二维流形",
};

export default function App() {
  const [mesh, setMesh] = useState(null);
  const [split, setSplit] = useState(null);
  const [solution, setSolution] = useState(null);
  const [selectedFace, setSelectedFace] = useState(null);
  const [mode, setMode] = useState("inspect"); // inspect | seam | anchor
  const [anchors, setAnchors] = useState([null, null]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const loadSample = useCallback(async (id) => {
    setBusy(true);
    setError("");
    setSelectedFace(null);
    setSolution(null);
    setAnchors([null, null]);
    try {
      const m = await fetchSample(id);
      m.seam_edges = m.seam_edges.map(([a, b]) => [
        Math.min(a, b),
        Math.max(a, b),
      ]);
      setMesh(m);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    loadSample("cube_cut");
  }, [loadSample]);

  // prepare whenever mesh/seams change
  useEffect(() => {
    if (!mesh) return;
    let cancelled = false;
    setBusy(true);
    setSolution(null);
    setAnchors([null, null]);
    setSplit(null);
    prepareMesh(mesh)
      .then((p) => {
        if (cancelled) return;
        setError("");
        setSplit(p);
      })
      .catch((e) => setError(String(e)))
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
  }, [mesh]);

  const toggleSeam = useCallback((edge) => {
    setMesh((m) => {
      const key = (e) => `${Math.min(...e)}-${Math.max(...e)}`;
      const have = new Set(m.seam_edges.map(key));
      const k = key(edge);
      const next = have.has(k)
        ? m.seam_edges.filter((e) => key(e) !== k)
        : [...m.seam_edges, edge];
      return { ...m, seam_edges: next };
    });
  }, []);

  const pickAnchor = useCallback((u) => {
    setAnchors(([a0, a1]) => {
      if (a0 == null) return [u, a1];
      if (a1 == null && u !== a0) return [a0, u];
      // restart the pair
      return [u, null];
    });
  }, []);

  const solve = useCallback(async () => {
    if (!mesh || anchors[0] == null || anchors[1] == null) return;
    setBusy(true);
    setError("");
    try {
      const res = await solveMesh({
        vertices: mesh.vertices,
        faces: mesh.faces,
        seam_edges: mesh.seam_edges,
        anchor0: anchors[0],
        anchor1: anchors[1],
      });
      if (!res.ok) {
        const w = res.witness;
        setError(
          res.error ||
            (w
              ? WITNESS_LABEL[w.kind] || `不是拓扑圆盘：${w.kind}`
              : "not a topological disk; solve was not run"),
        );
        setSolution(null);
      } else {
        setSolution(res);
      }
    } finally {
      setBusy(false);
    }
  }, [mesh, anchors]);

  const exportJSON = useCallback(() => {
    if (!solution) return;
    const blob = new Blob(
      [
        JSON.stringify(
          {
            note:
              "per-corner UVs with original face id; seam-separated corners " +
              "sharing a 3D vertex intentionally keep distinct uv_vertex ids",
            anchor_uv_vertices: solution.anchors,
            corner_count: solution.corners.length,
            corners: solution.corners,
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "lscm_corner_uvs.json";
    a.click();
    URL.revokeObjectURL(url);
  }, [solution]);

  const faceInfo = useMemo(() => {
    if (selectedFace == null) return null;
    const row = {};
    if (solution) {
      row.angleErr = solution.face_angle_error_deg[selectedFace];
      row.conformal = solution.face_conformal_residual[selectedFace];
      row.flipped = solution.flipped_faces.includes(selectedFace);
      row.degenerate = solution.degenerate_uv_faces.includes(selectedFace);
    }
    return row;
  }, [selectedFace, solution]);

  const diskOK = split?.valid;

  return (
    <div className="app">
      <header>
        <h1>LSCM 小网格 UV 展开工具</h1>
        <div className="toolbar">
          <select
            onChange={(e) => loadSample(e.target.value)}
            defaultValue="cube_cut"
          >
            {SAMPLES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          <button
            className={mode === "seam" ? "active" : ""}
            onClick={() => setMode(mode === "seam" ? "inspect" : "seam")}
          >
            切缝边选择
          </button>
          <button
            className={mode === "anchor" ? "active" : ""}
            disabled={!diskOK}
            onClick={() => setMode(mode === "anchor" ? "inspect" : "anchor")}
          >
            锚点选择
          </button>
          <button
            onClick={solve}
            disabled={!diskOK || anchors[0] == null || anchors[1] == null}
          >
            求解 LSCM
          </button>
          <button onClick={exportJSON} disabled={!solution}>
            导出角点 UV / 面 ID
          </button>
          {busy && <span className="busy">处理中…</span>}
        </div>
      </header>

      <div className="views">
        <div className="pane">
          <h2>三维模型</h2>
          <div className="canvas-wrap">
            {mesh && (
              <ThreeView
                mesh={mesh}
                split={split}
                solution={solution}
                selectedFace={selectedFace}
                anchors={solution ? solution.anchors : null}
                mode={mode}
                onPickFace={setSelectedFace}
                onToggleSeam={toggleSeam}
              />
            )}
          </div>
          <p className="hint">
            {mode === "seam"
              ? "点击靠近面的一条边可加入/移除切缝（红线）；点击面中部仅选中面。"
              : "点击任意面选中；与二维视图联动。"}
          </p>
        </div>

        <div className="pane">
          <h2>
            二维结果
            {solution ? "（LSCM + 棋盘纹理）" : "（Tutte 预览，仅供选锚）"}
          </h2>
          <div className="canvas-wrap">
            {split?.valid ? (
              <UVView
                split={split}
                solution={solution}
                selectedFace={selectedFace}
                anchors={
                  mode === "anchor"
                    ? anchors
                    : solution
                      ? solution.anchors
                      : anchors
                }
                mode={mode}
                onSelectFace={setSelectedFace}
                onPickAnchor={pickAnchor}
              />
            ) : (
              <div className="witness">尚未形成拓扑圆盘，无法显示二维图</div>
            )}
          </div>
          <p className="hint">
            {mode === "anchor"
              ? "在白色边界顶点上依次选两个不同锚点：绿=(0,0)，蓝=(1,0)。"
              : "点击三角形与三维视图联动；红面=翻转，橙面=退化。"}
          </p>
        </div>
      </div>

      <section className="panels">
        <div className="panel">
          <h3>拓扑状态</h3>
          {split && (
            <>
              <p>
                三角形：<b>{split.n_faces}</b>（上限 200）， 原顶点：
                <b>{split.n_orig_vertices}</b>， 拆分后 UV 顶点：
                <b>{split.n_uv_vertices}</b>
              </p>
              <p>
                切缝边：<b>{split.seam_edges.length}</b>
                ，边界环数：<b>{split.boundary_loops.length}</b>
                ，边界 UV 顶点：<b>{split.boundary_uvs.length}</b>
              </p>
              {diskOK ? (
                <p className="ok">✓ 切开后是一个拓扑圆盘，可以求解。</p>
              ) : (
                <p className="bad">
                  ✗ {WITNESS_LABEL[split.witness?.kind] || "不是拓扑圆盘"}
                  <pre>{JSON.stringify(split.witness, null, 2)}</pre>
                </p>
              )}
            </>
          )}
          {error && <p className="bad">{error}</p>}
        </div>

        <div className="panel">
          <h3>求解质量</h3>
          {solution ? (
            <>
              <p>
                秩：
                <b>
                  {solution.rank} / {solution.expected_rank}
                </b>
                {solution.rank_deficient ? (
                  <span className="bad"> （秩不足！解不唯一）</span>
                ) : (
                  <span className="ok"> （满秩）</span>
                )}
              </p>
              <p>
                RMS 行残差：<b>{solution.rms_row_residual.toExponential(3)}</b>
                ，能量 ΣA|C|²：<b>{solution.energy.toExponential(3)}</b>
              </p>
              <p>
                最大共形残差 |C|：
                <b>{solution.max_conformal_residual.toExponential(3)}</b>
              </p>
              <p>
                最小奇异值：
                {solution.smallest_singular_values
                  .map((s) => s.toExponential(2))
                  .join(", ")}
              </p>
              <p>
                翻转面：
                {solution.flipped_faces.length ? (
                  <span className="bad">
                    {solution.flipped_faces.join(", ")}
                  </span>
                ) : (
                  <span className="ok">无</span>
                )}
                ；退化面：
                {solution.degenerate_uv_faces.length ? (
                  <span className="bad">
                    {solution.degenerate_uv_faces.join(", ")}
                  </span>
                ) : (
                  <span className="ok">无</span>
                )}
              </p>
            </>
          ) : (
            <p className="muted">选择两个边界锚点后点击“求解 LSCM”。</p>
          )}
        </div>

        <div className="panel">
          <h3>面信息 {selectedFace != null ? `#${selectedFace}` : ""}</h3>
          {selectedFace == null ? (
            <p className="muted">在任一视图点击一个面。</p>
          ) : solution ? (
            <>
              <p>
                最大角误差：<b>{faceInfo.angleErr.toFixed(3)}°</b>
              </p>
              <p>
                该面共形残差 |C|：<b>{faceInfo.conformal.toExponential(3)}</b>
              </p>
              <p>
                {faceInfo.flipped && <span className="bad">翻转！ </span>}
                {faceInfo.degenerate && <span className="bad">退化！ </span>}
                棋盘纹理：
                {faceInfo.flipped ? "翻转（红色叠加）" : "已映射到三维面"}
              </p>
              <CornerTable solution={solution} face={selectedFace} />
            </>
          ) : (
            <p>求解后显示该面每个角的角度误差与 UV。</p>
          )}
        </div>
      </section>
    </div>
  );
}

function CornerTable({ solution, face }) {
  const rows = solution.corners.filter((c) => c.face === face);
  return (
    <table className="corners">
      <thead>
        <tr>
          <th>角</th>
          <th>原顶点</th>
          <th>UV 顶点</th>
          <th>u</th>
          <th>v</th>
          <th>角误差°</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.corner}>
            <td>{r.corner}</td>
            <td>{r.orig_vertex}</td>
            <td>{r.uv_vertex}</td>
            <td>{r.u.toFixed(4)}</td>
            <td>{r.v.toFixed(4)}</td>
            <td>{r.angle_error_deg.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
