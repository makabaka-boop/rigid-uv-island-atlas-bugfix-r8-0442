import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeCheckerTexture, normaliseUVs } from "./checker.js";

// Build seam lines from original vertex ids.
function seamLineSegments(vertices, seamEdges, color, opacity = 1) {
  const pos = [];
  for (const [a, b] of seamEdges) {
    pos.push(...vertices[a], ...vertices[b]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  const m = new THREE.LineBasicMaterial({
    color,
    linewidth: 2,
    opacity,
    transparent: opacity < 1,
  });
  return new THREE.LineSegments(g, m);
}

// Boundary of the SPLIT mesh: vertices can coincide (seam tips); drop the
// half-edges that are themselves seam edges since those are red elsewhere.
function boundaryLineSegments(split, seamSet) {
  const pos = [];
  const uv3 = split.uv_vertices_3d;
  const drawn = new Set();
  for (const loop of split.boundary_loops) {
    for (let i = 0; i < loop.length; i++) {
      const a = loop[i];
      const b = loop[(i + 1) % loop.length];
      const pa = uv3[a];
      const pb = uv3[b];
      const key = `${pa.join(",")}|${pb.join(",")}`;
      const orig = [split.orig_of_uv[a], split.orig_of_uv[b]];
      const edgeKey = `${Math.min(...orig)}-${Math.max(...orig)}`;
      if (seamSet.has(edgeKey) || drawn.has(key)) continue;
      drawn.add(key);
      pos.push(...pa, ...pb);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  return new THREE.LineSegments(
    g,
    new THREE.LineBasicMaterial({ color: 0xffa500 }),
  );
}

export default function ThreeView({
  mesh,
  split,
  solution,
  selectedFace,
  anchors,
  mode,
  onPickFace,
  onToggleSeam,
}) {
  const mountRef = useRef(null);
  const stateRef = useRef(null);

  // one-time scene setup
  useEffect(() => {
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1d24);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.position.set(2.5, 2, 3.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(3, 5, 4);
    scene.add(key);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    const state = {
      scene,
      camera,
      renderer,
      controls,
      raycaster,
      pointer,
      modelGroup: new THREE.Group(),
      pickables: [],
    };
    scene.add(state.modelGroup);
    stateRef.current = state;

    function frame() {
      requestAnimationFrame(frame);
      controls.update();
      renderer.render(scene, camera);
    }
    frame();

    const ro = new ResizeObserver(() => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(mount);

    return () => {
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  // closest of a triangle's three edges to a 3D point
  function nearestEdge(face, point) {
    const v = mesh.vertices;
    const ids = mesh.faces[face];
    const p = point;
    let best = null;
    for (let j = 0; j < 3; j++) {
      const a = new THREE.Vector3(...v[ids[j]]);
      const b = new THREE.Vector3(...v[ids[(j + 1) % 3]]);
      const ab = new THREE.Vector3().subVectors(b, a);
      const t = THREE.MathUtils.clamp(
        new THREE.Vector3().subVectors(p, a).dot(ab) / ab.lengthSq(),
        0,
        1,
      );
      const proj = a.clone().addScaledVector(ab, t);
      const d = new THREE.Vector3().subVectors(p, proj).length();
      if (!best || d < best.d) best = { d, edge: [ids[j], ids[(j + 1) % 3]] };
    }
    return best;
  }

  // pointer interaction
  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    const dom = state.renderer.domElement;

    function onClick(e) {
      const rect = dom.getBoundingClientRect();
      state.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      state.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      state.raycaster.setFromCamera(state.pointer, state.camera);
      const hits = state.raycaster.intersectObjects(state.pickables, false);
      if (!hits.length) return;
      const hit = hits[0];
      const face = hit.faceIndex;
      if (mode === "seam") {
        const { d, edge } = nearestEdge(face, hit.point);
        const maxEdge = Math.max(
          ...[0, 1, 2].map((j) => {
            const a = new THREE.Vector3(...mesh.vertices[mesh.faces[face][j]]);
            const b = new THREE.Vector3(
              ...mesh.vertices[mesh.faces[face][(j + 1) % 3]],
            );
            return a.distanceTo(b);
          }),
        );
        if (d < 0.35 * maxEdge) {
          onToggleSeam([Math.min(...edge), Math.max(...edge)]);
        } else {
          onPickFace(face);
        }
      } else {
        onPickFace(face);
      }
    }
    dom.addEventListener("click", onClick);
    return () => dom.removeEventListener("click", onClick);
  }, [mesh, mode, onPickFace, onToggleSeam]);

  // rebuild model when inputs change
  useEffect(() => {
    const state = stateRef.current;
    if (!state || !mesh) return;
    state.modelGroup.clear();
    state.pickables = [];

    const seamSet = new Set(
      mesh.seam_edges.map(([a, b]) => `${Math.min(a, b)}-${Math.max(a, b)}`),
    );

    let geometry;
    if (solution && split) {
      // non-indexed: split UV corners get their own checker coordinate
      const norm = normaliseUVs(solution.uv);
      const pos = [];
      const uv = [];
      for (const tri of split.corner_uv) {
        for (const u of tri) {
          const orig = split.orig_of_uv[u];
          pos.push(...mesh.vertices[orig]);
          uv.push(norm[u][0], norm[u][1]);
        }
      }
      geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(pos, 3),
      );
      geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
      geometry.computeVertexNormals();
    } else {
      geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(mesh.vertices.flat(), 3),
      );
      geometry.setIndex(mesh.faces.flat());
      geometry.computeVertexNormals();
    }

    const material = solution
      ? new THREE.MeshStandardMaterial({
          map: makeCheckerTexture(),
          roughness: 0.9,
          metalness: 0,
          side: THREE.DoubleSide,
        })
      : new THREE.MeshStandardMaterial({
          color: 0x9fb4d8,
          roughness: 0.8,
          side: THREE.DoubleSide,
          flatShading: true,
        });
    const model = new THREE.Mesh(geometry, material);
    state.modelGroup.add(model);
    state.pickables.push(model);

    // wireframe
    const wire = new THREE.LineSegments(
      new THREE.WireframeGeometry(geometry),
      new THREE.LineBasicMaterial({
        color: 0x22262e,
        transparent: true,
        opacity: 0.6,
      }),
    );
    state.modelGroup.add(wire);

    if (mesh.seam_edges.length) {
      state.modelGroup.add(
        seamLineSegments(mesh.vertices, mesh.seam_edges, 0xff3b3b),
      );
    }
    if (split && split.valid) {
      state.modelGroup.add(boundaryLineSegments(split, seamSet));
    }

    // selected face highlight
    if (selectedFace != null) {
      const ids = mesh.faces[selectedFace].flatMap((o) => mesh.vertices[o]);
      const hg = new THREE.BufferGeometry();
      hg.setAttribute("position", new THREE.Float32BufferAttribute(ids, 3));
      const hl = new THREE.Mesh(
        hg,
        new THREE.MeshBasicMaterial({
          color: 0xffe14d,
          transparent: true,
          opacity: 0.55,
          side: THREE.DoubleSide,
        }),
      );
      state.modelGroup.add(hl);
    }

    // anchors
    if (solution && anchors && split) {
      anchors.forEach((u, k) => {
        const s = new THREE.Mesh(
          new THREE.SphereGeometry(0.045, 16, 12),
          new THREE.MeshBasicMaterial({ color: k === 0 ? 0x2ecc71 : 0x3498ff }),
        );
        s.position.copy(new THREE.Vector3(...split.uv_vertices_3d[u]));
        state.modelGroup.add(s);
      });
    }

    // center / scale camera once per mesh
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    state.controls.target.copy(center);
    state.camera.position.copy(
      center
        .clone()
        .add(new THREE.Vector3(1.6, 1.2, 1.8).normalize().multiplyScalar(size)),
    );
    state.camera.near = size / 100;
    state.camera.far = size * 100;
    state.camera.updateProjectionMatrix();
  }, [mesh, split, solution, selectedFace, anchors]);

  return <div ref={mountRef} style={{ width: "100%", height: "100%" }} />;
}
