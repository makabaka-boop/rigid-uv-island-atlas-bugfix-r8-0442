document.getElementById("run").onclick = async () => {
  const output = document.getElementById("output"),
    view = document.getElementById("view");
  view.replaceChildren();
  try {
    const r = await fetch("/api/atlas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: document.getElementById("request").value,
    });
    const data = await r.json();
    output.textContent = JSON.stringify(data, null, 2);
    if (!r.ok) return;
    for (const island of data.atlas.islands) {
      const corners = island.corners;
      for (let k = 0; k < corners.length; k += 3) {
        const face = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "polygon",
        );
        face.setAttribute(
          "points",
          corners
            .slice(k, k + 3)
            .map((p) => `${p.u * 500},${(1 - p.v) * 500}`)
            .join(" "),
        );
        face.setAttribute("fill", "none");
        face.setAttribute("stroke", "#1464a0");
        face.dataset.island = island.id;
        face.dataset.face = corners[k].face;
        view.append(face);
      }
    }
    const a = document.createElement("a");
    a.textContent = "下载 UV 与身份";
    a.download = "atlas.json";
    a.href = URL.createObjectURL(
      new Blob([JSON.stringify(data)], { type: "application/json" }),
    );
    output.append(a);
  } catch (e) {
    output.textContent = e.message;
  }
};
