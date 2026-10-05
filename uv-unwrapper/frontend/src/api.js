// Thin wrapper around the Flask backend.

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok && !data.witness && !data.error) {
    throw new Error(`HTTP ${res.status}`);
  }
  return data;
}

export async function fetchSample(name) {
  const res = await fetch(`/api/samples/${name}`);
  if (!res.ok) throw new Error(`sample ${name} not found`);
  return res.json();
}

export const prepareMesh = (payload) => postJSON("/api/prepare", payload);
export const solveMesh = (payload) => postJSON("/api/solve", payload);
