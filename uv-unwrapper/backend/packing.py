"""Vertical atlas layout."""

import numpy as np


def pack_islands(islands, padding):
    ordered = sorted(islands, key=lambda island: island["id"])
    output = []
    height = (1 - padding * (len(islands) + 1)) / len(islands)
    for index, island in enumerate(ordered):
        result = island["result"]
        uv = np.array(result["uv"], dtype=float)
        if np.ptp(uv[:, 0]) > np.ptp(uv[:, 1]):
            uv = uv[:, [1, 0]]
        uv -= uv.min(axis=0)
        scale = min((1 - 2 * padding) / np.ptp(uv[:, 0]), height / np.ptp(uv[:, 1]))
        uv = uv * scale + [padding, padding + index * (height + padding)]
        identity = islands[index]["id"]
        corners = []
        for corner in result["corners"]:
            row = dict(corner)
            row["u"], row["v"] = uv[row["uv_vertex"]].tolist()
            row["island_id"] = identity
            corners.append(row)
        output.append(
            dict(
                id=identity, uv=uv.tolist(), corners=corners, scale=scale, source=result
            )
        )
    return dict(scale=output[0]["scale"], padding=padding, islands=output)
