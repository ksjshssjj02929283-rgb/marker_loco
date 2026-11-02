#!/usr/bin/env python3
"""Derive the gripper TCP from the Onshape STEP assembly (``vis_assembly.step``).

The STEP file is the source of truth for the *rigid* part of the gripper: the
marker cube (``cube``) and the fingertips (``soft_gripper_finger``).  It does
NOT contain the ArUco stickers, so the final ``T_reference_TCP`` still needs two
physical facts that only the person who mounted the markers knows:

  * which cube face carries the reference marker (ID 2), and
  * the sticker's in-plane rotation on that face.

This script reads the assembly geometry, reports the cube faces and the
fingertip midpoint, and writes a ``tcp_transform`` JSON whose translation is

    p_ref = (TCP_assembly - id2_face_center_assembly)

under the identity-rotation convention already used by the previous physical
estimate (``configs/tcp_transform.json``): the ID-2 marker axes are assumed
aligned with the gripper axes.  That convention is exactly right only when ID 2
is mounted on the +Z face (or another face with the same normal), so verify the
emitted axes on a static RGB frame before trusting it for control.

Example:
    python scripts/cad_tcp_from_step.py \\
      --step ~/下载/vis_assembly.step \\
      --id2-face +z \\
      --tcp-output configs/tcp_transform_cad.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# STEP (ISO-10303-21) parsing — only the entities this task needs.
# --------------------------------------------------------------------------- #

_REF_RE = re.compile(r"#(\d+)")
_POINT_RE = re.compile(r"CARTESIAN_POINT\('([^']*)',\(([^)]*)\)\)")
_DIRECTION_RE = re.compile(r"DIRECTION\('([^']*)',\(([^)]*)\)\)")
_AXIS2_RE = re.compile(r"AXIS2_PLACEMENT_3D\('([^']*)',#(\d+),#(\d+)(?:,#(\d+))?\)")
_BREP_RE = re.compile(r"MANIFOLD_SOLID_BREP\('([^']*)',#(\d+)\)")
_SHAPEREP_RE = re.compile(r"SHAPE_REPRESENTATION\('([^']*)'")
_RELSHIP_RE = re.compile(
    r"REPRESENTATION_RELATIONSHIP\('([^']*)','([^']*)',#(\d+),#(\d+)\)\s*"
    r"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\(#(\d+)\)"
)
_IDT_RE = re.compile(r"ITEM_DEFINED_TRANSFORMATION\('([^']*)','([^']*)',#(\d+),#(\d+)\)")
_VERTEX_RE = re.compile(r"VERTEX_POINT\('([^']*)',#(\d+)\)")


def _parse_entities(path: Path) -> dict[int, str]:
    text = " ".join(line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines())
    entities: dict[int, str] = {}
    index, length = 0, len(text)
    while index < length:
        match = re.match(r"#(\d+)=", text[index:])
        if not match:
            index += 1
            continue
        entity_id = int(match.group(1))
        cursor = index + match.end()
        depth, end = 0, cursor
        while end < length:
            char = text[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == ";" and depth == 0:
                break
            end += 1
        entities[entity_id] = text[cursor:end].strip()
        index = end + 1
    return entities


def _point(entities: dict[int, str], entity_id: int) -> list[float] | None:
    body = entities.get(entity_id)
    if not body:
        return None
    match = _POINT_RE.search(body)
    return [float(part) for part in match.group(2).split(",")] if match else None


def _direction(entities: dict[int, str], entity_id: int) -> list[float] | None:
    body = entities.get(entity_id)
    if not body:
        return None
    match = _DIRECTION_RE.search(body)
    return [float(part) for part in match.group(2).split(",")] if match else None


def _axis2_matrix(entities: dict[int, str], entity_id: int) -> np.ndarray | None:
    body = entities.get(entity_id)
    if not body:
        return None
    match = _AXIS2_RE.search(body)
    if not match:
        return None
    origin = _point(entities, int(match.group(2)))
    z_axis = _direction(entities, int(match.group(3)))
    x_ref = _direction(entities, int(match.group(4))) if match.group(4) else None
    if origin is None or z_axis is None:
        return None
    z_axis = np.asarray(z_axis, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    if x_ref is None:
        x_axis = np.asarray([1.0, 0.0, 0.0] if abs(z_axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    else:
        x_axis = np.asarray(x_ref, dtype=float)
    x_axis -= x_axis.dot(z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    transform = np.eye(4)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin
    return transform


def _refs(body: str) -> list[int]:
    return [int(x) for x in _REF_RE.findall(body)]


def _subtree(entities: dict[int, str], start: int) -> set[int]:
    seen: set[int] = set()
    stack = [start]
    while stack:
        entity_id = stack.pop()
        if entity_id in seen:
            continue
        seen.add(entity_id)
        if entity_id in entities:
            stack.extend(_refs(entities[entity_id]))
    return seen


def _brep_vertices(entities: dict[int, str], brep_id: int) -> np.ndarray:
    points: set[tuple[float, float, float]] = set()
    for entity_id in _subtree(entities, brep_id):
        match = _VERTEX_RE.match(entities.get(entity_id, ""))
        if not match:
            continue
        point = _point(entities, int(match.group(2)))
        if point:
            points.add(tuple(point))
    return np.asarray(list(points), dtype=float)


def _placements(entities: dict[int, str]) -> dict[str, list[np.ndarray]]:
    """Map each part name to its instance placement matrices in the assembly frame."""
    shape_rep: dict[str, int] = {}
    for entity_id, body in entities.items():
        match = _SHAPEREP_RE.match(body)
        if match and match.group(1) and match.group(1) != "vis_assembly":
            shape_rep[match.group(1)] = entity_id

    relationships: dict[int, tuple[int, int]] = {}
    for entity_id, body in entities.items():
        match = _RELSHIP_RE.search(body)
        if match:
            relationships[entity_id] = (int(match.group(3)), int(match.group(5)))

    transformations: dict[int, tuple[int, int]] = {}
    for entity_id, body in entities.items():
        match = _IDT_RE.match(body)
        if match:
            transformations[entity_id] = (int(match.group(3)), int(match.group(4)))

    placements: dict[str, list[np.ndarray]] = {}
    for name, rep_id in shape_rep.items():
        matrices = []
        for _, (rep, transform_id) in relationships.items():
            if rep != rep_id:
                continue
            target = transformations[transform_id][1]
            matrix = _axis2_matrix(entities, target)
            if matrix is not None:
                matrices.append(matrix)
        placements[name] = matrices
    return placements


# --------------------------------------------------------------------------- #
# Geometry extraction.
# --------------------------------------------------------------------------- #

FACES = {
    "+x": np.asarray([1.0, 0.0, 0.0]),
    "-x": np.asarray([-1.0, 0.0, 0.0]),
    "+y": np.asarray([0.0, 1.0, 0.0]),
    "-y": np.asarray([0.0, -1.0, 0.0]),
    "+z": np.asarray([0.0, 0.0, 1.0]),
    "-z": np.asarray([0.0, 0.0, -1.0]),
}


def marker_frame(normal: np.ndarray) -> np.ndarray:
    """Build the ID-2 marker frame (assembly -> marker rotation) for a face.

    Convention: marker +Z is the outward face normal, marker "up" is assembly
    +Z, marker +X points right (all assuming the sticker was mounted upright).
    For a +Z/-Z face this reduces to identity, matching the old config.
    """
    z_axis = normal / np.linalg.norm(normal)
    if abs(z_axis[2]) > 0.9:
        x_axis = np.asarray([1.0, 0.0, 0.0])
    else:
        x_axis = np.cross(z_axis, np.asarray([0.0, 0.0, 1.0]))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    # The marker basis vectors are expressed in the assembly frame, so they are
    # the ROWS of the assembly->marker rotation (row i = marker axis i).
    return np.vstack([x_axis, y_axis, z_axis])


def load_geometry(step: Path) -> dict:
    entities = _parse_entities(step)
    brep: dict[str, int] = {}
    for entity_id, body in entities.items():
        match = _BREP_RE.match(body)
        if match:
            brep[match.group(1)] = entity_id
    placements = _placements(entities)

    cube_vertices = _brep_vertices(entities, brep["cube"])
    cube_center = (cube_vertices.min(0) + cube_vertices.max(0)) / 2.0

    # Fingertip pads: for each soft-finger instance, take the centroid of the
    # vertices at the far tip (max assembly Y), then average the two pads.
    tip_centers = []
    for matrix in placements["soft_gripper_finger"]:
        vertices = np.asarray(
            [matrix[:3, :3] @ point + matrix[:3, 3] for point in _brep_vertices(entities, brep["soft_gripper_finger"])]
        )
        tip = vertices[np.abs(vertices[:, 1] - vertices[:, 1].max()) < 1e-9]
        tip_centers.append(tip.mean(0))
    tcp_assembly = np.mean(tip_centers, axis=0)

    face_centers = {}
    for name, normal in FACES.items():
        # Closest bounding-box face along the normal.
        extreme = cube_vertices[np.argmax(cube_vertices @ normal)]
        offset = normal * ((extreme - cube_center) @ normal)
        face_centers[name] = cube_center + offset

    return {
        "cube_center_m": cube_center,
        "cube_size_m": cube_vertices.max(0) - cube_vertices.min(0),
        "tcp_assembly_m": tcp_assembly,
        "face_centers_m": face_centers,
        "fingertip_pads_m": tip_centers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=Path, default=Path.home() / "下载" / "vis_assembly.step")
    parser.add_argument("--id2-face", choices=sorted(FACES), help="cube face carrying the reference marker (ID 2)")
    parser.add_argument("--id2-center-mm", type=str, help="explicit ID-2 plane centre in the assembly frame, 'x,y,z' mm (overrides --id2-face)")
    parser.add_argument("--tcp-mm", type=str, help="explicit TCP point in the assembly frame, 'x,y,z' mm (overrides the fingertip midpoint)")
    parser.add_argument("--tcp-output", type=Path, help="write a tcp_transform JSON")
    parser.add_argument("--reference-tag-id", type=int, default=2)
    args = parser.parse_args()

    geometry = load_geometry(args.step)

    print(f"marker cube  : centre {np.round(geometry['cube_center_m'] * 1000, 1)} mm, size {np.round(geometry['cube_size_m'] * 1000, 1)} mm")
    for pad, center in zip(("left", "right"), geometry["fingertip_pads_m"]):
        print(f"fingertip    : {pad:5s} pad centre {np.round(center * 1000, 1)} mm")
    print(f"TCP (pad mid): {np.round(geometry['tcp_assembly_m'] * 1000, 1)} mm")
    print("\ncube face centres (assembly frame, mm):")
    for name, center in geometry["face_centers_m"].items():
        print(f"  {name:>2s}: {np.round(center * 1000, 1)}")

    if args.id2_center_mm:
        id2_center = np.asarray([float(part) for part in args.id2_center_mm.split(",")]) / 1000.0
    elif args.id2_face:
        id2_center = geometry["face_centers_m"][args.id2_face]
    else:
        raise SystemExit("specify --id2-face or --id2-center-mm")

    tcp = geometry["tcp_assembly_m"] if not args.tcp_mm else np.asarray([float(part) for part in args.tcp_mm.split(",")]) / 1000.0
    translation = tcp - id2_center
    print(f"\nID-2 centre -> TCP translation: {np.round(translation * 1000, 1)} mm (|.| = {np.linalg.norm(translation) * 1000:.1f} mm)")

    rotation = marker_frame(FACES[args.id2_face]) if args.id2_face else np.eye(3)
    translation_marker = rotation @ translation

    if args.tcp_output:
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation_marker
        payload = {
            "schema_version": 1,
            "units": "m",
            "source": f"Onshape STEP {args.step.name} (id2 face {args.id2_face or 'custom'})",
            "reference_tag_id": args.reference_tag_id,
            "T_reference_TCP": transform.tolist(),
            "convention": "T_reference_TCP maps points from the TCP frame into the reference-tag (ID 2) frame: p_ref = T_reference_TCP @ p_tcp.",
            "tcp_definition": (
                "Fingertip-pad midpoint from STEP geometry. Marker frame convention: +Z is the "
                "outward face normal, sticker 'up' is assembly +Z, +X points right. Validate the "
                "projected TCP axes on a static RGB frame before use."
            ),
            "cad_geometry_mm": {
                "cube_center": (geometry["cube_center_m"] * 1000).round(1).tolist(),
                "tcp": (tcp * 1000).round(1).tolist(),
                "id2_center": (id2_center * 1000).round(1).tolist(),
            },
        }
        args.tcp_output.parent.mkdir(parents=True, exist_ok=True)
        args.tcp_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.tcp_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
