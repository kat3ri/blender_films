"""The filmmaking API — the only module that speaks Blender.

Everything above this layer (shot specs, the compiler, the importers) deals in
directing terms: characters, positions, poses, camera moves. This module is the
single place those become ``bpy`` calls, which is what keeps Blender a
replaceable implementation detail rather than the product.

Runs inside Blender only. Import it from a host-side process and it will fail
on ``import bpy``, by design.

Known Blender API gotchas hit while building this (recorded so a future
session doesn't silently re-break one):

* **Workbench ``shading.color_type`` must be ``"MATERIAL"``, not the default
  ``"OBJECT"``.** ``OBJECT`` colours the whole object from ``obj.color`` and
  silently ignores every per-face material — a per-part ``"color"`` override
  in an asset JSON will build correctly and render as nothing. See
  :func:`configure_render`.
* **``matrix_world`` does not reflect an assigned ``.location`` /
  ``.rotation_euler`` (or parenting) until the dependency graph is
  evaluated** — reading it right after building the scene returns stale,
  usually-identity transforms even for an unparented object. Call
  ``bpy.context.view_layer.update()`` first. See :func:`scene_manifest`.
* **bmesh face indices go stale after ``bmesh.ops`` calls add geometry** —
  call ``bm.faces.ensure_lookup_table()`` again before indexing into
  ``bm.faces``, both before measuring a face count and after adding more
  geometry. See the per-part material grouping in :func:`build_proxy`.
* **There is no native capsule primitive.** Build one from a cylinder plus
  two ``uv_sphere`` caps composed with ``Matrix.Translation @ Matrix.Rotation
  @ Matrix.Diagonal`` — see :func:`_emit_part`.
* **A movie-strip render filepath gets Blender's own frame-range suffix
  appended** — it will not land at the exact path you set. Render to a
  scratch stem and move the result into place. See
  :func:`render_control_video`.
* **An imported glTF's `material.diffuse_color` is left at Blender's unset
  default `(0.8, 0.8, 0.8)`**, even though the Principled BSDF's Base Color
  is correctly wired to the source image texture — Workbench `MATERIAL`
  shading reads only `diffuse_color` and never samples that texture, so an
  imported real asset renders as flat mid-grey unless something explicitly
  derives and assigns a representative colour first. Confirmed by an actual
  render, not assumed — see :func:`_flatten_mesh_materials`.
"""

from __future__ import annotations

import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix

from . import mocap
from . import mocap_bvh
from . import rig
from .motion import POSE_TABLE, look_at_euler, pad3

UNIT_SPHERE_SEGMENTS = (16, 8)
UNIT_CYLINDER_SEGMENTS = 16


# ---------------------------------------------------------------------------
# scene setup
# ---------------------------------------------------------------------------


def new_scene(fps=12, duration_seconds=6.0):
    """Wipe the file and start an empty scene sized to the shot."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = int(fps)
    scene.frame_start = 1
    scene.frame_end = max(1, int(round(duration_seconds * fps)))
    scene.frame_set(1)

    # A world background so nothing renders against pure black.
    world = bpy.data.worlds.new("PrevisWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.06, 0.08, 1.0)
    scene.world = world
    return scene


def add_key_light():
    """A sun, only needed if the shot renders through EEVEE instead of Workbench."""
    light_data = bpy.data.lights.new(name="KeyLight", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("KeyLight", light_data)
    light.rotation_euler = Euler((math.radians(50.0), 0.0, math.radians(35.0)), "XYZ")
    bpy.context.collection.objects.link(light)
    return light


# ---------------------------------------------------------------------------
# proxy geometry
# ---------------------------------------------------------------------------


def _part_matrix(position, size, rotation_deg=(0.0, 0.0, 0.0)):
    translation = Matrix.Translation(position)
    rotation = Euler([math.radians(r) for r in rotation_deg], "XYZ").to_matrix().to_4x4()
    scale = Matrix.Diagonal((size[0], size[1], size[2], 1.0))
    return translation @ rotation @ scale


def _emit_unit_shape(bm, shape, matrix):
    """Add one primitive whose unit form fills a 1x1x1 box at the origin."""
    if shape in ("box", "plane"):
        bmesh.ops.create_cube(bm, size=1.0, matrix=matrix)
    elif shape == "cylinder":
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=UNIT_CYLINDER_SEGMENTS,
            radius1=0.5,
            radius2=0.5,
            depth=1.0,
            matrix=matrix,
        )
    elif shape == "cone":
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=UNIT_CYLINDER_SEGMENTS,
            radius1=0.5,
            radius2=0.0,
            depth=1.0,
            matrix=matrix,
        )
    elif shape == "uv_sphere":
        bmesh.ops.create_uvsphere(
            bm,
            u_segments=UNIT_SPHERE_SEGMENTS[0],
            v_segments=UNIT_SPHERE_SEGMENTS[1],
            radius=0.5,
            matrix=matrix,
        )
    else:
        raise ValueError(f"unsupported proxy shape {shape!r}")


def _emit_part(bm, part):
    """Add one asset part to the mesh being built."""
    shape = part.get("shape", "box")
    position = list(part.get("position", [0.0, 0.0, 0.0]))
    if len(position) == 2:
        position.append(0.0)
    size = list(part.get("size", [1.0, 1.0, 1.0]))
    if len(size) == 2:
        size.append(0.02)
    if shape == "plane":
        size[2] = min(size[2], 0.02) if size[2] else 0.02
    rotation = part.get("rotation_deg", [0.0, 0.0, 0.0])
    base = _part_matrix(position, size, rotation)

    if shape != "capsule":
        _emit_unit_shape(bm, shape, base)
        return

    # A capsule is a cylinder capped with two hemispheres. Sub-shapes are
    # expressed in the part's unit space so the base matrix still scales them.
    width = min(size[0], size[1])
    height = max(size[2], width + 1e-4)
    barrel = (height - width) / height
    cap_offset = (height - width) / (2.0 * height)
    cap_scale = (width / size[0], width / size[1], width / height)

    _emit_unit_shape(bm, "cylinder", base @ Matrix.Diagonal((1.0, 1.0, barrel, 1.0)))
    for direction in (1.0, -1.0):
        cap = (
            base
            @ Matrix.Translation((0.0, 0.0, direction * cap_offset))
            @ Matrix.Diagonal((cap_scale[0], cap_scale[1], cap_scale[2], 1.0))
        )
        _emit_unit_shape(bm, "uv_sphere", cap)


def _flat_material(name, color):
    """A flat unlit-ish material, so EEVEE renders match Workbench's clarity."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = (color[0], color[1], color[2], 1.0)
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        principled.inputs["Roughness"].default_value = 0.85
        if "Specular IOR Level" in principled.inputs:
            principled.inputs["Specular IOR Level"].default_value = 0.1
    return material


def build_proxy(name, asset, color_override=None):
    """Build an object from an asset's part list — primitives, real imported
    meshes, or a mix of both.

    A part may carry its own ``color``, overriding the asset's default for
    just that part — a shelf board can read lighter than its backing, a
    bottle can read a different colour than the shelf it sits on. Primitive
    parts are grouped by colour into material slots on one mesh (not one mesh
    per part), so per-part colour costs material slots, not draw calls.

    A part with ``"shape": "mesh"`` is a real imported asset (see
    :func:`_import_mesh_part`) rather than procedural geometry, and cannot
    share the bmesh merge primitives use — Blender's importers create their
    own object(s). When an asset mixes primitives and mesh parts, or is
    mesh-only, the root returned is an Empty with the primitive mesh (if any)
    and every imported mesh part parented under it as children, so the whole
    thing still moves as one object under ``place_character``/``place_prop``.
    """
    default_color = list(color_override or asset.get("color", [0.6, 0.6, 0.6]))
    parts = asset.get("parts", [])
    primitive_parts = [p for p in parts if p.get("shape") != "mesh"]
    mesh_parts = [p for p in parts if p.get("shape") == "mesh"]

    obj = None
    if primitive_parts or not mesh_parts:
        mesh = bpy.data.meshes.new(f"{name}_mesh")
        bm = bmesh.new()
        slot_for_color = {}
        materials = []
        for part in primitive_parts:
            color = tuple(part.get("color", default_color))
            if color not in slot_for_color:
                slot_for_color[color] = len(materials)
                materials.append(_flat_material(f"{name}_mat{len(materials)}", color))
            bm.faces.ensure_lookup_table()
            before = len(bm.faces)
            _emit_part(bm, part)
            bm.faces.ensure_lookup_table()
            slot = slot_for_color[color]
            for face in bm.faces[before:]:
                face.material_index = slot

        bm.to_mesh(mesh)
        bm.free()
        mesh.shade_flat()
        for material in materials:
            mesh.materials.append(material)

        obj = bpy.data.objects.new(name, mesh)
        obj.color = (default_color[0], default_color[1], default_color[2], 1.0)
        bpy.context.collection.objects.link(obj)

    if mesh_parts:
        if obj is None:
            obj = bpy.data.objects.new(name, None)
            obj.empty_display_size = 0.05
            bpy.context.collection.objects.link(obj)
        for index, part in enumerate(mesh_parts):
            _import_mesh_part(f"{name}_mesh{index}", part, obj, default_color)

    return obj


def _import_mesh_part(name, part, parent, default_color):
    """Import a real asset file as a child of ``parent``.

    Supports glTF/GLB, FBX and OBJ — whichever a source (Poly Haven today)
    happens to export. Multiple imported objects are joined into one, so a
    mesh part behaves like every other part: one logical object with one
    transform. Materials are always flattened to a solid colour (see
    :func:`_flatten_mesh_materials`) — Workbench's ``MATERIAL`` shading only
    ever reads ``material.diffuse_color`` and never samples an image texture,
    so an imported asset's real PBR textures render as flat mid-grey
    (Blender's unset default) unless something derives a representative
    colour and assigns it explicitly. Confirmed by an actual render before
    writing this, not assumed.
    """
    path = Path(part["file"])
    if not path.is_file():
        raise FileNotFoundError(f"mesh part file not found: {path}")

    before = set(bpy.data.objects.keys())
    suffix = path.suffix.lower()
    if suffix in (".gltf", ".glb"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"unsupported mesh file type {path.suffix!r}: {path}")

    imported = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in before]
    mesh_objs = [o for o in imported if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"import of {path} produced no mesh objects")

    # Non-mesh leftovers (an Empty root node, typically) must be collected
    # *before* any join() runs: join() deletes every non-active mesh object
    # outright, so a reference collected afterward and then touched (even
    # just `.name` on it) raises "StructRNA of type Object has been removed"
    # -- hit for real on a multi-part asset (large_castle_door: door leaf,
    # hinges and frame import as separate meshes) even though it never
    # showed up on the single-mesh assets tried first.
    non_mesh_leftover = [o for o in imported if o.type != "MESH"]

    if len(mesh_objs) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()
        mesh_obj = bpy.context.view_layer.objects.active
    else:
        mesh_obj = mesh_objs[0]

    for o in non_mesh_leftover:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)

    override_color = part.get("color")
    if override_color is not None:
        color = list(override_color)
        _replace_materials(mesh_obj, color)
    else:
        color = _flatten_mesh_materials(mesh_obj, default_color)
    mesh_obj.color = (color[0], color[1], color[2], 1.0)

    mesh_obj.name = name
    mesh_obj.parent = parent
    mesh_obj.location = tuple(pad3(part.get("position", [0.0, 0.0, 0.0])))
    rotation = part.get("rotation_deg", [0.0, 0.0, 0.0])
    mesh_obj.rotation_euler = Euler([math.radians(r) for r in rotation], "XYZ")
    scale = part.get("scale", 1.0)
    mesh_obj.scale = (
        (scale, scale, scale) if isinstance(scale, (int, float)) else tuple(scale)
    )
    return mesh_obj


def _flatten_mesh_materials(obj, fallback_color):
    """Replace every material on ``obj`` with a flat colour sampled from its
    original base-colour texture, falling back to ``fallback_color`` if a
    slot has no texture to sample. Keeps an imported real asset visually
    consistent with every hand-built primitive around it, instead of
    rendering as flat mid-grey (see :func:`_import_mesh_part`)."""
    representative = None
    for slot_index, material in enumerate(list(obj.data.materials)):
        color = fallback_color
        if material and material.use_nodes:
            principled = material.node_tree.nodes.get("Principled BSDF")
            base = principled.inputs.get("Base Color") if principled else None
            if base is not None:
                if base.is_linked:
                    image = _find_linked_image(base)
                    sampled = _average_image_color(image) if image else None
                    if sampled:
                        color = sampled
                else:
                    color = list(base.default_value)[:3]
        obj.data.materials[slot_index] = _flat_material(f"{obj.name}_flat{slot_index}", color)
        if representative is None:
            representative = color
    if representative is None:
        obj.data.materials.append(_flat_material(f"{obj.name}_flat0", fallback_color))
        representative = fallback_color
    return representative


def _find_linked_image(socket):
    for link in socket.links:
        if link.from_node.type == "TEX_IMAGE" and link.from_node.image:
            return link.from_node.image
    return None


def _average_image_color(image, sample_every=97):
    """A texture's average colour via strided sampling.

    A one-time per-asset cost paid at import, not per frame, so a plain
    Python loop over ``image.pixels`` is fine — no numpy, no pip dependency.
    ``sample_every`` (in pixels, not floats) trades accuracy for speed; 97 is
    an arbitrary prime chosen only to avoid resonating with square texture
    dimensions or tiling patterns.
    """
    try:
        pixels = image.pixels[:]
    except Exception:
        return None
    channels = max(1, image.channels or 4)
    step = channels * sample_every
    total = [0.0, 0.0, 0.0]
    count = 0
    for i in range(0, len(pixels) - channels + 1, step):
        total[0] += pixels[i]
        total[1] += pixels[i + 1] if channels > 1 else pixels[i]
        total[2] += pixels[i + 2] if channels > 2 else pixels[i]
        count += 1
    return [c / count for c in total] if count else None


def _replace_materials(obj, color):
    material = _flat_material(f"{obj.name}_flat", color)
    if not obj.data.materials:
        obj.data.materials.append(material)
        return
    for slot_index in range(len(obj.data.materials)):
        obj.data.materials[slot_index] = material


def load_set(asset, name=None):
    """Place a set/location asset at the world origin."""
    obj = build_proxy(name or f"SET_{asset.get('asset_id', 'set')}", asset)
    obj.location = (0.0, 0.0, 0.0)
    return obj


def place_character(object_id, asset, position, facing_deg=0.0):
    obj = build_proxy(f"CHAR_{object_id}", asset)
    obj.location = tuple(position)
    obj.rotation_euler = Euler((0.0, 0.0, math.radians(facing_deg)), "XYZ")
    return obj


def place_prop(object_id, asset, position, facing_deg=0.0):
    obj = build_proxy(f"PROP_{object_id}", asset)
    obj.location = tuple(position)
    obj.rotation_euler = Euler((0.0, 0.0, math.radians(facing_deg)), "XYZ")
    return obj


def add_ground(size_m=(12.0, 12.0), grid=True, color=(0.22, 0.23, 0.25)):
    """A floor plus a 1m reference grid.

    The grid is real geometry rather than a viewport overlay because it has to
    survive into the rendered video — it is what makes distance travelled and
    camera movement legible to the downstream model.
    """
    width, depth = float(size_m[0]), float(size_m[1])
    mesh = bpy.data.meshes.new("ground_mesh")
    bm = bmesh.new()
    _emit_unit_shape(bm, "box", _part_matrix((0.0, 0.0, -0.01), (width, depth, 0.02)))
    bm.to_mesh(mesh)
    bm.free()
    ground = bpy.data.objects.new("GROUND", mesh)
    ground.color = (color[0], color[1], color[2], 1.0)
    mesh.materials.append(_flat_material("ground_mat", color))
    bpy.context.collection.objects.link(ground)

    if not grid:
        return ground

    line_color = (0.34, 0.36, 0.40)
    grid_mesh = bpy.data.meshes.new("grid_mesh")
    bm = bmesh.new()
    thickness = 0.025
    for x in range(int(-width // 2), int(width // 2) + 1):
        _emit_unit_shape(
            bm, "box", _part_matrix((x, 0.0, 0.005), (thickness, depth, 0.012))
        )
    for y in range(int(-depth // 2), int(depth // 2) + 1):
        _emit_unit_shape(
            bm, "box", _part_matrix((0.0, y, 0.005), (width, thickness, 0.012))
        )
    bm.to_mesh(grid_mesh)
    bm.free()
    grid_obj = bpy.data.objects.new("GROUND_GRID", grid_mesh)
    grid_obj.color = (*line_color, 1.0)
    grid_mesh.materials.append(_flat_material("grid_mat", line_color))
    bpy.context.collection.objects.link(grid_obj)
    return ground


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------


def _set_linear_interpolation(obj):
    animation = obj.animation_data
    if not animation or not animation.action:
        return
    for fcurve in animation.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def animate_character(obj, track, fps):
    """Stamp a motion track onto a character proxy.

    Poses are applied as a vertical squash plus a forward lean — crude, but it
    reads as crouching/reaching in silhouette, which is all the control video
    needs to convey.
    """
    for key in track.keys:
        frame = 1 + int(round(key["t"] * fps))
        squash, lean_deg = POSE_TABLE.get(key["pose"], POSE_TABLE["stand"])
        obj.location = tuple(key["position"])
        obj.rotation_euler = Euler(
            (0.0, math.radians(lean_deg), math.radians(key["facing_deg"])), "XYZ"
        )
        obj.scale = (1.0, 1.0, squash)
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_euler", frame=frame)
        obj.keyframe_insert("scale", frame=frame)
    _set_linear_interpolation(obj)
    return obj


def build_rigged_proxy(name, asset):
    """Build an articulated humanoid from the rest skeleton in ``rig``.

    Joints are parented empties rather than a Blender armature: no edit-mode
    round trip (which is the fragile part of doing this headless), and because
    BVH is itself a hierarchy of named joints with per-frame Euler rotations,
    driving these empties from a mocap clip later is a name lookup rather than
    an armature retarget.
    """
    scale = rig.scale_for(asset)
    colour = list(asset.get("color", [0.6, 0.6, 0.6]))
    material = _flat_material(f"{name}_mat", colour)

    root = bpy.data.objects.new(name, None)
    root.empty_display_size = 0.12
    bpy.context.collection.objects.link(root)

    joints = {}
    for joint_name, spec in rig.JOINTS.items():
        empty = bpy.data.objects.new(f"{name}_{joint_name}", None)
        empty.empty_display_size = 0.05
        bpy.context.collection.objects.link(empty)
        empty.parent = joints[spec["parent"]] if spec["parent"] else root
        empty.location = tuple(v * scale for v in spec["offset"])
        joints[joint_name] = empty

    # Limb masses span from one joint to another.
    for joint_name, to_joint, shape, thickness in rig.BONES:
        child_offset = [v * scale for v in rig.rest_offset(joint_name, to_joint)]
        length = math.dist((0.0, 0.0, 0.0), child_offset)
        if length < 1e-4:
            continue
        width = thickness * scale
        mesh = bpy.data.meshes.new(f"{name}_{joint_name}_mesh")
        bm = bmesh.new()
        # Build along +Z, then rotate the object to point at the child.
        _emit_part(bm, {"shape": shape,
                        "position": [0.0, 0.0, length / 2.0],
                        "size": [width, width, length]})
        bm.to_mesh(mesh)
        bm.free()
        mesh.shade_flat()
        mesh.materials.append(material)
        limb = bpy.data.objects.new(f"{name}_{joint_name}_limb", mesh)
        limb.color = (colour[0], colour[1], colour[2], 1.0)
        bpy.context.collection.objects.link(limb)
        limb.parent = joints[joint_name]
        limb.rotation_euler = _direction_to_euler(child_offset)

    # Head, hands, feet.
    for joint_name, specs in rig.TIPS.items():
        for index, spec in enumerate(specs):
            mesh = bpy.data.meshes.new(f"{name}_{joint_name}_tip{index}_mesh")
            bm = bmesh.new()
            _emit_part(bm, {"shape": spec["shape"],
                            "position": [v * scale for v in spec["offset"]],
                            "size": [v * scale for v in spec["size"]],
                            "rotation_deg": spec.get("rotation_deg", [0.0, 0.0, 0.0])})
            bm.to_mesh(mesh)
            bm.free()
            mesh.shade_flat()
            mesh.materials.append(material)
            tip = bpy.data.objects.new(f"{name}_{joint_name}_tip{index}", mesh)
            tip.color = (colour[0], colour[1], colour[2], 1.0)
            bpy.context.collection.objects.link(tip)
            tip.parent = joints[joint_name]

    # An object carried in a hand (a staff, a lantern). Parented to the hand
    # joint, so it inherits the hand's per-frame keyframes for free — it swings
    # with the arm on a walk and lowers when the body kneels, no extra
    # animation. Geometry lives on the character asset in the hand's local
    # frame; crude primitives, same as any proxy.
    held = asset.get("held_prop")
    if isinstance(held, dict):
        joint_name = held.get("joint", "r_hand")
        anchor = joints.get(joint_name)
        if anchor is not None:
            held_colour = list(held.get("color", colour))
            held_mat = _flat_material(f"{name}_held_mat", held_colour)
            base = [v * scale for v in held.get("offset", [0.0, 0.0, 0.0])]
            mesh = bpy.data.meshes.new(f"{name}_held_mesh")
            bm = bmesh.new()
            for part in held.get("parts", []):
                _emit_part(bm, {
                    "shape": part.get("shape", "cylinder"),
                    "position": [base[i] + part.get("position", [0.0, 0.0, 0.0])[i] * scale
                                 for i in range(3)],
                    "size": [v * scale for v in part.get("size", [0.05, 0.05, 1.0])],
                    "rotation_deg": part.get("rotation_deg", [0.0, 0.0, 0.0]),
                })
            bm.to_mesh(mesh)
            bm.free()
            mesh.shade_flat()
            mesh.materials.append(held_mat)
            prop = bpy.data.objects.new(f"{name}_held", mesh)
            prop.color = (held_colour[0], held_colour[1], held_colour[2], 1.0)
            bpy.context.collection.objects.link(prop)
            prop.parent = anchor
            prop.rotation_euler = Euler(
                [math.radians(r) for r in held.get("rotation_deg", [0.0, 0.0, 0.0])], "XYZ"
            )

    return root, joints


def _direction_to_euler(direction):
    """Euler that points a +Z-aligned limb along ``direction``."""
    x, y, z = direction
    horizontal = math.hypot(x, y)
    if horizontal < 1e-6:
        return Euler((0.0, 0.0, 0.0), "XYZ") if z >= 0 else Euler((math.pi, 0.0, 0.0), "XYZ")
    return Euler((math.atan2(horizontal, z), 0.0, math.atan2(y, x) + math.pi / 2.0), "XYZ")


def _blend_angle_deg(a, b, w):
    """Blend two angles via the shortest rotational delta."""
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + delta * w


def _blend_xyz_deg(a, b, w):
    return [_blend_angle_deg(a[i], b[i], w) for i in range(3)]


def _apply_root_mode(position, facing_deg, segment, clip_root, runtime_state, t_s):
    """Optional root translation from clip for from_clip/blend modes.

    Clip translation units are source-dependent, usually centimeters, so
    root_scale_m defaults to 0.01.
    """
    if clip_root is None:
        return list(position)

    mode = segment.get("root_mode", "lock_xy")
    if mode == "lock_xy":
        return list(position)

    key = id(segment)
    state = runtime_state.setdefault(key, {})
    if "clip_root0" not in state:
        state["clip_root0"] = list(clip_root)
    if "world_start" not in state:
        state["world_start"] = list(position)

    root_scale_m = float(segment.get("root_scale_m", 0.01))
    dx_local = (clip_root[0] - state["clip_root0"][0]) * root_scale_m
    dy_local = (clip_root[1] - state["clip_root0"][1]) * root_scale_m
    dz_world = (clip_root[2] - state["clip_root0"][2]) * root_scale_m

    yaw = math.radians(facing_deg)
    dx_world = dx_local * math.cos(yaw) - dy_local * math.sin(yaw)
    dy_world = dx_local * math.sin(yaw) + dy_local * math.cos(yaw)
    target = [
        state["world_start"][0] + dx_world,
        state["world_start"][1] + dy_world,
        state["world_start"][2] + dz_world,
    ]

    if mode == "from_clip":
        return target
    # blend mode
    w = mocap.segment_blend_weight(segment, t_s)
    return [position[i] + (target[i] - position[i]) * w for i in range(3)]


def _mocap_overlay(angles, track, t, runtime_state):
    """Blend active mocap segment onto procedural joint angles in-place."""
    segment = track.mocap_segment_at(t)
    if not segment:
        return angles, None, None

    clip_id = segment["clip_id"]
    clip_path = runtime_state["clip_paths"].get(clip_id)
    clip = runtime_state["clips"].get(clip_id)
    if clip is None:
        try:
            clip_path = mocap.resolve_clip_path(clip_id)
            clip = mocap_bvh.load_bvh(clip_path)
            runtime_state["clips"][clip_id] = clip
            runtime_state["clip_paths"][clip_id] = clip_path
        except (FileNotFoundError, ValueError) as exc:
            if clip_id not in runtime_state["warned_missing"]:
                runtime_state["warned_missing"].add(clip_id)
                print(f"[previs] WARNING     mocap clip {clip_id!r} skipped: {exc}")
            return angles, None, None

    clip_t = mocap.clip_time_for_segment(segment, t, clip.duration_seconds)
    source_rot = clip.sample_joint_rotations(clip_t)
    mapped = mocap.map_rotations(
        source_rot,
        mocap.canonical_joint_map(segment.get("joint_map")),
        source_up_axis=segment.get("source_up_axis", "y"),
    )
    w = mocap.segment_blend_weight(segment, t)
    if w > 1e-6:
        for joint_name, source_angle in mapped.items():
            current = angles.get(joint_name)
            if current is None:
                continue
            angles[joint_name] = _blend_xyz_deg(current, source_angle, w)
    return angles, segment, clip.sample_root_translation(clip_t)


def animate_rigged_character(root, joints, track, asset, fps, frame_end):
    """Keyframe an articulated character: root transform plus every joint."""
    gait = rig.resolve_gait(asset)
    scale = rig.scale_for(asset)
    hips_rest_z = rig.JOINTS["hips"]["offset"][2] * scale
    runtime_state = {
        "clips": {},
        "clip_paths": {},
        "warned_missing": set(),
        "root_segments": {},
    }

    for frame in range(1, frame_end + 1):
        t = (frame - 1) / fps
        position, facing, _ = track.sample(t)
        pose, previous_pose, pose_u = track.pose_at(t)
        angles, root_drop = rig.evaluate(
            track.distance_at(t), track.speed_at(t), pose, gait,
            previous_pose=previous_pose, pose_u=pose_u,
        )

        angles, segment, clip_root = _mocap_overlay(angles, track, t, runtime_state)
        if segment is not None:
            key = id(segment)
            state = runtime_state["root_segments"].setdefault(key, {"last_t": t})
            state["last_t"] = t
            position = _apply_root_mode(
                position, facing, segment, clip_root, runtime_state["root_segments"], t
            )

        root.location = tuple(position)
        root.rotation_euler = Euler((0.0, 0.0, math.radians(facing)), "XYZ")
        root.keyframe_insert("location", frame=frame)
        root.keyframe_insert("rotation_euler", frame=frame)

        hips = joints["hips"]
        hips.location = (0.0, 0.0, hips_rest_z - root_drop * scale)
        hips.keyframe_insert("location", frame=frame)

        for joint_name, empty in joints.items():
            euler = angles.get(joint_name)
            empty.rotation_euler = Euler(
                tuple(math.radians(a) for a in euler) if euler else (0.0, 0.0, 0.0), "XYZ"
            )
            empty.keyframe_insert("rotation_euler", frame=frame)

    for obj in [root] + list(joints.values()):
        _set_linear_interpolation(obj)
    return root


def create_camera(lens_mm=35.0, name="CAMERA"):
    camera_data = bpy.data.cameras.new(name)
    camera_data.lens = float(lens_mm)
    camera_data.clip_start = 0.05
    camera_data.clip_end = 500.0
    camera = bpy.data.objects.new(name, camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def animate_camera(camera, keys):
    """Apply pre-computed per-frame camera keys.

    Rotation is baked from look-at maths rather than a Track To constraint:
    fully deterministic, and it lets pan/tilt and target-tracking mix in one
    timeline without constraint influence juggling.
    """
    for key in keys:
        camera.location = tuple(key.position)
        camera.rotation_euler = Euler(tuple(key.rotation_euler), "XYZ")
        camera.keyframe_insert("location", frame=key.frame)
        camera.keyframe_insert("rotation_euler", frame=key.frame)
    _set_linear_interpolation(camera)
    return camera


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def configure_render(scene, engine="WORKBENCH", resolution=(960, 540), fps=12):
    """Cheap, high-legibility settings — silhouettes over beauty."""
    scene.render.engine = "BLENDER_WORKBENCH" if engine == "WORKBENCH" else "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)
    scene.render.film_transparent = False

    if engine == "WORKBENCH":
        shading = scene.display.shading
        shading.light = "STUDIO"
        # MATERIAL (not OBJECT) so per-part colour overrides on a single mesh
        # -- e.g. bottles vs. the shelf they sit on -- actually render. OBJECT
        # mode reads only obj.color for the whole object and ignores the
        # per-face materials build_proxy() assigns, which silently no-opped
        # every per-part "color" override up to this point.
        shading.color_type = "MATERIAL"
        shading.show_object_outline = True
        shading.show_shadows = True
        shading.show_cavity = True
        shading.cavity_type = "BOTH"
        scene.display.render_aa = "8"

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    return scene


def render_control_video(scene, output_path):
    """Render the animation to a single video file at exactly ``output_path``.

    Blender decorates movie filepaths with the rendered frame range, so this
    renders to a scratch stem and moves the result into place.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.parent / f"_{output_path.stem}_render"

    scene.render.filepath = str(stem)
    scene.render.use_file_extension = True
    bpy.ops.render.render(animation=True)

    produced = sorted(
        output_path.parent.glob(f"_{output_path.stem}_render*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not produced:
        raise RuntimeError(f"Blender produced no output for {output_path}")
    if output_path.exists():
        output_path.unlink()
    produced[-1].replace(output_path)
    for leftover in produced[:-1]:
        leftover.unlink()
    return output_path


def render_preview_frame(scene, output_path, frame=1):
    """Render exactly one still frame to a PNG — a cheap sanity check.

    Catches a wrong blocking, occlusion, or missing asset before paying for a
    full multi-second animated render and an FFmpeg encode. Not a scaled-down
    version of the real render: same engine, same resolution, same materials
    — only the frame count differs, so what it shows is what the full render
    will show at that instant.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_format = scene.render.image_settings.file_format
    scene.frame_set(int(frame))
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path.with_suffix(""))
    bpy.ops.render.render(write_still=True)
    scene.render.image_settings.file_format = original_format

    produced = output_path.with_suffix("").with_suffix(".png")
    if produced != output_path and produced.is_file():
        if output_path.exists():
            output_path.unlink()
        produced.replace(output_path)
    if not output_path.is_file():
        raise RuntimeError(f"Blender produced no still for {output_path}")
    return output_path


def scene_manifest(scene):
    """A cheap, fast textual summary of every object actually in the scene.

    The equivalent of a live-session `get_scene_info()` query, but computed
    from the one-shot scene this process just built rather than round-tripped
    over a socket. Meant to be printed and read by a human (or by me) before
    trusting a render: catches "only 3 objects came out of a 6-fixture set"
    class bugs for free, without ever opening the file.
    """
    # matrix_world does not reflect assigned .location/.rotation_euler (nor
    # parenting) until the dependency graph is evaluated -- reading it right
    # after building the scene silently returns stale (usually identity)
    # transforms, which is exactly how the camera briefly reported (0,0,0)
    # here despite having no parent at all.
    bpy.context.view_layer.update()

    lines = []
    for obj in sorted(scene.objects, key=lambda o: o.name):
        if obj.type not in ("MESH", "CAMERA"):
            continue
        # World-space, not obj.location: rig limbs are parented to joint
        # empties with a local offset of zero (their shape is baked into the
        # mesh itself), so obj.location on any of them reads (0,0,0) --
        # correct locally, useless for a "where did this actually land" check.
        world = obj.matrix_world.translation
        loc = tuple(round(v, 2) for v in world)
        if obj.type == "CAMERA":
            lines.append(f"  {obj.name:<28} CAMERA  loc={loc}")
            continue
        dims = tuple(round(v, 2) for v in obj.dimensions)
        faces = len(obj.data.polygons) if obj.data else 0
        lines.append(f"  {obj.name:<28} MESH    loc={loc} dims={dims} faces={faces}")
    return "\n".join(lines)


def _lerp3(a, b, u):
    return [a[i] + (b[i] - a[i]) * u for i in range(3)]


def draw_camera_path(camera_keys, sample_every=3, marker_size=0.22):
    """Lay a visible trail along the camera's actual computed trajectory —
    small spheres at sampled positions, connected by thin segments, green at
    the start shading to red at the end so direction of travel is unambiguous
    at a glance. Also drops a slightly larger marker at every *individual*
    move's boundary (a still-common source of "wrong" paths: two moves that
    don't actually connect where you assumed).

    This is the only way to actually see a camera path rather than infer it
    from what the camera itself renders — the render-and-look loop this whole
    system runs on can't diagnose "goes through a wall" or "arcs the wrong
    way" because a shot's own camera obviously never sees itself.
    """
    if not camera_keys:
        return []

    green, red = (0.15, 0.85, 0.25), (0.85, 0.15, 0.15)
    sampled = camera_keys[::sample_every]
    if sampled[-1] is not camera_keys[-1]:
        sampled.append(camera_keys[-1])

    markers = []
    for index, key in enumerate(sampled):
        u = index / max(1, len(sampled) - 1)
        color = _lerp3(green, red, u)
        mesh = bpy.data.meshes.new(f"campath_dot{index}_mesh")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(
            bm, u_segments=8, v_segments=6, radius=marker_size / 2.0,
            matrix=Matrix.Translation(key.position),
        )
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(f"campath_dot{index}", mesh)
        obj.color = (color[0], color[1], color[2], 1.0)
        mesh.materials.append(_flat_material(f"campath_dot{index}_mat", color))
        bpy.context.collection.objects.link(obj)
        markers.append(obj)

        if index > 0:
            start, end = sampled[index - 1].position, key.position
            length = math.dist(start, end)
            if length > 1e-4:
                mid = _lerp3(start, end, 0.5)
                seg_mesh = bpy.data.meshes.new(f"campath_seg{index}_mesh")
                bm = bmesh.new()
                bmesh.ops.create_cube(
                    bm, size=1.0,
                    matrix=Matrix.Translation(mid) @ _direction_to_euler(
                        [end[i] - start[i] for i in range(3)]
                    ).to_matrix().to_4x4() @ Matrix.Diagonal(
                        (marker_size * 0.35, marker_size * 0.35, length, 1.0)
                    ),
                )
                bm.to_mesh(seg_mesh)
                bm.free()
                seg = bpy.data.objects.new(f"campath_seg{index}", seg_mesh)
                seg.color = (color[0], color[1], color[2], 1.0)
                seg_mesh.materials.append(_flat_material(f"campath_seg{index}_mat", color))
                bpy.context.collection.objects.link(seg)
                markers.append(seg)

    return markers


def add_observer_camera(focus_points, lens_mm=24.0, mode="angle"):
    """A static camera positioned from *outside* the shot, framing every
    point in ``focus_points`` (camera path positions plus scene geometry
    bounds) — the actual point of a path visualization, since the shot's own
    camera can never show where it itself is.

    ``mode``: "top" looks straight down (clearest read of left/right sweep
    and whether the path clips geometry in plan view); "angle" is a 3/4
    elevated view (clearer read of altitude change, e.g. a dolly's climb or
    descent, which a top-down view flattens away entirely).
    """
    xs = [p[0] for p in focus_points]
    ys = [p[1] for p in focus_points]
    zs = [p[2] for p in focus_points]
    center = [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]
    radius = max(
        1.0,
        max(math.dist([x, y], center[:2]) for x, y in zip(xs, ys)),
        (max(zs) - min(zs)) / 2.0,
    )

    camera = create_camera(lens_mm, name="OBSERVER_CAMERA")
    if mode == "top":
        camera.location = (center[0], center[1], max(zs) + radius * 1.6 + 2.0)
        aim = center
    else:
        distance = radius * 2.3 + 3.0
        camera.location = (
            center[0] - distance * 0.6,
            center[1] - distance * 0.9,
            max(zs) + radius * 0.7 + 2.0,
        )
        aim = center
    camera.rotation_euler = Euler(tuple(look_at_euler(list(camera.location), aim)), "XYZ")
    bpy.context.scene.camera = camera
    return camera
