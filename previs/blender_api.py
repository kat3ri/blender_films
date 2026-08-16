"""The filmmaking API — the only module that speaks Blender.

Everything above this layer (shot specs, the compiler, the importers) deals in
directing terms: characters, positions, poses, camera moves. This module is the
single place those become ``bpy`` calls, which is what keeps Blender a
replaceable implementation detail rather than the product.

Runs inside Blender only. Import it from a host-side process and it will fail
on ``import bpy``, by design.
"""

from __future__ import annotations

import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix

from . import rig
from .motion import POSE_TABLE

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
    """Create a single mesh object from an asset's part list."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    for part in asset.get("parts", []):
        _emit_part(bm, part)
    bm.to_mesh(mesh)
    bm.free()
    mesh.shade_flat()

    obj = bpy.data.objects.new(name, mesh)
    color = list(color_override or asset.get("color", [0.6, 0.6, 0.6]))
    obj.color = (color[0], color[1], color[2], 1.0)
    mesh.materials.append(_flat_material(f"{name}_mat", color))
    bpy.context.collection.objects.link(obj)
    return obj


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
    for joint_name, spec in rig.TIPS.items():
        mesh = bpy.data.meshes.new(f"{name}_{joint_name}_tip_mesh")
        bm = bmesh.new()
        _emit_part(bm, {"shape": spec["shape"],
                        "position": [v * scale for v in spec["offset"]],
                        "size": [v * scale for v in spec["size"]]})
        bm.to_mesh(mesh)
        bm.free()
        mesh.shade_flat()
        mesh.materials.append(material)
        tip = bpy.data.objects.new(f"{name}_{joint_name}_tip", mesh)
        tip.color = (colour[0], colour[1], colour[2], 1.0)
        bpy.context.collection.objects.link(tip)
        tip.parent = joints[joint_name]

    return root, joints


def _direction_to_euler(direction):
    """Euler that points a +Z-aligned limb along ``direction``."""
    x, y, z = direction
    horizontal = math.hypot(x, y)
    if horizontal < 1e-6:
        return Euler((0.0, 0.0, 0.0), "XYZ") if z >= 0 else Euler((math.pi, 0.0, 0.0), "XYZ")
    return Euler((math.atan2(horizontal, z), 0.0, math.atan2(y, x) + math.pi / 2.0), "XYZ")


def animate_rigged_character(root, joints, track, asset, fps, frame_end):
    """Keyframe an articulated character: root transform plus every joint."""
    gait = rig.resolve_gait(asset)
    scale = rig.scale_for(asset)
    hips_rest_z = rig.JOINTS["hips"]["offset"][2] * scale

    for frame in range(1, frame_end + 1):
        t = (frame - 1) / fps
        position, facing, _ = track.sample(t)
        pose, previous_pose, pose_u = track.pose_at(t)
        angles, root_drop = rig.evaluate(
            track.distance_at(t), track.speed_at(t), pose, gait,
            previous_pose=previous_pose, pose_u=pose_u,
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
        shading.color_type = "OBJECT"  # uses each object's flat proxy colour
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
