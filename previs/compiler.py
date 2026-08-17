"""Turn a validated shot spec into filmmaking-API calls.

Deliberately thin: the schema mirrors the API, the maths lives in ``motion``,
so this is mostly dispatch and bookkeeping. Runs inside Blender.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import blender_api as api
from . import rig
from .asset_library import AssetLibrary, expand_fixtures
from .framing import expand_presets
from .motion import build_camera_keys, build_tracks, check_camera_bounds, pad3


def _assemble_scene(shot, assets_root=None, verbose=True):
    """Build the whole scene (set, props, characters, camera) and return the
    handles the render/export steps need. Shared by :func:`compile_shot` and
    :func:`compile_bundle` so both drive an identical scene.
    """
    library = AssetLibrary(assets_root)
    expand_fixtures(shot, library)

    fps = int(shot.get("fps", 12))
    duration = float(shot["duration_seconds"])
    render_settings = shot.get("render") or {}
    fps = int(render_settings.get("fps", fps))

    scene = api.new_scene(fps=fps, duration_seconds=duration)
    api.add_key_light()

    stage = shot.get("stage") or {}
    size = stage.get("size_m", [12.0, 12.0])
    api.add_ground(size_m=size, grid=stage.get("ground_grid", True))

    set_spec = shot.get("set") or {}
    if set_spec.get("asset_id"):
        api.load_set(library.get("sets", set_spec["asset_id"]))

    for prop in shot.get("props", []):
        if not isinstance(prop, dict):
            continue
        api.place_prop(
            prop["id"],
            library.get("props", prop.get("asset_id", "")),
            pad3(prop.get("position", [0, 0, 0])),
            float(prop.get("facing_deg", 0.0)),
        )

    tracks = build_tracks(shot, library)
    rigged = {}
    for character in shot.get("characters", []):
        if not isinstance(character, dict):
            continue
        asset = library.get("characters", character.get("asset_id", ""))
        track = tracks[character["id"]]
        start_position, start_facing, _ = track.sample(0.0)

        if rig.is_rigged(asset):
            root, joints = api.build_rigged_proxy(f"CHAR_{character['id']}", asset)
            api.animate_rigged_character(root, joints, track, asset, fps, scene.frame_end)
            rigged[character["id"]] = joints
        else:
            obj = api.place_character(character["id"], asset, start_position, start_facing)
            api.animate_character(obj, track, fps)

    # Framing presets resolve against where the subjects actually stand, so
    # they expand after tracks exist and before any camera maths runs. Every
    # stage downstream sees ordinary move dicts.
    n_presets = expand_presets(shot, tracks, library)

    camera_spec = shot.get("camera") or {}
    camera = api.create_camera(float(camera_spec.get("lens_mm", 35.0)))
    camera_keys = build_camera_keys(shot, tracks, library, fps)
    api.animate_camera(camera, camera_keys)
    warnings = check_camera_bounds(camera_keys, stage)

    api.configure_render(
        scene,
        engine=render_settings.get("engine", "WORKBENCH"),
        resolution=render_settings.get("resolution", [960, 544]),
        fps=fps,
    )

    if verbose:
        print(f"[previs] shot        {shot.get('shot_id')}")
        print(f"[previs] duration    {duration}s @ {fps}fps "
              f"({scene.frame_start}-{scene.frame_end})")
        print(f"[previs] characters  {[c.get('id') for c in shot.get('characters', [])]}"
              + (f"  (articulated: {list(rigged)})" if rigged else ""))
        print(f"[previs] props       {[p.get('id') for p in shot.get('props', [])]}")
        moves_desc = [m.get("_preset") or m.get("type")
                      for m in camera_spec.get("moves", [])]
        print(f"[previs] camera      {moves_desc}"
              + (f"  ({n_presets} preset(s) expanded)" if n_presets else ""))
        for kind, asset_id in library.missing:
            print(f"[previs] WARNING     no {kind[:-1]} asset {asset_id!r}; using placeholder")
        for warning in warnings:
            print(f"[previs] WARNING     {warning}")

    return {
        "scene": scene,
        "library": library,
        "tracks": tracks,
        "camera": camera,
        "camera_keys": camera_keys,
        "rigged": rigged,
        "fps": fps,
        "duration": duration,
        "render_settings": render_settings,
        "warnings": warnings,
    }


def compile_shot(shot, output_path, assets_root=None, verbose=True, preview_frame=None):
    """Build the scene described by ``shot`` and render it to ``output_path``.

    ``preview_frame``, if given, skips the full animated render entirely and
    renders just that one frame to a PNG instead — same scene, same engine,
    same resolution, so what it shows is what the real render will show at
    that instant. Also prints a scene manifest (every mesh/camera actually
    built, with dimensions and face counts). Use this to catch a wrong
    blocking, an occluding wall, or a placeholder asset before paying for a
    full multi-second render and an FFmpeg encode.
    """
    ctx = _assemble_scene(shot, assets_root=assets_root, verbose=verbose)
    scene = ctx["scene"]

    if preview_frame is not None:
        if verbose:
            print(f"[previs] scene manifest ({len(scene.objects)} objects):")
            print(api.scene_manifest(scene))
        result = api.render_preview_frame(scene, output_path, frame=preview_frame)
        if verbose:
            print(f"[previs] wrote preview frame {preview_frame} -> {result}")
        return result

    result = api.render_control_video(scene, output_path)
    if verbose:
        print(f"[previs] wrote       {result}")
    return result


def _estimate_depth_range(shot, ctx):
    """Fit a single fixed near/far depth range to the actual scene content, so
    the depth pass uses the full 0..1 range without per-frame flicker.

    A blanket 30 m far plane crushes an intimate interior into a flat bright
    band; fitting to the real camera-to-subject distances restores contrast
    while staying constant for the whole clip.
    """
    import math

    camera_keys = ctx["camera_keys"]
    tracks = ctx["tracks"]

    points = []
    for prop in shot.get("props", []):
        if isinstance(prop, dict):
            points.append(pad3(prop.get("position", [0, 0, 0])))

    distances = []
    for key in camera_keys:
        cam_pos = key.position
        sample_points = list(points)
        for track in tracks.values():
            pos, _facing, _pose = track.sample(key.t)
            # sample body base and head, so a tall figure's near and far
            # extents both influence the range
            sample_points.append([pos[0], pos[1], pos[2]])
            sample_points.append([pos[0], pos[1], pos[2] + 1.7])
        for pt in sample_points:
            d = math.sqrt(sum((cam_pos[i] - pt[i]) ** 2 for i in range(3)))
            distances.append(d)

    if not distances:
        return 0.5, 30.0

    near = max(0.1, min(distances) - 1.0)
    far = max(near + 1.0, max(distances) + 2.0)
    return near, far


def compile_bundle(shot, out_dir, assets_root=None, verbose=True,
                   generators=None, with_depth=True, with_pose=True, with_stills=True):
    """Export a full control-layer bundle for a downstream AI video generator.

    Renders the reference video and (optionally) a depth pass, captures exact
    ground-truth pose landmarks and stills, and writes the host-computable
    sidecars (camera_motion / metadata / prompts / manifest / README) via
    :mod:`previs.bundle`. The bundle layout mirrors Blockout's shot package.
    """
    from . import bundle as bundle_mod

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shot_id = shot.get("shot_id", "shot")

    ctx = _assemble_scene(shot, assets_root=assets_root, verbose=verbose)
    scene = ctx["scene"]
    fps = ctx["fps"]
    generators = tuple(generators) if generators else bundle_mod.DEFAULT_GENERATORS

    entries = {}

    # Host-computable sidecars first (exact, need no render).
    written, metadata = bundle_mod.write_sidecars(
        out_dir, shot, ctx["tracks"], ctx["camera_keys"], ctx["library"], fps,
        generators=generators, render_settings=ctx["render_settings"],
    )
    entries.update(written)
    if verbose:
        print(f"[previs] bundle      wrote {len(written)} sidecar file(s)")

    # Reference control video.
    reference = out_dir / f"{shot_id}_reference.mp4"
    api.render_control_video(scene, reference)
    entries["reference"] = reference.name
    if verbose:
        print(f"[previs] bundle      reference -> {reference.name}")

    # Ground-truth pose landmarks (exact 3D + 2D projection).
    if with_pose and ctx["rigged"]:
        pose = api.capture_pose_landmarks(
            scene, ctx["camera"], ctx["rigged"], fps, scene.frame_end,
            metadata["resolution"],
        )
        bundle_mod._dump(out_dir / "pose_landmarks.json", pose)
        entries["pose_landmarks"] = "pose_landmarks.json"
        if verbose:
            print(f"[previs] bundle      pose_landmarks ({len(ctx['rigged'])} figure(s))")

        # Pre-flight: the pose data already knows whether anyone left frame.
        # Say so here, where it is still cheap to re-block, rather than after
        # a generation comes back wrong.
        quality = bundle_mod.build_quality_report(
            pose, metadata.get("target_constraints"))
        bundle_mod._dump(out_dir / "quality_report.json", quality)
        entries["quality_report"] = "quality_report.json"
        if verbose:
            for warning in quality["warnings"]:
                print(f"[previs] WARNING     {warning}")

    # Stills at each camera-mark boundary plus first and last.
    if with_stills:
        mark_times = {0.0, ctx["duration"]}
        for move in (shot.get("camera") or {}).get("moves", []):
            if isinstance(move, dict):
                mark_times.add(float(move.get("start_t", 0.0)))
                mark_times.add(float(move.get("end_t", 0.0)))
        frames = sorted({max(1, min(scene.frame_end, int(round(t * fps)) + 1))
                         for t in mark_times})
        stills = api.render_stills(scene, out_dir / "stills", frames, prefix=shot_id)
        if stills:
            entries["stills"] = "stills/"
            if verbose:
                print(f"[previs] bundle      {len(stills)} still(s) -> stills/")


    # Depth pass last: it swaps the engine and compositor, so run it after
    # everything that depends on the reference render setup.
    if with_depth:
        depth = out_dir / f"{shot_id}_depth.mp4"
        near_m, far_m = _estimate_depth_range(shot, ctx)
        try:
            api.render_depth_pass(scene, depth, near_m=near_m, far_m=far_m)
            entries["depth"] = depth.name
            if verbose:
                print(f"[previs] bundle      depth -> {depth.name} "
                      f"(range {near_m:.1f}-{far_m:.1f}m)")
        except Exception as exc:  # depth is optional; never fail the bundle on it
            if verbose:
                print(f"[previs] WARNING     depth pass failed: {exc}")

    # Top-down staging diagram LAST: it builds a fresh scene (draw_* geometry
    # would otherwise pollute the reference render), which invalidates the
    # scene every render step above still needs. Verified the hard way --
    # running it before the depth pass killed depth with
    # "StructRNA of type Scene has been removed".
    if with_stills:
        try:
            diagram = out_dir / "stills" / "blocking_diagram.png"
            compile_camera_path(json.loads(json.dumps(shot)), diagram,
                                assets_root=assets_root, verbose=False,
                                mode="top", with_tracks=True)
            entries["blocking_diagram"] = "stills/blocking_diagram.png"
            if verbose:
                print(f"[previs] bundle      blocking_diagram -> stills/")
        except Exception as exc:  # diagnostic art; never fail the bundle on it
            if verbose:
                print(f"[previs] WARNING     blocking diagram failed: {exc}")

    bundle_mod.write_readme(out_dir, shot, entries)
    entries["readme"] = "README.txt"
    manifest = bundle_mod.write_manifest(
        out_dir, shot, entries, fps,
        extra={"generators": list(generators), "warnings": ctx["warnings"],
               "target_constraints": metadata.get("target_constraints", {})},
    )
    if verbose:
        print(f"[previs] bundle      manifest -> bundle_manifest.json "
              f"({len(manifest['files'])} entries)")
    return out_dir



def compile_camera_path(shot, output_path, assets_root=None, verbose=True,
                        mode="angle", with_tracks=False):
    """Build the scene (set, props, characters posed but not animated) and
    render it from a static *external* camera with the shot's actual computed
    camera trajectory drawn as a visible trail — green start, red end,
    connected segments. The shot's own camera can never show where it itself
    is; this is the only way to actually see the path rather than infer it
    from what that camera renders.
    """
    library = AssetLibrary(assets_root)
    expand_fixtures(shot, library)

    fps = int(shot.get("fps", 12))
    duration = float(shot["duration_seconds"])
    render_settings = shot.get("render") or {}
    fps = int(render_settings.get("fps", fps))

    scene = api.new_scene(fps=fps, duration_seconds=duration)
    api.add_key_light()

    stage = shot.get("stage") or {}
    size = stage.get("size_m", [12.0, 12.0])
    api.add_ground(size_m=size, grid=stage.get("ground_grid", True))

    set_spec = shot.get("set") or {}
    if set_spec.get("asset_id"):
        api.load_set(library.get("sets", set_spec["asset_id"]))

    for prop in shot.get("props", []):
        if not isinstance(prop, dict):
            continue
        api.place_prop(
            prop["id"],
            library.get("props", prop.get("asset_id", "")),
            pad3(prop.get("position", [0, 0, 0])),
            float(prop.get("facing_deg", 0.0)),
        )

    tracks = build_tracks(shot, library)
    for character in shot.get("characters", []):
        if not isinstance(character, dict):
            continue
        asset = library.get("characters", character.get("asset_id", ""))
        track = tracks[character["id"]]
        start_position, start_facing, _ = track.sample(0.0)
        if rig.is_rigged(asset):
            api.build_rigged_proxy(f"CHAR_{character['id']}", asset)
            # Posed at t=0 only, for scale reference -- not animated, since
            # the path visualisation cares about the camera, not the actors.
        else:
            api.place_character(character["id"], asset, start_position, start_facing)

    expand_presets(shot, tracks, library)
    camera_keys = build_camera_keys(shot, tracks, library, fps)
    warnings = check_camera_bounds(camera_keys, stage)
    api.draw_camera_path(camera_keys)

    # Character trails: blue->magenta, so they never read as the camera's
    # green->red. This is what turns a camera-path view into a staging diagram.
    if with_tracks:
        steps = max(2, int(round(duration * fps)))
        for index, (char_id, track) in enumerate(sorted(tracks.items())):
            positions = [track.sample(duration * i / (steps - 1))[0]
                         for i in range(steps)]
            tint = 0.35 + 0.25 * (index % 3)
            api.draw_motion_trail(
                positions, (0.15, 0.35, 0.95), (0.95, 0.20, tint),
                prefix=f"track_{char_id}",
            )

    set_extent = []
    if set_spec.get("asset_id"):
        half = [float(v) / 2.0 for v in size]
        set_extent = [[-half[0], -half[1], 0.0], [half[0], half[1], 0.0]]
    focus_points = [k.position for k in camera_keys] + set_extent
    api.add_observer_camera(focus_points, mode=mode)

    api.configure_render(
        scene,
        engine="WORKBENCH",
        resolution=render_settings.get("resolution", [960, 544]),
        fps=fps,
    )

    if verbose:
        print(f"[previs] shot        {shot.get('shot_id')} (camera path, {mode} view)")
        print(f"[previs] camera keys {len(camera_keys)} frames, "
              f"start={tuple(round(v, 2) for v in camera_keys[0].position)} "
              f"end={tuple(round(v, 2) for v in camera_keys[-1].position)}")
        for warning in warnings:
            print(f"[previs] WARNING     {warning}")

    result = api.render_preview_frame(scene, output_path, frame=1)
    if verbose:
        print(f"[previs] wrote camera path view -> {result}")
    return result
