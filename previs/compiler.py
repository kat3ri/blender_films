"""Turn a validated shot spec into filmmaking-API calls.

Deliberately thin: the schema mirrors the API, the maths lives in ``motion``,
so this is mostly dispatch and bookkeeping. Runs inside Blender.
"""

from __future__ import annotations

from . import blender_api as api
from . import rig
from .asset_library import AssetLibrary, expand_fixtures
from .motion import build_camera_keys, build_tracks, check_camera_bounds, pad3


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

    # Props first: characters may be told to face them.
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
    rigged = []
    for character in shot.get("characters", []):
        if not isinstance(character, dict):
            continue
        asset = library.get("characters", character.get("asset_id", ""))
        track = tracks[character["id"]]
        start_position, start_facing, _ = track.sample(0.0)

        if rig.is_rigged(asset):
            root, joints = api.build_rigged_proxy(f"CHAR_{character['id']}", asset)
            api.animate_rigged_character(
                root, joints, track, asset, fps, scene.frame_end
            )
            rigged.append(character["id"])
        else:
            obj = api.place_character(character["id"], asset, start_position, start_facing)
            api.animate_character(obj, track, fps)

    camera_spec = shot.get("camera") or {}
    camera = api.create_camera(float(camera_spec.get("lens_mm", 35.0)))
    camera_keys = build_camera_keys(shot, tracks, library, fps)
    api.animate_camera(camera, camera_keys)
    warnings = check_camera_bounds(camera_keys, stage)

    api.configure_render(
        scene,
        engine=render_settings.get("engine", "WORKBENCH"),
        resolution=render_settings.get("resolution", [960, 540]),
        fps=fps,
    )

    if verbose:
        print(f"[previs] shot        {shot.get('shot_id')}")
        print(f"[previs] duration    {duration}s @ {fps}fps "
              f"({scene.frame_start}-{scene.frame_end})")
        print(f"[previs] characters  {[c.get('id') for c in shot.get('characters', [])]}"
              + (f"  (articulated: {rigged})" if rigged else ""))
        print(f"[previs] props       {[p.get('id') for p in shot.get('props', [])]}")
        print(f"[previs] camera      {[m.get('type') for m in camera_spec.get('moves', [])]}")
        for kind, asset_id in library.missing:
            print(f"[previs] WARNING     no {kind[:-1]} asset {asset_id!r}; using placeholder")
        for warning in warnings:
            print(f"[previs] WARNING     {warning}")

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


def compile_camera_path(shot, output_path, assets_root=None, verbose=True, mode="angle"):
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

    camera_keys = build_camera_keys(shot, tracks, library, fps)
    warnings = check_camera_bounds(camera_keys, stage)
    api.draw_camera_path(camera_keys)

    set_extent = []
    if set_spec.get("asset_id"):
        half = [float(v) / 2.0 for v in size]
        set_extent = [[-half[0], -half[1], 0.0], [half[0], half[1], 0.0]]
    focus_points = [k.position for k in camera_keys] + set_extent
    api.add_observer_camera(focus_points, mode=mode)

    api.configure_render(
        scene,
        engine="WORKBENCH",
        resolution=render_settings.get("resolution", [960, 540]),
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
