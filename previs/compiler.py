"""Turn a validated shot spec into filmmaking-API calls.

Deliberately thin: the schema mirrors the API, the maths lives in ``motion``,
so this is mostly dispatch and bookkeeping. Runs inside Blender.
"""

from __future__ import annotations

from . import blender_api as api
from . import rig
from .asset_library import AssetLibrary, expand_fixtures
from .motion import build_camera_keys, build_tracks, check_camera_bounds, pad3


def compile_shot(shot, output_path, assets_root=None, verbose=True):
    """Build the scene described by ``shot`` and render it to ``output_path``."""
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

    result = api.render_control_video(scene, output_path)
    if verbose:
        print(f"[previs] wrote       {result}")
    return result
