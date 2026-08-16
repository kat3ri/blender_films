# AI-Directed Blender Previs

Blender as an automated blocking engine. You describe a shot; it renders a crude
3D control video — who is where, how they move, where the camera goes, timing —
to condition a downstream AI video model. You never open Blender.

The video is deliberately ugly. Its job is to give the generative model
something deterministic to follow.

## Quick start

```bash
python -m previs.cli info                                    # check the Blender install
python -m previs.cli render shots/examples/synth_test_shot.json
```

The control video lands in `renders/`. That test shot is the prototype proof: a
character walks A→B, crouches at a crate, and the camera tracks then orbits.

## Working with it

You are not meant to hand-write shot JSON. Ask Claude for what you want —
*"have the cleaner cross the platform and crouch at the drain, camera tracking
wide behind her"* — and it authors the blocking, renders, checks the frames, and
gives you the video. Then give notes (*"camera's too far", "she walks too
fast", "stop closer to the fire"*) and it revises. The
[`previs-blocking`](.claude/skills/previs-blocking/SKILL.md) skill is the
procedure it follows.

## Importing your shot lists

Source repos are registered in [`projects.json`](projects.json), so imports use
short names:

```bash
python -m previs.cli import fortress a1s1     # Fortress scene -> 5 shot stubs
python -m previs.cli import ums EP01          # MiniMax episode -> 7 shot stubs
```

A full path still works (`import fortress D:/.../a1s1.json`). Nothing is ever
written back into your story repos unless you set `writeback` in the registry.

Importers extract the mechanical half — durations, beat timings, who is present,
which location, and how clips chain — and carry the prose through verbatim as
`_source_text`. They **never invent positions**, because neither format contains
any: blocking is a judgement call made afterwards. Stubs arrive as
`status: "needs_blocking"` with empty actions and camera moves; that is the
handoff point.

Re-importing is safe: a shot already marked `blocked` keeps its blocking and
only has its source-derived fields refreshed.

## Cross-shot continuity

Both formats chain clips, and the importers preserve it — Fortress via
`is_chain_start`/`order`, MiniMax via prose ("continuing directly from Job 1",
"after Job 6"). Each shot carries:

```jsonc
"continuity": {
  "order": 2,
  "continues_from": "A1S1_SEG02",
  "carry": {"position": true, "camera": "match"}
}
```

```bash
python -m previs.cli continuity check     # report breaks across every chain
python -m previs.cli continuity apply     # seed start states from the previous shot
```

`check` catches characters that teleport, snap round or change pose between
shots. `apply` seeds the next shot's `start_position`/`start_facing_deg`/
`start_pose` from where the previous one actually ends — computed with the same
trajectory maths that renders, so it always agrees with the video.

**Camera carry is conditional, not automatic.** `"match"` means the clips are
genuinely continuous, so the camera must not jump and `apply` moves the opening
camera to the previous shot's final position. `"cut"` means a new setup, free to
go anywhere — the MiniMax default, since each job is a separate generation. The
importer decides from what the source says; blocking can override it.

`A1S1_SEG02 → A1S1_SEG03` is a worked example: the last frame of one and the
first frame of the other show the same thing.

Worked examples of the finished result: [`shots/fortress/a1s1_seg03.json`](shots/fortress/a1s1_seg03.json)
and [`shots/minimax/EP01_JOB01.json`](shots/minimax/EP01_JOB01.json).

## Stage convention

```
+X screen-right     +Y upstage (away from a camera parked at -Y)     +Z up
facing_deg: degrees CCW from +X.  0 = screen-right, 90 = upstage, -90 = toward camera.
```

Metres and seconds throughout. The rendered 1m floor grid is what makes distance
and camera movement legible in the output.

## Assets are authored once and reused

`assets/sets/`, `assets/characters/`, `assets/props/` — JSON proxy definitions
built from primitives (`box`, `cylinder`, `capsule`, `uv_sphere`, `cone`,
`plane`). **A location is blocked out once and every shot there reuses it.**
Importers only ever create a placeholder if nothing exists; they never overwrite
an authored asset.

Build a set from a reference image plus its prose, then look at it before you
block anything inside it:

```bash
python -m previs.cli survey lowlit_bar_near_closing --figure-at 0.2 1.8
```

`survey` renders the empty space with a human-scale figure in it. The default
**push** travels from the open -Y end down the length of the room, which is the
viewpoint reference plates are usually shot from — so the survey is directly
comparable to the image the set was built from, and it moves along the open axis
so it cannot drive through furniture. `--mode pan` looks around from one spot;
`--mode orbit` circles the set, which is fine outdoors and clips indoors.

A set can declare `fixtures` — named interactable objects that are part of the
location itself (a door, a hearth, a gargoyle). They appear in every shot at
that location automatically and are valid `interact` targets, so no shot in a
chain can end up missing a piece of its own set. `props` is for objects specific
to one shot.

`shape` is an open enum. `"mesh"` is real: a part can reference a downloaded
asset file instead of procedural geometry --

```bash
python -m previs.cli asset-search-polyhaven "bar stool"
python -m previs.cli asset-fetch-polyhaven bar_chair_round_01 --kind props --as bar_stool_real
```

[Poly Haven](https://polyhaven.com)'s API is open (no auth, just a
`User-Agent` header) and every asset is CC0 -- free for any use, no
attribution. `asset-search-polyhaven` ranks candidates by tag/name match for
you to review and pick, same principle as `mocap search`: asset choice is a
directorial judgement call, never auto-selected. `asset-fetch-polyhaven`
downloads the glTF (cached under `PREVIS_ASSET_CACHE`, default
`~/previs_asset_cache/polyhaven/` -- outside the repo, same reasoning as the
planned mocap cache) and writes a ready-to-use asset JSON.

A mesh part's materials are always flattened to a solid colour (sampled from
the source texture, or an explicit `"color"` override) before rendering --
Workbench's `MATERIAL` shading reads only `material.diffuse_color` and never
samples an image texture, so an imported asset's real PBR textures render as
flat mid-grey otherwise. Confirmed by an actual render before this was built,
not assumed. See `blender_api.py`'s module docstring for this and every other
hard-won Blender API gotcha hit building this system.

A part (or set fixture) can also carry `"repeat": {"axis": "x", "count": N,
"spacing": m}` to become N evenly spaced copies, and its own `"color"`
distinct from the asset default. Both exist because hand-placing every
repeated element (stones, bottles, bricks) makes authors under-count real
density, and a video model conditioned on a too-sparse control clip tends to
take the count literally and hallucinate trying to reconcile it.

## Articulated characters

A character asset with `"rig": "humanoid"` gets a jointed body instead of a
capsule — head, torso, arms, legs, feet — with a procedural walk cycle. Without
it there is no *body mechanics* at all, which is a named authority channel for
video-reference models, and the reason an "old man's walk" could otherwise only
be expressed through pacing.

```jsonc
{"rig": "humanoid", "height_m": 1.74, "gait": {"preset": "elderly"}}
```

Ageing or energising a walk is a gait profile, not new geometry: `elderly`,
`brisk`, `child`, or inline overrides for `stride_m`, `hip_swing_deg`,
`knee_lift_deg`, `arm_swing_deg`, `stoop_deg`, `torso_bob_m`.

**The gait cycle is a function of distance travelled, not time.** Stride
therefore always agrees with ground actually covered, so the feet cannot slide
at any speed — the hardest problem in retargeted mocap never arises. Poses
(`crouch`, `kneel`, `sit`, `reach`) bend real joints and ease from one to the
next rather than popping.

Rest pose is a **T-pose with conventional joint names** (`hips`, `spine`,
`chest`, `l_shoulder`, `l_knee`, …), deliberately matching the SnapMoGen BVH
convention. Joints are parented empties rather than a Blender armature, so
driving them from a mocap clip later is a joint-name lookup over a parsed BVH
file — no armature retargeting, no addon, still stdlib-only. See
`shots/examples/rig_test.json` for a mechanics test.

Characters without `"rig"` keep the flat capsule proxy and render exactly as
before.

## How it fits together

```
shot JSON  ->  compiler  ->  blender_api (bpy)  ->  Workbench render  ->  control.mp4
                   ^
              motion.py: trajectories + look-at camera maths
```

| Module | Role | Needs Blender |
|---|---|---|
| `previs/schema.py` | shot spec + validation | no |
| `previs/asset_library.py` | proxy geometry defs | no |
| `previs/motion.py` | trajectory and camera maths | no |
| `previs/rig.py` | humanoid skeleton, poses, gait | no |
| `previs/blender_api.py` | the filmmaking API | **yes** |
| `previs/compiler.py` | spec → API calls | **yes** |
| `previs/driver.py` | entry point inside Blender | **yes** |
| `previs/cli.py` | host-side launcher | no |
| `previs/importers/` | story-format parsers | no |

Only three modules touch `bpy`, and the maths that actually decides what the
shot looks like is testable without Blender. That is what keeps Blender a
replaceable implementation detail rather than the product.

Everything is stdlib-only — Blender bundles its own Python with no pip packages,
so anything running inside it cannot depend on third-party code. Asset and shot
files are JSON rather than YAML for the same reason.

## Requirements

- **Blender 4.5 LTS** — `winget install BlenderFoundation.Blender.LTS.4.5 --source winget`,
  or run [`scripts/setup_blender.ps1`](scripts/setup_blender.ps1). Found via
  `PREVIS_BLENDER`, then `PATH`, then the standard install locations; override
  with `--blender`.
- **Python 3.9+** for the CLI. No packages to install.
- **ffmpeg** (optional) only for pulling contact sheets out of a render to
  review it.

Rendering is headless: `blender --background --factory-startup --python`. Nothing
opens a window.

## Commands

```bash
python -m previs.cli render <shot.json> [--out path] [--allow-unblocked] [--preview [FRAME]]
python -m previs.cli import <project> <name>    # project from projects.json
python -m previs.cli survey <set_id> [--mode push|pan|orbit] [--figure-at X Y]
python -m previs.cli asset-search-polyhaven "<query>" [--kind models|hdris|textures]
python -m previs.cli asset-fetch-polyhaven <id> --kind <props|sets|characters> --as <name>
python -m previs.cli camera-path <shot.json> [--mode top|angle]
python -m previs.cli continuity <check|apply> [paths...]
python -m previs.cli validate [paths...]        # defaults to every shot in shots/
python -m previs.cli info
```

`render` refuses a `needs_blocking` stub unless you pass `--allow-unblocked`,
which previews it with a default wide static camera.

`--preview` renders one still frame (default: frame 1) to a PNG plus a scene
manifest (every mesh actually built, with world-space position, dimensions,
face count) instead of the full animated video -- a cheap sanity check before
paying for a full render. Same engine, same resolution, same materials as the
real render; only the frame count differs. Worth knowing: Blender's own
process-startup cost (~20s) dominates total time for a typical short shot, so
`--preview` isn't dramatically faster than a full render at that scale -- its
real value is the manifest and an instantly-viewable PNG, not raw speed.

`camera-path` renders the shot's *actual computed* camera trajectory as a
visible trail (green start, red end) viewed from a static camera placed
outside the shot, in either a top-down plan view or a 3/4 elevated view. The
shot's own camera can never show where it itself is -- this exists because a
shot can look wrong for reasons a rendered frame alone won't reveal: a dolly
that stops short of its subject instead of arriving, an orbit whose radius
sends it past a wall, a track that runs backward. Caught exactly the first of
those on `cam_test_aerial_establish.json` the first time this was used for
real -- the descent's end position was ~14m out from the keep instead of
converging on it, which only became obvious once the path was actually drawn
rather than inferred from the rendered frames.

Watch the compiler's `WARNING` lines — they flag placeholder assets and cameras
that have left the stage (which renders as the back of a wall).

## Not built yet

Multiple blocking candidates per shot (A/B/C variants), rigged character
animation, sourcing real geometry for clay blockouts, and any hookup to the
generative video model itself. The schema leaves room for all of them.
