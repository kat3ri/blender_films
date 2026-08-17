# Bundle contract (v1.0)

The interface between previs and the shot orchestrator. previs renders and
packages; the orchestrator composes the final scene prompt and owns model
submission. Everything the orchestrator needs is reachable from ONE file:

    <bundle>/bundle_manifest.json

Read `contract_version` first. It bumps on any breaking change to file roles
or the fields promised below (additive changes do not bump it).

## bundle_manifest.json

```jsonc
{
  "format": "previs.bundle_manifest",
  "version": "1.0",              // bundle format
  "contract_version": "1.1",     // THIS document
  "shot_id": "A1S1_SEG03",
  "fps": 12,                     // as rendered (targets may resample)
  "duration_s": 5.167,
  "files": { "<role>": "<relative path>", ... },
  "generators": ["generic", "seedance", "minimax"],
  "warnings": [ ... ],           // camera-bounds etc, from the compile
  "target_constraints": { "minimax": { ... } }   // see below
}
```

The manifest is written LAST — if it exists, the bundle is complete.

## File roles (`files`)

| role | file | notes |
|---|---|---|
| `reference` | `<shot>_reference.mp4` | the grey-box control render; H3's guide video |
| `depth` | `<shot>_depth.mp4` | optional; absent if the depth pass failed |
| `camera_motion` | `camera_motion.json` | per-frame camera, exact (authored, not solved) |
| `pose_landmarks` | `pose_landmarks.json` | per-frame 3D + 2D joints, exact; only when rigged |
| `metadata` | `metadata.json` | marks / lenses / timings digest |
| `prompt_<gen>` | `prompt.<gen>.txt` | rendered prompt text, standalone use |
| `prompt_fragments_<gen>` | `prompt_fragments.<gen>.json` | **the orchestrator input** |
| `stills` | `stills/` | frame at each camera-move boundary + first/last |
| `blocking_diagram` | `stills/blocking_diagram.png` | top-down staging: camera path (green→red) + character trails (blue→magenta) |
| `quality_report` | `quality_report.json` | pre-flight visibility analysis; see below |
| `plates` | `plates/` | background plates cut from the set's panorama at what the camera actually looks at |
| `plates_timeline` | `plates.json` | which plate is on screen when; see below |
| `readme` | `README.txt` | human index |

## prompt_fragments.<generator>.json

The prompt as addressable pieces. `prompt.<gen>.txt` is rendered FROM these,
so text and fragments cannot drift. Compose the final prompt by splicing your
scene/character prose around them; the pieces are already in the target's
token vocabulary (`<Video 1>` / `<Picture N>` for MiniMax H3).

| field | type | meaning |
|---|---|---|
| `generator` | str | profile these fragments target |
| `subjects_block` | [str] | per-reference role assignment ("<Video 1> controls only: ...") |
| `retain_lines` | [str] | "Retain the character poses and camera motion from <Video 1>." ... |
| `replace_lines` | [str] | one "Replace the figure ... with <name> from <Picture N>" per character, + one "Replace the grey geometry with <set>" |
| `scene_translation` | str | the set-replacement line alone (also last of replace_lines) |
| `camera_prose` | str | shot-size + move language derived from the actual camera keys |
| `action_prose` | [str] | per-character beats, in order, deduplicated |
| `shot_data` | obj | duration_s, lens_mm, aspect, max_duration_s, kept_duration_s |
| `notes` | str | the shot's human-authored notes, verbatim |

Picture numbering: `<Picture N>` follows the order of `characters[]` in the
shot JSON. The orchestrator must attach its reference images in that order,
or rewrite the tokens to match its own ordering.

## target_constraints.<generator>

What the target model actually keeps of this shot — computed, so the
orchestrator never re-derives H3's grid math:

| field | meaning |
|---|---|
| `target_fps` | fps the target resamples the guide to (H3: 24) |
| `resampled_frames` | frames after resample |
| `kept_frames` | frames after the target's 17k+5 trim |
| `kept_duration_s` | seconds of blocking that actually survive |
| `canvas_valid` | resolution is a multiple of 32 per axis (absent if no resolution set) |
| `warnings` | human-readable versions of any of the above being lossy |

Gate on `kept_duration_s < duration_s` if the shot's landing beat matters —
the trim cuts from the END.

## quality_report.json

Computed from the exact pose data before anything is generated — catches the
silent failure where a character walks out of frame or gets too small for
identity references to survive.

| field | meaning |
|---|---|
| `people[].id` | character id |
| `people[].in_frame_fraction` | 0-1; fraction of frames where most of the body is in frame |
| `people[].min_screen_height` / `max_screen_height` | body height as a fraction of frame height |
| `people[].first_lost_at_s` | when they first left frame, or null |
| `warnings` | human-readable; includes the target_constraints warnings, prefixed `[generator]` |

Thresholds warn at <90% in-frame and <15% screen height. It warns, it never
gates — a deliberate out-of-frame exit is a legitimate choice.

## plates.json  (contract 1.1)

Background plates, cut from the set's equirectangular panorama by reprojecting
the frustum the camera actually traverses. Because previs authored the camera,
this is a deterministic reprojection, not an estimate.

| field | meaning |
|---|---|
| `pano` | source equirect |
| `pano_yaw_offset_deg` | calibration between the set's world yaw and the pano's 0 (see `previs pano-check`) |
| `fov_x_deg` / `plate_margin` | taking FOV, and how much wider each plate is cut |
| `plates[].file` | image under `plates/` |
| `plates[].yaw_deg` / `pitch_deg` | plate centre direction |
| `plates[].t_start` / `t_end` | when this plate is the background |
| `plates[].frame_start` / `frame_end` | the frames it covers; consecutive plates are contiguous |

A locked-off shot yields one plate; a pan yields two or three. Capped at 3,
because MiniMax H3 has nine image slots shared with the cast.

The matching prompt fragment is `background_lines[]`, which numbers the plates
`<Picture N>` **after** the cast so identity references keep stable numbers, and
names the handover moment ("At 2.54s the camera turns and <Picture 3> becomes
the background").

## Worked example

Bundle any blocked shot and read the result:

    python -m previs.cli bundle shots/fortress/a1s1_seg03.json
    cat renders/A1S1_SEG03/bundle_manifest.json

The consumer test (`tests/test_h3_contract.py`) asserts the fragment fields
above; `tests` is the contract's executable half.
