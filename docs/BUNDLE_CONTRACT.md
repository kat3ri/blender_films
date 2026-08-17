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
  "contract_version": "1.0",     // THIS document
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

## Worked example

Bundle any blocked shot and read the result:

    python -m previs.cli bundle shots/fortress/a1s1_seg03.json
    cat renders/A1S1_SEG03/bundle_manifest.json

The consumer test (`tests/test_h3_contract.py`) asserts the fragment fields
above; `tests` is the contract's executable half.
