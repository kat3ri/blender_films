---
name: previs-blocking
description: Author or revise 3D blocking for a previs shot in this repo — turn prose shot descriptions (or an imported needs_blocking stub) into positions, timed actions and camera moves in a shot JSON, then render the control video. Use when asked to create a shot, block a scene, change camera/staging, or act on revision notes like "camera too far" or "she walks too fast".
---

# Blocking a previs shot

This repo renders crude 3D control videos to condition a downstream AI video
model. Your job is the one step that cannot be automated: turning prose into
actual space and time. You are the previs artist, not a parser.

Read `README.md` first if you have not already — it defines the stage
convention you are about to use.

## 0. Prefer real data over inference

If the source already carries explicit spatial or camera data — coordinates, a
camera path, measured room dimensions — **use it directly**. The heuristics
below exist only because the current story formats carry prose. When upstream
tooling starts emitting positions, this step shrinks to a pass-through.

## 1. Read the source before you place anything

For an imported stub, the prose is already in the file:

- `_beats` — every timed beat of the shot
- `characters[].​_source_text` — the beats that mention that character
- `camera._source_text` — the `[CAMERA]` block and/or beat camera language
- `_scene`, `_identity_locks`, `_acting`, `_negatives` (MiniMax only)

Check `characters[]` against `_identity_locks`. The MiniMax importer *guesses*
identities from prose and sometimes misses one (a job whose locks say "exactly
one named subject — the youth" yields no character). Add or rename characters
as needed, and delete any that are scenery rather than people.

## 2. Stage convention

```
+X screen-right      +Y upstage (away from a camera parked at -Y)      +Z up
facing_deg: degrees CCW from +X.  0 = screen-right, 90 = upstage, -90 = toward camera.
```

Metres and seconds throughout. Keep everything inside `stage.size_m` — the
compiler warns if the camera leaves the stage, because a camera behind a wall
renders as a flat grey rectangle and wastes a review round.

## 3. Continuity: check what this shot continues from

Look at `continuity` before blocking anything:

```jsonc
"continuity": {
  "order": 2,
  "continues_from": "A1S1_SEG02",
  "carry": {"position": true, "camera": "match"},
  "_reason": "duration/beat cap reached — chains from the previous clip"
}
```

Both source formats state chaining outright and the importers preserve it —
Fortress via `is_chain_start`/`order`, MiniMax via prose ("continuing directly
from Job 1", "Same as Job 3", "after Job 6").

- **`continues_from` set** — block the *predecessor* first, then run
  `python -m previs.cli continuity apply` to seed this shot's
  `start_position`, `start_facing_deg` and `start_pose` from where that shot
  actually ends. Do not retype coordinates by hand; the carry is computed from
  the same trajectory maths that renders.
- **`carry.camera: "match"`** — the clips are genuinely continuous, so the
  camera must not jump. `apply` moves this shot's first camera move to the
  predecessor's final camera position and matches the lens. Verify by comparing
  the last frame of one against the first frame of the next; they should show
  the same thing.
- **`carry.camera: "cut"`** — a new setup. Put the camera wherever the beat
  wants. MiniMax jobs default to this: each job is a separate generation.

The importer's guess is not binding. If the prose says the camera holds but the
metadata says cut (or vice versa), change it and say why in `notes`.

Finish with `python -m previs.cli continuity check` — it reports characters that
teleport, snap round, or change pose between shots, and cameras that jump on a
`match` pair.

## 4. Assets: reuse before you create

Check `assets/sets/`, `assets/characters/`, `assets/props/` first. **A location
is blocked out once and reused by every shot there** — never re-derive a set
per shot.

**Before hand-building furniture or set-dressing, check for a real asset.**
`python -m previs.cli asset-search-polyhaven "<query>"` searches Poly Haven's
CC0 catalogue (free for any use, no attribution) -- chairs, tables, doors,
statues, and genuine architectural pieces (a search for "castle door" turns up
an actual `large_castle_door` asset). Review the ranked list, pick one, then
`asset-fetch-polyhaven <id> --kind props --as <name>` downloads it and writes
a ready-to-use asset JSON. The one thing that stays hand-built primitives: a
set's own bounding walls/floor/ceiling, since no catalogue asset can match a
custom-dimensioned room -- everything placed *within* those walls is fair
game for a real asset if a good match exists.

A set stub with `"needs_blockout": true` is a placeholder box room. To make it
real:

1. If the asset has a `reference_image` path, **look at the image** (Read it)
   and base the layout on what you see: room bounds, where walls and openings
   are, where the big furniture sits. Estimate metres from human scale in frame
   — a doorway is ~2.1m, a bar counter ~1.1m, a ceiling ~2.7m domestic.
2. Otherwise build from the prose in `notes` / `_scene`.
3. Express it as `parts` — boxes, planes, cylinders. Crude is correct. Leave
   the side the camera shoots from open so it can see in.

   **Match the reference image's repeated-element density, not a token few.**
   Count roughly how many stones/bottles/bricks/planks/rails are actually
   visible and get within shouting distance of that — not exactly, but not 3
   when the photo shows 15 either. A too-sparse count reads to a downstream
   video model as a literal, deliberate feature count, and it will hallucinate
   or warp the geometry trying to reconcile a control clip against what the
   scene "should" have. Use `repeat` (see `previs/asset_library.py`) instead of
   hand-placing each copy — `{"shape": "box", "position": [...], "size": [...],
   "repeat": {"axis": "x", "count": 8, "spacing": 0.4}}` — so getting a
   plausible count costs one line, not one line per copy. Give repeated
   elements a small real gap (not perfectly flush) or they merge into one
   smooth mass under flat shading with no visible seam, which defeats the
   point.

   A part (or a set's `fixtures`) may also carry its own `"color"` distinct
   from the asset default — material variation (mortar, wood grain, a warning
   light) reads as real detail instead of a flat mass. This is genuinely
   rendered now (Workbench uses per-face materials), not free extra polygons.
4. Remove `needs_blockout` and save. Every later shot at that location inherits it.
5. **Look at it before using it:**
   `python -m previs.cli survey <set_id> --figure-at X Y`
   This renders the empty space with a 1.75m figure standing in it. Read the
   frames against the reference image: is the room the right size next to the
   figure, are the big masses in the right places, can a camera at the open end
   see in? Fix and re-survey. A set is used by every shot at that location, so
   an error here is an error in all of them.

Scale anchors for reading a reference photo: doorway 2.0-2.1m, bar counter
1.05-1.15m, table 0.72-0.78m, stool seat 0.65-0.78m, chair back ~0.9m,
domestic ceiling 2.4-2.7m, basement/industrial 2.8-3.2m. Find one of these in
frame and measure everything else against it.

**Fixed scenery belongs to the set, not the shot.** A door, hearth, wall panel
or statue that is physically part of the location goes in the set's `fixtures`
array, not in a shot's `props`:

```jsonc
"fixtures": [
  {"id": "door", "asset_id": "wooden_door", "position": [-5.85, 4.0, 2.4], "facing_deg": 0}
]
```

Fixtures appear in every shot at that location automatically and are valid
`target_id`s for `interact`. Listing them per shot instead is how one shot in a
chain ends up missing a piece of its own set — which is a continuity break that
nothing else will catch. Reserve `props` for objects genuinely specific to one
shot, or ones a shot needs to move.

New characters: copy an existing character JSON and adjust height and colour.
Give each person on stage a distinct colour so silhouettes stay tellable apart.

Set `"rig": "humanoid"` with a `height_m` to get an articulated body and a walk
cycle instead of a capsule. Pick a `gait` preset that matches the character —
`elderly`, `brisk`, `child` — or override individual numbers. This is how you
age or energise a walk; do not try to fake it with pacing alone once a character
is rigged.

## 5. Translate prose into blocking

Shot size sets camera distance from the subject (35mm lens):

| Prose | Distance | Camera height |
|---|---|---|
| wide / establishing | 7–10 m | 1.8–3.0 m |
| medium | 3–4.5 m | 1.6 m (eye level) |
| medium-close | 2–3 m | 1.6 m |
| close-up | 1.2–2 m | subject's head height |
| low angle | — | 0.4–0.8 m |
| high angle / overhead | — | 3–6 m |

Camera language maps to move types:

| Prose | Move |
|---|---|
| static, held, locked off | `static` |
| pans across / sweeps | `pan` |
| tilts up/down | `tilt` |
| tracks with / follows | `track` (keeps a constant offset from a moving target) |
| pushes in, pulls back, dollies | `dolly` |
| arcs, circles, orbits around | `orbit` |

Action verbs map to character actions: walks/crosses/enters/descends →
`walk_to`; stops/waits/holds/listens → `idle`; turns/looks toward → `turn_to`;
reaches/crouches/kneels/touches → `interact` with a `pose`.

Pace sanity-check: an ordinary walk is **1.2–1.4 m/s**. Divide the distance by
the time; if it implies 3 m/s the character will look like they are sprinting.

## 6. Rules the compiler relies on

- Every action and camera move needs `start_t` and `end_t` in seconds.
- **Camera moves must not overlap in time** — validation rejects it. Butt them
  end to end (`0→4`, `4→8`).
- Make consecutive camera moves *positionally continuous*, or the cut will pop.
  Compute where the previous move ends and start the next one there. Example:
  a `track` at offset `[0, -4.5, 1.8]` from a subject ending at `[0.5, 1.0]`
  leaves the camera at `[0.5, -3.5, 1.8]`; an `orbit` on that subject with
  `radius_m: 4.5`, `start_deg: -90`, `height_m: 1.8` starts at exactly that point.
- Cover the full `duration_seconds`. Gaps hold the last state, which is legal
  but usually not what you meant.
- Set `status` to `"blocked"` and write one line in `notes` recording the
  interpretation choices you made, so a later revision knows your reasoning.

## 7. Validate, render, report

```bash
python -m previs.cli validate shots/<path>.json
python -m previs.cli render   shots/<path>.json
```

Read the compiler's `WARNING` lines — placeholder assets and out-of-bounds
cameras both surface there.

**Then look at what you made.** Do not hand over a video you have not seen:

```bash
ffmpeg -y -i renders/<SHOT_ID>.mp4 -vf "select='not(mod(n\,19))',tile=3x2" -frames:v 1 /tmp/sheet.png
```

Read that contact sheet and check, in this order — these are the failures that
actually happen:

1. **Is the subject visible in every beat?** Set geometry on the sight line is
   the most common bug and nothing warns you about it. A column, wall or stair
   mass directly between camera and subject hides them completely. Trace the
   camera→subject line against the set's `parts` and offset the camera sideways
   if anything sits on it.
2. **Is anyone cropped?** At 35mm the vertical frame is roughly `0.68 x
   distance` metres, centred on the aim point. A 1.75m figure aimed at 0.5m
   height from 3m away loses their head. Raise the aim or pull back.
3. **Is the camera inside geometry**, or shooting the back of a wall? The
   out-of-bounds warning catches the stage edge, not interior walls.
4. **Does the action read** — does the walk cover real ground, does the pose
   change register in silhouette?

Fix and re-render before reporting. Then tell the user the output path and what
they should look at.

If a shot's camera behaviour looks wrong and it's not obvious *why* from the
rendered frames alone (a push that never seems to arrive, an orbit that clips
something, a direction that doesn't match what was authored) --
`python -m previs.cli camera-path <shot.json> --mode top` renders the actual
computed trajectory as a visible trail from outside the shot. The shot's own
camera can never show where it itself is, so this is the only way to actually
see a path rather than infer it from what that camera renders. Caught a real
bug the first time it was used: a dolly's end position was ~14m short of the
subject it was supposed to be arriving at, invisible in the rendered frames
alone since each one looked individually plausible.

## 8. Revision notes

Notes are edits to the shot JSON, not rebuilds. Change the smallest thing that
answers the note, re-render, re-check:

| Note | Edit |
|---|---|
| "camera too far / too close" | camera move `position` distance, or `lens_mm` |
| "she walks too fast" | widen the `walk_to` span, or shorten the distance |
| "stop closer to the fireplace" | the `walk_to` `position` |
| "hold on her longer" | extend the `idle`/`interact` span, shift later actions |
| "start on the door instead" | the first camera move's `position`/`target_id` |

Preserve everything the user already approved.
