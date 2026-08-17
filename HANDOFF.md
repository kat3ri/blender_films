# Handoff — previs overnight session (2026-08-16 → 17)

Branch **`feat/h3-prompt-contract`**, 6 commits off `040d645`, tree clean,
**57 tests** (`python3 -m unittest discover -s tests`, host-side, no Blender).

Goal was the Tuesday demo: one wooden-lounge shot end to end, joining the
ea_worlds decomposition, the pano pipeline, previs and MiniMax H3. **That path
works and has been through H3 successfully.**

---

## 1. What changed, by commit

| commit | what |
|---|---|
| `2a913ad` | H3 retention prompts, frame-grid awareness, orchestrator bundle contract |
| `2bd6415` | Framing presets (`ots`, `two_shot`, `dutch`…) + camera roll |
| `f77a0c8` | Pre-flight `quality_report.json`, `blocking_diagram.png` |
| `a3eb2fb` | ea_worlds importer (real meshes) + camera-driven pano plates |
| `b30919a` | Plates moved host-side (Blender's Python has no PIL) |
| `bdfdef0` | Fixed mirrored plates; off-centre parallax instrumentation |

New modules: `previs/pano.py`, `previs/framing.py`,
`previs/importers/ea_world.py`, `previs/pose_render.py`.
New docs: `docs/BUNDLE_CONTRACT.md` (contract **1.1**).

---

## 2. The demo path

```bash
export PREVIS_BLENDER=/weka/home-kateriw/3d/blender-4.2.0-linux-x64/blender

previs import-world /weka/home-kateriw/ea_worlds/out/world_wooden_lounge \
    wooden_lounge --no-shell --force
previs bundle shots/test/lounge_pan01.json --generators minimax --no-depth
```

Bundle lands in `renders/bundles/<SHOT_ID>/`: reference video, `plates/`,
`plates.json`, `prompt.minimax.txt`, `prompt_fragments.minimax.json`,
`camera_motion.json`, `bundle_manifest.json`, `stills/` (incl.
`blocking_diagram.png`).

**Always run Blender work under sbatch**, never on the login node. Working
sbatch templates: `/tmp/lounge_bundle.sbatch`, `/tmp/previs_verify.sbatch`
(p5en-small needs `--gpus=1` even for CPU-only Blender, or submission is
rejected).

---

## 3. The four expensive gotchas (all found by running, not by tests)

1. **Depth costs 19.9 s/frame — 41 min for a 124-frame shot**, and H3 cannot
   consume depth at all. It timed out a 20-minute job at frame 52.
   **Always pass `--no-depth` on the H3 path.** ~45 min → ~1 min.
2. **The ea_worlds origin is the pano camera, ~1.13 m above the floor.** Every
   object sat below previs's z=0 ground plane, which then hid the whole room —
   the first render was floor grid and nothing else. `ea_world.py` now detects
   the floor from the lowest object and lifts the room; `pano_origin_m` on the
   set records where the capture point ended up.
3. **Blender's bundled Python has no pip** (repo constraint). Plate rendering
   needs numpy + Pillow, so it runs **host-side** in
   `previs.pano.attach_to_bundle`, called from the CLI *after* Blender exits.
   Do not move it back into `compile_bundle`.
4. **Plates were horizontally mirrored.** With forward=+X and up=+Z, `right` is
   −Y, so the image's last column must sample +right. A mirrored room looks
   entirely plausible — it survived an H3 render and only surfaced because a
   sign read backwards. Both yaw tests checked the **centre pixel**, which a
   mirror leaves untouched. `TestHandedness` now guards both axes.

Ordering constraint: the blocking diagram builds a *fresh* Blender scene, which
invalidates the scene the depth pass needs. It must stay **last** in
`compile_bundle`.

---

## 4. Key design decisions

**Prompts.** `GENERATOR_PROFILES["minimax"]` shipped the phrasing that *failed*
live (pure negation, `Video1` instead of the literal `<Video 1>` token). It now
emits replace/retain imperatives that *translate* the blockout:

```
Retain the character poses and camera motion from <Video 1>.
Replace the figure from <Video 1> that marks Mauryl's position and timing
  with the character Mauryl from <Picture 1>.
Replace the grey geometry with <set display_name + notes>.
```

Emitted twice: `prompt.minimax.txt` for standalone use, and
`prompt_fragments.minimax.json` as addressable pieces for the **shot
orchestrator**, which composes the final scene prompt and owns submission —
previs deliberately does **not** talk to any model. The text is rendered *from*
the fragments so they cannot drift. See `docs/BUNDLE_CONTRACT.md`.

**H3 frame grid.** H3 resamples the guide to 24 fps then trims frames down to
the `17k+5` grid, so a 5.0 s shot silently loses its last 0.54 s — usually the
landing mark. `check_target_constraints()` warns with the two nearest valid
durations and puts the kept duration in the prompt. Safe durations @24 fps:
2.333 / 3.75 / **5.167** / 6.583 / 8.0 / 9.417 / 10.833 / 12.25 / 13.667 / 15.083 s.
Default canvas moved 960×540 → **960×544** (540 isn't a multiple of 32).

**Real geometry.** `room.json` gives label, MEASURED metric position/scale, and
a full `matrix_world`; its own note says only the rotation comes from the SAM 3D
pose. Meshes are the canonical `v0_uv40k.glb` (~1 MB vs 17 MB for `v0`), placed
by the decomposed matrix. `blender_api` flattens mesh materials for Workbench,
so **real geometry still renders grey** — accurate silhouettes and occlusion,
zero appearance leakage. Objects flagged `reliable: false` fall back to a box
sized from the mesh's own bounds × measured scale (sizing from `scale` alone
made a carpet a 3.6 m cube; correct is 3.64 × 1.95 × 0.04).

**Plates.** previs authored the camera, so `camera_motion.json` holds the exact
look direction per frame — the crop is a deterministic reprojection, not an
estimate. Frames are segmented into the fewest plates covering the shot
(locked-off → 1, pan → 2–3, capped at 3 because H3 has **nine image slots total,
shared with the cast**). Equirect convention matches the vendored MoGe helpers
exactly (`u = 1 − (atan2(y,x)/2π) % 1`, +X at u=1.0).
**`pano_yaw_offset_deg = 0`** for the wooden lounge, verified by cutting a plate
at the pool table's computed bearing (dead centre).

---

## 5. Test status

| test | state |
|---|---|
| End-to-end through H3 (locked/pan, camera at pano origin) | **PASSED** — user reports "perfect" |
| 1. Off-centre camera path | instrumentation **done**; first run measured **45°** parallax — see §6 |
| 2. Walk an actor through it | **PASSED** — `LOUNGE_WALK01` (SnapMoGen limb overlay, plate 0.08°) through H3; two-stage recipe below |
| 3. More complex camera | not started |

Test 2's blockout exercised both firsts: cast + plate share the `<Picture N>`
budget correctly (Mauryl = Picture 1, plate = Picture 2) and the offscreen
pre-flight caught a real framing error (walk path spanned ±35° of bearing
against a 35 mm lens's ±27° FOV; fixed with 28 mm + a ±28° path).

**Actor recipe (validated on H3, 2026-08-17)**: stage 1 with `<Video 1>` guide
+ cast + plate as usual; stage 2 **resample at higher denoise with the same
character and plate but the ref video removed**. The first pass bakes staging,
timing and screen geography into the latent; the guide-free resample lets H3
re-time the limbs at human dynamics, killing the video-game-character feel.
Mocap quality only needs to be "good enough" — direction and timing, not
polish.

**SnapMoGen is now local**: `/weka/home-kateriw/previs_mocap_cache/SnapMoGen/`
(9,155 BVHs + captions; `~/previs_mocap_cache` symlinks there, so the default
cache root resolves). `mocap-search` → `mocap-fetch` → paste the printed
`mocap_clip` action. A clip overlapping a `walk_to` is a **limb overlay**
(root stays on the authored path, `root_mode: lock_xy`); commit `0e7017c`
fixed the out-of-order root keys that pattern used to inject. The 13 GB
`renamed_feats.zip` (training features previs never reads) was not kept.

---

## 6. Off-centre instrumentation (for test 1)

A plate is only an exact reprojection **from the pano capture point**. Two
changes so leaving it is measurable:

- Plates are aimed **from the pano origin toward the camera's aim point**, not
  along the camera's heading — otherwise a moved camera points at the right
  subject from the wrong place. `CameraKey` and `camera_motion.json` now carry
  `aim` alongside `forward`.
- Each plate reports `camera_offset_m`, `nearest_subject_m`, and
  `parallax_deg = atan(offset / distance)`, warning past **12°**
  (`PARALLAX_WARN_DEG`). 1 m off with the subject 4 m away ≈ 14°; the same step
  1.5 m away ≈ 34°.

**First real measurement** (`LOUNGE_OFFCENTRE01`, dolly 1.6 m off-axis tracking
the pool table):

```
plate 1  yaw -120.1  offset 2.72 m  nearest subject 2.73 m  parallax 44.9 deg  ok=False
```

Offset ≈ subject distance, so the disagreement is ~45° — deliberately past the
point of plausibility, and a useful upper bound. Note it also collapsed to **one**
plate: the aim point barely moves in angle from the pano origin even though the
camera travels 3.8 m, which is the origin-relative aiming working as intended.

**What to do with it:** bracket downward — the same shot at ~1 m and ~0.5 m
offset gives ~20° and ~10°. Put those three plates through H3 and find where it
stops coping; that angle becomes the shot-design rule. The 12° threshold is a
guess until then.

---

## 7. Known gaps / next

- **Nothing verifies plate-vs-render agreement automatically.** The parallax
  number is geometry, not perception; only an H3 render says whether it holds.
- `--no-shell` omits the room envelope, so H3 leans entirely on the plates for
  the space. Whether the shell should go back in is undecided.
- **`pose_render.py`** (`previs pose-render <bundle>`) draws BODY_25 skeleton
  video from the exact pose data — built but **never run**. H3 can't consume it;
  it's for Wan/LTX-class control adapters.
- P2 items not started: A/B/C blocking variants, contact sheet, keyframable lens.
- `assets/`, `shots/`, `renders/` are gitignored — the wooden-lounge set asset
  and test shots exist only on disk at `/weka/home-kateriw/blender_films/`.
- Test fixtures live in `shots/test/`: `lounge_pan01.json` (90° pan, the demo
  shot), `lounge_offcentre01.json` (test 1), `panocheck_yaw0.json` (calibration).
- **SAM 3D Objects** (`/weka/home-kateriw/3d/sam-3d-objects`) is what gives
  ea_worlds its object rotations, via `ea_worlds/scripts/sam3d_lift.py`. It is
  under the bespoke **SAM License** — no non-commercial clause in the text, but
  it is not MIT/Apache, which sits against `ue-asset-pipeline`'s stated
  commercial-clean posture. Worth a read before anything ships on it.

## 8. Naming trap

"**sam3**" in `scene_decomp_out/*_sam3/` is the **2D** text-grounded segmenter
(comfyui-sam3), unrelated to **SAM 3D Objects**. Also: the demo room is
`ea_worlds/out/world_wooden_lounge`, **not** `scene_decomp_out/wooden_lounge_sam3`
(masks only, no lift) and not `lythwood_room` (a different room entirely).
