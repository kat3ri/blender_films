# AI-Directed Blender Previs / Control Video System

## Project Goal

Build a lightweight system that uses Blender as an **automated 3D blocking/previsualization engine** for an AI video production pipeline.

The user already has:

- Sets/environments
- Characters
- Props
- Shot plans
- Scene/story information
- Downstream AI video generation models

The missing piece is reliable **spatial and camera control**.

Instead of asking an AI video model to infer an entire shot from text/images, we want to create a crude 3D representation of the shot in Blender and render it as a control video.

The Blender scene does **not** need to look good.

It only needs to communicate:

- Where objects/characters are
- How they move
- How they interact
- Where the camera is
- How the camera moves
- Timing
- Rough composition

The AI video model will handle the final visual appearance.

---

## Core Concept

The intended pipeline is:

```text
SHOT PLAN
    ↓
AI / DIRECTOR AGENT
    ↓
STRUCTURED SHOT DESCRIPTION
    ↓
BLENDER AUTOMATION
    ↓
ROUGH 3D BLOCKING
    ↓
CONTROL VIDEO
    ↓
AI VIDEO MODEL
    ↓
FINAL CINEMATIC SHOT
```

The user should ideally **never have to manually operate Blender**.

Blender should behave like a programmable virtual camera/blocking system.

---

## Important Design Principle

Do NOT build this as a conventional Blender filmmaking workflow.

We are deliberately trying to avoid requiring the user to:

- Open Blender manually
- Position objects manually
- Animate characters manually
- Keyframe cameras manually
- Learn Blender's UI
- Build detailed 3D scenes
- Render expensive photorealistic frames

The system should make Blender effectively invisible.

The user should interact with a higher-level interface such as:

> "Have Alice walk from the library door to the fireplace, stop, turn toward the window, and have the camera track backward before arcing around her."

The system should translate that into Blender operations automatically.

---

## What Blender Is Responsible For

Blender is primarily responsible for four things:

### 1. Spatial Layout

Represent:

- Characters
- Props
- Sets
- Important environmental geometry

Exact visual fidelity is not important.

### 2. Motion

Represent approximate:

- Character movement
- Object movement
- Interactions
- Simple poses
- Timing
- Start/end positions

### 3. Camera

Represent:

- Camera position
- Camera orientation
- Focal length
- Camera targets
- Tracking
- Dolly
- Orbit
- Pan
- Tilt
- Other basic cinematic movements

### 4. Control Video

Render a cheap video showing the blocking.

The output should prioritize:

- clear silhouettes
- spatial relationships
- motion
- camera movement
- timing

over visual quality.

---

# Example

A shot plan might say:

> Elena enters the library, walks toward the fireplace, stops, hears something, and looks toward the balcony. Camera tracks backward in front of her and then slowly arcs to her side.

The Blender representation could be extremely crude:

```text
ELENA = capsule
FIREPLACE = box
LIBRARY = simplified geometry
BALCONY = box/platform
CAMERA = virtual camera
```

The system creates the animation and camera path automatically.

The resulting control video might look like a primitive animation.

That is completely acceptable.

The downstream AI model receives that control video and generates the actual cinematic library, character, lighting, clothing, atmosphere, etc.

---

# Higher-Level API

Do not make the LLM directly manipulate arbitrary Blender Python whenever possible.

Create a small filmmaking abstraction/API.

Potential commands include:

```text
create_shot()
load_set()
place_character()
place_prop()

move_object()
move_character()
set_pose()

animate_walk()
animate_action()
set_duration()

create_camera()
set_camera_position()
set_camera_target()
set_focal_length()

track_camera()
orbit_camera()
dolly_camera()
pan_camera()
tilt_camera()

render_control_video()
```

The exact API should be determined after investigating Blender's Python API and the simplest robust architecture.

The abstraction should make Blender replaceable in the future.

---

# Shot Representation

A shot should ideally have a machine-readable representation.

For example:

```json
{
  "shot_id": "SEQ01_SHOT03",
  "duration": 6,
  "location": "great_hall",
  "characters": [
    {
      "id": "elena",
      "start_position": "...",
      "action": "walk_to_fireplace",
      "end_position": "...",
      "final_action": "look_toward_balcony"
    }
  ],
  "camera": {
    "type": "tracking_arc",
    "lens": 35,
    "start_position": "...",
    "target": "elena"
  }
}
```

This is illustrative, not a fixed schema.

Design a clean schema that can eventually support a complete production.

---

# Iterative Workflow

The system should support iterative blocking.

Example:

```text
User:
Create the shot.

Agent:
Builds Blender scene.

Agent:
Renders control video.

User:
The camera is too far away and Elena walks too fast.

Agent:
Modifies the shot.

Agent:
Renders revised control video.

User:
Camera is good. Have Elena stop closer to the fireplace.

Agent:
Modifies the blocking.

Agent:
Renders again.

User:
Approve.
```

Once approved, the control video can be passed downstream to the generative video pipeline.

---

# Multiple Blocking Candidates

A useful future capability is generating several cheap camera/blocking variations automatically.

For example:

```text
Shot 23

A — wide tracking
B — medium tracking
C — over-the-shoulder
D — side profile
E — low-angle push-in
```

The user chooses one, then the agent refines it.

This is preferable to asking a generative video model to randomly invent five different camera compositions.

---

# Asset Philosophy

Assets should be reusable.

A project might contain:

```text
/assets
    /sets
        /library
        /castle
        /bedroom

    /characters
        /elena
        /marcus

    /props
        /book
        /chair
        /candle
```

The same character/set/prop should be usable across many shots.

For the first prototype, simplified proxy geometry is acceptable.

Eventually, existing production assets should be usable directly.

---

# Rendering Requirements

The control render should be intentionally cheap.

Possible approaches to investigate:

- Blender Eevee
- Workbench
- viewport rendering
- flat/shaded proxy materials
- low resolution
- low frame rate if appropriate
- simplified geometry
- headless Blender rendering

The important output is a video file suitable for use as a conditioning/control input.

---

# Headless Operation

The long-term system should support:

```text
Shot JSON
    ↓
Blender Python
    ↓
.blend scene
    ↓
headless Blender
    ↓
control.mp4
```

The user should not need to interact with the Blender GUI.

Investigate the most reliable way to launch Blender, execute Python, load assets, construct/update scenes, animate objects, and render video from the command line.

---

# Architecture Goal

Think of the system as:

```text
                 DIRECTOR AGENT
                       ↓
                 SHOT SPECIFICATION
                       ↓
                 FILMMAKING API
                       ↓
                    BLENDER
                       ↓
                CONTROL VIDEO
                       ↓
               GENERATIVE VIDEO
                       ↓
                 FINAL SHOT
```

Blender is an implementation detail, not the user-facing product.

The filmmaking API should ideally be sufficiently abstract that another 3D engine could replace Blender later.

---

# First Prototype

Do NOT attempt to build the entire production system immediately.

The first prototype should prove:

1. A shot can be described as structured data.
2. An automated script can create a Blender scene from that data.
3. Primitive objects can represent characters/props/environment.
4. Character/object movement can be generated automatically.
5. A camera can be positioned and animated automatically.
6. Blender can render the resulting blocking as a video without manual UI interaction.
7. The resulting video is visually understandable as a control signal.

A simple test shot is enough:

> One character walks from Point A to Point B while interacting with one object, while the camera tracks and/or orbits around the action.

Once this works reliably, expand the API rather than overbuilding the initial prototype.

---

# Key Success Criterion

The system succeeds if the user can think like a director rather than a Blender operator.

The ideal interaction is:

> **Describe shot → see rough blocking → give notes → see revision → approve → send control video downstream.**

The purpose of Blender is not to make the movie.

The purpose of Blender is to give the generative model **something deterministic to follow**.