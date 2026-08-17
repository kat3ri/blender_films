"""BVH parsing and sampling, stdlib-only.

This module intentionally does not import bpy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class BVHJoint:
    name: str
    parent: str | None
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    channel_indices: tuple[int, ...]


@dataclass
class BVHClip:
    source_path: Path
    root_joint: str
    joints: dict[str, BVHJoint]
    frame_time_s: float
    frames: list[list[float]]

    @property
    def frame_count(self):
        return len(self.frames)

    @property
    def duration_seconds(self):
        if not self.frames:
            return 0.0
        return max(0.0, (len(self.frames) - 1) * self.frame_time_s)

    def sample_joint_rotations(self, t_s):
        """Return per-joint XYZ Euler degrees at time t_s.

        BVH stores rotations in per-joint channel order (often not XYZ). We
        compose that order as a matrix, then convert to equivalent XYZ Euler.
        """
        i0, i1, u = self._sample_indices(t_s)
        f0 = self.frames[i0]
        f1 = self.frames[i1]
        out = {}
        for joint_name, joint in self.joints.items():
            channels0 = []
            channels1 = []
            for channel, index in zip(joint.channels, joint.channel_indices):
                value0 = f0[index]
                value1 = f1[index]
                if channel.endswith("rotation"):
                    channels0.append((channel[0], value0))
                    channels1.append((channel[0], value1))
            e0 = _ordered_channels_to_xyz_euler(channels0)
            e1 = _ordered_channels_to_xyz_euler(channels1)
            out[joint_name] = [_blend_angle_deg(e0[i], e1[i], u) for i in range(3)]
        return out

    def sample_root_translation(self, t_s):
        """Return root translation channels as [x, y, z], or None if absent."""
        root = self.joints[self.root_joint]
        i0, i1, u = self._sample_indices(t_s)
        f0 = self.frames[i0]
        f1 = self.frames[i1]
        xyz0 = [0.0, 0.0, 0.0]
        xyz1 = [0.0, 0.0, 0.0]
        has_translation = False
        for channel, index in zip(root.channels, root.channel_indices):
            value0 = f0[index]
            value1 = f1[index]
            if channel == "Xposition":
                xyz0[0], xyz1[0], has_translation = value0, value1, True
            elif channel == "Yposition":
                xyz0[1], xyz1[1], has_translation = value0, value1, True
            elif channel == "Zposition":
                xyz0[2], xyz1[2], has_translation = value0, value1, True
        if not has_translation:
            return None
        return [xyz0[i] + (xyz1[i] - xyz0[i]) * u for i in range(3)]

    def _sample_indices(self, t_s):
        if not self.frames:
            return 0, 0, 0.0
        if self.frame_count == 1 or self.frame_time_s <= 1e-9:
            return 0, 0, 0.0
        t = max(0.0, min(float(t_s), self.duration_seconds))
        frame_f = t / self.frame_time_s
        i0 = int(frame_f)
        i1 = min(i0 + 1, self.frame_count - 1)
        return i0, i1, frame_f - i0


def load_bvh(path):
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    index = 0
    channel_cursor = 0
    joints = {}

    def fail(message):
        raise ValueError(f"{path}: {message}")

    def next_line():
        nonlocal index
        while index < len(lines) and not lines[index]:
            index += 1
        if index >= len(lines):
            return None
        line = lines[index]
        index += 1
        return line

    def parse_joint(header, parent):
        nonlocal channel_cursor
        parts = header.split()
        if len(parts) < 2:
            fail(f"invalid joint header: {header!r}")
        kind, name = parts[0], parts[1]
        if kind not in ("ROOT", "JOINT"):
            fail(f"expected ROOT/JOINT, got {kind!r}")

        if next_line() != "{":
            fail(f"expected '{{' after {name}")

        offset = (0.0, 0.0, 0.0)
        channels = ()
        channel_indices = ()

        while True:
            line = next_line()
            if line is None:
                fail(f"unterminated joint block for {name}")
            if line == "}":
                break
            if line.startswith("OFFSET "):
                nums = line.split()[1:]
                if len(nums) != 3:
                    fail(f"OFFSET must have 3 numbers for {name}")
                offset = (float(nums[0]), float(nums[1]), float(nums[2]))
                continue
            if line.startswith("CHANNELS "):
                parts2 = line.split()
                count = int(parts2[1])
                names = tuple(parts2[2:])
                if len(names) != count:
                    fail(f"CHANNELS count mismatch for {name}")
                channels = names
                channel_indices = tuple(range(channel_cursor, channel_cursor + count))
                channel_cursor += count
                continue
            if line.startswith("JOINT "):
                parse_joint(line, parent=name)
                continue
            if line == "End Site":
                if next_line() != "{":
                    fail("expected '{' after End Site")
                depth = 1
                while depth > 0:
                    sub = next_line()
                    if sub is None:
                        fail("unterminated End Site block")
                    if sub == "{":
                        depth += 1
                    elif sub == "}":
                        depth -= 1
                continue
            fail(f"unexpected line in {name}: {line!r}")

        joints[name] = BVHJoint(
            name=name,
            parent=parent,
            offset=offset,
            channels=channels,
            channel_indices=channel_indices,
        )

    first = next_line()
    if first != "HIERARCHY":
        fail("missing HIERARCHY header")
    root_header = next_line()
    if root_header is None or not root_header.startswith("ROOT "):
        fail("missing ROOT declaration")
    root_name = root_header.split()[1]
    parse_joint(root_header, parent=None)

    motion = next_line()
    if motion != "MOTION":
        fail("missing MOTION section")

    frames_line = next_line()
    if not frames_line or not frames_line.startswith("Frames:"):
        fail("missing Frames: line")
    frame_count = int(frames_line.split(":", 1)[1].strip())

    frame_time_line = next_line()
    if not frame_time_line or not frame_time_line.startswith("Frame Time:"):
        fail("missing Frame Time: line")
    frame_time_s = float(frame_time_line.split(":", 1)[1].strip())

    channel_count = 0
    for joint in joints.values():
        channel_count = max(channel_count, max(joint.channel_indices, default=-1) + 1)

    frames = []
    for frame_index in range(frame_count):
        line = next_line()
        if line is None:
            fail(f"expected {frame_count} frame rows, got {frame_index}")
        values = [float(v) for v in line.split()]
        if len(values) != channel_count:
            fail(
                f"frame {frame_index} has {len(values)} values, expected {channel_count}"
            )
        frames.append(values)

    return BVHClip(
        source_path=path,
        root_joint=root_name,
        joints=joints,
        frame_time_s=frame_time_s,
        frames=frames,
    )


def _blend_angle_deg(a, b, u):
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + delta * u


def _ordered_channels_to_xyz_euler(channels):
    # Compose ordered axis rotations as R = R * R_axis.
    matrix = _identity3()
    for axis, angle_deg in channels:
        r = _axis_matrix(axis, math.radians(angle_deg))
        matrix = _mul3(matrix, r)
    return _matrix_to_euler_xyz_deg(matrix)


def _identity3():
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _mul3(a, b):
    return [
        [a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]
         for col in range(3)]
        for row in range(3)
    ]


def _axis_matrix(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "X":
        return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    if axis == "Y":
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    if axis == "Z":
        return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return _identity3()


def _matrix_to_euler_xyz_deg(m):
    # Decompose as M = Rz . Ry . Rx — Blender's 'XYZ' Euler convention
    # (rotations applied X, then Y, then Z about fixed axes). These angles are
    # ultimately assigned to rotation_euler, so the decomposition here MUST
    # match how Blender recomposes them; an Rx.Ry.Rz decomposition agrees only
    # for single-axis rotations and scrambles real multi-axis mocap frames.
    if abs(m[2][0]) < 0.999999:
        y = -math.asin(m[2][0])
        x = math.atan2(m[2][1], m[2][2])
        z = math.atan2(m[1][0], m[0][0])
    else:
        # Gimbal lock fallback.
        y = -math.pi / 2.0 if m[2][0] >= 0 else math.pi / 2.0
        x = math.atan2(-m[1][2], m[1][1])
        z = 0.0
    return [math.degrees(x), math.degrees(y), math.degrees(z)]