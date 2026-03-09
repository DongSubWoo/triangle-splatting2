"""
Orbit Capture Rig for Unreal Engine 5.6
========================================
Automatically captures high-res screenshots from multiple camera positions
orbiting around a target point. Designed for triangle splatting / 3D reconstruction.

Requirements:
  - Enable the "PythonAutomationTest" plugin (Edit > Plugins > search "PythonAutomationTest")
  - Enable the "Editor Scripting Utilities" plugin
  - Restart editor after enabling plugins

Usage:
  1. Open your level in UE5.6 editor with lighting built
  2. Select the actor you want to capture (or edit TARGET_LOCATION below)
  3. Run via: File > Execute Python Script > select this file

  Screenshots & JSON go to:
    Project/Saved/CaptureRig/<LevelName>/<YYYY-MM-DD_HH-MM-SS>/

Configuration:
  Edit the settings in the CONFIG section below.
"""

import unreal
import math
import json
import os
import shutil
import glob
import random
from datetime import datetime

# ============================================================================
# CONFIG - Edit these to match your needs
# ============================================================================

# Target point to orbit around (world space)
# Set to None to use the currently selected actor's location
TARGET_LOCATION = unreal.Vector(0, 0, 0)

# --------------------------------------------------------------------------
# RADIUS
# --------------------------------------------------------------------------
BASE_ORBIT_RADIUS = 3914.0
RADIUS_MULTIPLIER = 1.7

HORIZONTAL_STEPS  = 12*3   # shots per ring at equator (scaled by cos(elevation))
VERTICAL_STEPS    = 3*3         # rings per dome half (equator→pole, exclusive of 0°)
LOWER_DOME_MULTIPLIER = 0.0    # 0 = skip all below-horizon shots (not visible in-game)

# --------------------------------------------------------------------------
# ADAPTIVE DENSITY BOOST (around game camera elevation)
# --------------------------------------------------------------------------
GAME_CAMERA_ELEVATION = 46.0

ADAPTIVE_HORIZONTAL_MULTIPLIER = 2.0  # was 1.5
ADAPTIVE_HORIZONTAL_RANGE = 25.0      # was 20.0

ADAPTIVE_ELEVATION_MULTIPLIER = 2.0   # was 1.5
ADAPTIVE_ELEVATION_RANGE = 25.0       # was 20.0

# --------------------------------------------------------------------------
# CAMERA
# --------------------------------------------------------------------------
SCREENSHOT_WIDTH = 1024
SCREENSHOT_HEIGHT = 1024
CAMERA_FOV = 49.773628          # vertical FOV in degrees (Maintain Y-Axis FOV)
CAPTURE_DELAY_SECONDS = 0.75
OUTPUT_SUBDIR = "CaptureRig"
EXPORT_CAMERA_JSON = True

# --------------------------------------------------------------------------
# NERF TRAIN/TEST SPLIT
# --------------------------------------------------------------------------
NERF_TRAIN_RATIO = 0.8
NERF_RANDOM_SEED = 42

# --------------------------------------------------------------------------
# NORMAL MAP CAPTURE
# --------------------------------------------------------------------------
# Set to True when capturing normal maps (second pass).
# Before running: manually assign your unlit vertex-normal material to the scene.
# Screenshots will go to NORMAL_OUTPUT_DIR/normals/ with matching filenames.
# Transforms JSON will NOT be re-exported.
#
# NORMAL_OUTPUT_DIR:
#   - Leave empty ("") to automatically use the last RGB capture folder.
#     The last RGB capture dir is saved to:
#       T:\git\triangle-splatting2\Unreal_Export\.last_capture_dir.json
#   - Set explicitly to override, e.g.:
#       r"T:\git\triangle-splatting2\Unreal_Export\P_Base\2026-03-06_19-29-31"
CAPTURE_NORMALS = False
NORMAL_OUTPUT_DIR = r""

# ============================================================================
# COMPUTED CONFIG (derived from above — do not edit)
# ============================================================================

ORBIT_RADIUS = BASE_ORBIT_RADIUS * RADIUS_MULTIPLIER

# Compute elevation angles
_step = 90.0 / (VERTICAL_STEPS + 1)
ELEVATION_ANGLES = (
    [-90.0 + i * _step for i in range(VERTICAL_STEPS + 1)]
    + [0.0]
    + [_step * (i + 1) for i in range(VERTICAL_STEPS + 1)]
)

# Adaptive elevation: insert extra rings in game camera zone
if ADAPTIVE_ELEVATION_MULTIPLIER > 1.0:
    zone_min = GAME_CAMERA_ELEVATION - ADAPTIVE_ELEVATION_RANGE
    zone_max = GAME_CAMERA_ELEVATION + ADAPTIVE_ELEVATION_RANGE
    extra_angles = []
    sorted_angles = sorted(ELEVATION_ANGLES)
    for i in range(len(sorted_angles) - 1):
        a = sorted_angles[i]
        b = sorted_angles[i + 1]
        if a >= zone_max or b <= zone_min:
            continue
        gap_in_zone_start = max(a, zone_min)
        gap_in_zone_end = min(b, zone_max)
        if gap_in_zone_end - gap_in_zone_start < 1.0:
            continue
        num_inserts = max(1, round(ADAPTIVE_ELEVATION_MULTIPLIER - 1.0))
        for j in range(1, num_inserts + 1):
            frac = j / (num_inserts + 1)
            new_angle = a + (b - a) * frac
            if zone_min <= new_angle <= zone_max:
                extra_angles.append(round(new_angle, 2))
    ELEVATION_ANGLES = sorted(set(ELEVATION_ANGLES + extra_angles))


# ============================================================================
# NERF CONVERSION HELPERS
# ============================================================================

def _matmul3(A, B):
    """3x3 matrix multiply (pure Python, no numpy)."""
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def unreal_pyr_to_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """
    Convert Unreal Engine pitch/yaw/roll (degrees) to a 3x3 rotation matrix.

    Unreal Engine conventions:
      - Left-handed coordinate system: X=forward, Y=right, Z=up
      - Rotation order: Yaw (around Z) -> Pitch (around Y) -> Roll (around X)
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)

    cy, sy = math.cos(y), math.sin(y)
    Ryaw = [
        [ cy, -sy, 0],   # UE left-handed: positive yaw = turn right (X→+Y)
        [ sy,  cy, 0],
        [  0,   0, 1],
    ]

    cp, sp = math.cos(p), math.sin(p)
    Rpitch = [
        [ cp, 0, -sp],   # UE left-handed: positive pitch = nose up (X→+Z)
        [  0, 1,   0],
        [ sp, 0,  cp],
    ]

    cr, sr = math.cos(r), math.sin(r)
    Rroll = [
        [1,   0,    0],
        [0,  cr,  -sr],
        [0,  sr,   cr],
    ]

    # Combined: Yaw first, then Pitch, then Roll
    return _matmul3(_matmul3(Ryaw, Rpitch), Rroll)


def _remap_ue_to_nerf(ux, uy, uz):
    """
    Remap a vector from UE world space to NeRF world space.
    UE (LH): X=forward, Y=right, Z=up
    NeRF (RH, OpenGL): X=right, Y=up, Z=backward
    Mapping: (uy, uz, -ux)  — matches M matrix in convert_unreal_to_nerf.py
    Applied consistently to both position and rotation axes.
    """
    return (uy, uz, -ux)


def unreal_to_nerf_c2w(position_cm, rotation_pyr):
    """
    Convert Unreal Engine camera pose to a NeRF camera-to-world matrix (4x4).

    Coordinate system mapping:
      Unreal (left-handed):  X=forward, Y=right, Z=up
      NeRF/OpenGL (right-handed): X=right, Y=up, Z=backward (out of screen)

    The same axis remap  (ux,uy,uz) -> (uy, uz, -ux)  is applied to BOTH
    position and each camera axis vector. This is essential: the rotation
    columns live in world space and must be remapped the same way as position.

      nerf_right    =  remap(right_ue)     UE +Y -> NeRF +X
      nerf_up       =  remap(up_ue)        UE +Z -> NeRF +Y
      nerf_backward = -remap(forward_ue)   UE +X -> NeRF -Z (camera looks along -Z)

    det=+1 verified, backward vector aligns with position (away from origin) verified.
    Position units: centimeters -> meters (divide by 100).
    """
    pitch, yaw, roll = rotation_pyr
    R = unreal_pyr_to_rotation_matrix(pitch, yaw, roll)

    # Extract camera axes as columns of the UE rotation matrix (world-space directions)
    forward_ue = (R[0][0], R[1][0], R[2][0])   # UE camera +X in world
    right_ue   = (R[0][1], R[1][1], R[2][1])   # UE camera +Y in world
    up_ue      = (R[0][2], R[1][2], R[2][2])   # UE camera +Z in world

    # Remap each axis vector from UE world to NeRF world
    r_fwd = _remap_ue_to_nerf(*forward_ue)
    r_rgt = _remap_ue_to_nerf(*right_ue)
    r_up  = _remap_ue_to_nerf(*up_ue)

    nerf_right    = ( r_rgt[0],  r_rgt[1],  r_rgt[2])   # remap(right_ue)
    nerf_up       = ( r_up[0],   r_up[1],   r_up[2])   # remap(up_ue)
    nerf_backward = (-r_fwd[0], -r_fwd[1], -r_fwd[2])  # -remap(forward_ue): fwd -> bwd

    # Remap position using the same transform, then convert cm -> m
    px, py, pz = position_cm
    pr = _remap_ue_to_nerf(px, py, pz)
    pos_nerf = (pr[0] / 100.0, pr[1] / 100.0, pr[2] / 100.0)

    # Build 4x4 c2w: columns are [right, up, backward, translation]
    c2w = [
        [nerf_right[0], nerf_up[0], nerf_backward[0], pos_nerf[0]],
        [nerf_right[1], nerf_up[1], nerf_backward[1], pos_nerf[1]],
        [nerf_right[2], nerf_up[2], nerf_backward[2], pos_nerf[2]],
        [0.0,           0.0,        0.0,               1.0        ],
    ]
    return c2w


def export_nerf_transforms(output_dir, all_metadata, cam_data):
    """
    Write transforms_train.json, transforms_test.json, and transforms.json
    in NeRF/Blender format, ready for triangle-splatting2's train.py.

    Note: triangle-splatting2's dataset_readers.py applies c2w[:3,1:3] *= -1
    after loading (flipping Y and Z columns), converting from OpenGL/Blender
    convention to COLMAP convention internally. Our output is correct as-is.
    """
    fov_x = math.radians(cam_data["fov_x_deg"])
    fov_y = math.radians(cam_data["fov_y_deg"])
    W, H  = cam_data["resolution"]
    fx    = cam_data["intrinsics"]["fx"]
    fy    = cam_data["intrinsics"]["fy"]
    cx    = cam_data["intrinsics"]["cx"]
    cy    = cam_data["intrinsics"]["cy"]

    frames = []
    for shot in all_metadata:
        c2w = unreal_to_nerf_c2w(shot["position"], shot["rotation_pyr"])
        frames.append({
            "file_path": shot["filename"],   # already includes .png extension
            "transform_matrix": c2w,
        })

    base = {
        "camera_angle_x": fov_x,
        "camera_angle_y": fov_y,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": W,
        "h": H,
    }

    # Deterministic shuffle before split
    random.seed(NERF_RANDOM_SEED)
    shuffled = frames[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * NERF_TRAIN_RATIO)
    train_frames = shuffled[:split]
    test_frames  = shuffled[split:]

    for filename, subset in [
        ("transforms_train.json", train_frames),
        ("transforms_test.json",  test_frames),
        ("transforms.json",       frames),       # unshuffled, all frames
    ]:
        out = dict(base)
        out["frames"] = subset
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(out, f, indent=2)

    unreal.log(f"NeRF exports: {len(train_frames)} train / {len(test_frames)} test")
    unreal.log(f"  -> transforms_train.json")
    unreal.log(f"  -> transforms_test.json")
    unreal.log(f"  -> transforms.json")


# ============================================================================
# UNREAL HELPERS
# ============================================================================

def get_target_location():
    if TARGET_LOCATION is not None:
        return TARGET_LOCATION
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    selected = actor_subsystem.get_selected_level_actors()
    if len(selected) > 0:
        loc = selected[0].get_actor_location()
        unreal.log(f"Using selected actor location: {loc}")
        return loc
    unreal.log_warning("No actor selected and TARGET_LOCATION is None. Using origin.")
    return unreal.Vector(0, 0, 0)


def compute_camera_transforms(target, radius, base_horizontal, elevation_angles):
    transforms = []
    for elev_idx, elev_deg in enumerate(elevation_angles):
        elev_rad = math.radians(elev_deg)

        dist_from_game_cam = abs(elev_deg - GAME_CAMERA_ELEVATION)
        if ADAPTIVE_HORIZONTAL_MULTIPLIER > 1.0 and dist_from_game_cam < ADAPTIVE_HORIZONTAL_RANGE:
            falloff = math.cos(dist_from_game_cam / ADAPTIVE_HORIZONTAL_RANGE * math.pi * 0.5)
            multiplier = 1.0 + (ADAPTIVE_HORIZONTAL_MULTIPLIER - 1.0) * falloff
        else:
            multiplier = 1.0

        cos_elev = math.cos(elev_rad)
        lower_mult = LOWER_DOME_MULTIPLIER if elev_deg < 0 else 1.0
        if abs(elev_deg) >= 89.0:
            ring_shots = 1
        else:
            ring_shots = max(1, round(base_horizontal * cos_elev * multiplier * lower_mult))

        for h_idx in range(ring_shots):
            yaw_deg = (h_idx / ring_shots) * 360.0
            yaw_rad = math.radians(yaw_deg)

            x = radius * math.cos(elev_rad) * math.cos(yaw_rad)
            y = radius * math.cos(elev_rad) * math.sin(yaw_rad)
            z = radius * math.sin(elev_rad)

            cam_loc = unreal.Vector(target.x + x, target.y + y, target.z + z)
            cam_rot = unreal.MathLibrary.find_look_at_rotation(cam_loc, target)

            metadata = {
                "index":              len(transforms),
                "ring":               elev_idx,
                "ring_shots":         ring_shots,
                "adaptive_multiplier": round(multiplier, 2),
                "shot_in_ring":       h_idx,
                "elevation_deg":      elev_deg,
                "azimuth_deg":        yaw_deg,
                "position":           [cam_loc.x, cam_loc.y, cam_loc.z],
                "rotation_pyr":       [cam_rot.pitch, cam_rot.yaw, cam_rot.roll],
                "fov":                CAMERA_FOV,
                "resolution":         [SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT],
            }
            transforms.append((cam_loc, cam_rot, metadata))
    return transforms


def get_level_name():
    try:
        editor_subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        editor_world = editor_subsystem.get_editor_world()
        return editor_world.get_name()
    except Exception:
        return "UnknownLevel"


_LAST_CAPTURE_FILE = os.path.join(r"T:\git\triangle-splatting2\Unreal_Export", ".last_capture_dir.json")

def save_last_capture_dir(output_dir):
    with open(_LAST_CAPTURE_FILE, "w") as f:
        json.dump({"last_capture_dir": output_dir}, f)

def load_last_capture_dir():
    if not os.path.exists(_LAST_CAPTURE_FILE):
        raise RuntimeError("No previous RGB capture found. Run RGB capture first.")
    with open(_LAST_CAPTURE_FILE) as f:
        return json.load(f)["last_capture_dir"]

def get_output_dir():
    if CAPTURE_NORMALS:
        output_dir = NORMAL_OUTPUT_DIR if NORMAL_OUTPUT_DIR else load_last_capture_dir()
        normals_dir = os.path.join(output_dir, "normals")
        os.makedirs(normals_dir, exist_ok=True)
        unreal.log(f"Normal capture -> {normals_dir}")
        return output_dir
    level_name = get_level_name()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(r"T:\git\triangle-splatting2\Unreal_Export", level_name, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_last_capture_dir(output_dir)
    return output_dir


def get_screenshot_source_dir():
    return os.path.join(unreal.Paths.project_saved_dir(), "Screenshots", "WindowsEditor")


# ============================================================================
# GLOBAL STATE (shared across latent commands)
# ============================================================================

capture_state = {
    "camera_actor": None,
    "transforms":   [],
    "all_metadata": [],
    "target":       None,
    "output_dir":   "",
}


# ============================================================================
# LATENT COMMANDS
# ============================================================================

@unreal.AutomationScheduler.add_latent_command
def setup_capture():
    """Set up camera actor and compute all shot positions."""
    target = get_target_location()
    capture_state["target"]     = target
    capture_state["output_dir"] = get_output_dir()

    transforms = compute_camera_transforms(
        target, ORBIT_RADIUS, HORIZONTAL_STEPS, ELEVATION_ANGLES
    )
    capture_state["transforms"] = transforms

    total = len(transforms)
    unreal.log(f"=== Orbit Capture Rig ===")
    unreal.log(f"Target:      ({target.x:.1f}, {target.y:.1f}, {target.z:.1f})")
    unreal.log(f"Radius:      {ORBIT_RADIUS:.1f} cm")
    unreal.log(f"Total shots: {total}")
    unreal.log(f"Rings:       {len(ELEVATION_ANGLES)}")
    unreal.log(f"Resolution:  {SCREENSHOT_WIDTH}x{SCREENSHOT_HEIGHT}")
    unreal.log(f"Output:      {capture_state['output_dir']}")
    unreal.log(f"========================")

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    camera_actor = actor_subsystem.spawn_actor_from_class(
        unreal.CameraActor,
        target,
        unreal.Rotator(0, 0, 0),
    )
    camera_actor.set_actor_label("OrbitCaptureCamera")

    camera_component = camera_actor.get_editor_property("camera_component")
    camera_component.set_editor_property("field_of_view", CAMERA_FOV)
    camera_component.set_editor_property("aspect_ratio", SCREENSHOT_WIDTH / SCREENSHOT_HEIGHT)
    camera_component.set_editor_property("constrain_aspect_ratio", False)
    try:
        camera_component.set_editor_property(
            "aspect_ratio_axis_constraint",
            unreal.AspectRatioAxisConstraint.ASPECT_RATIO_MAINTAIN_YFOV,
        )
        camera_component.set_editor_property("override_aspect_ratio_axis_constraint", True)
        unreal.log("Aspect ratio constraint: Maintain Y-Axis FOV")
    except Exception as e:
        unreal.log_warning(f"Could not set aspect_ratio_axis_constraint: {e}")

    capture_state["camera_actor"] = camera_actor
    capture_state["all_metadata"] = []
    unreal.log("Camera spawned. Starting capture sequence...")


@unreal.AutomationScheduler.add_latent_command
def run_captures():
    """Move camera to each position, wait, and capture screenshot."""
    camera_actor = capture_state["camera_actor"]
    transforms   = capture_state["transforms"]
    total        = len(transforms)

    for idx, (cam_loc, cam_rot, metadata) in enumerate(transforms):
        camera_actor.set_actor_location(cam_loc, sweep=False, teleport=True)
        camera_actor.set_actor_rotation(cam_rot, teleport_physics=True)

        unreal.log(
            f"Shot {idx + 1}/{total}  |  "
            f"elev={metadata['elevation_deg']:6.1f}  "
            f"azim={metadata['azimuth_deg']:5.1f}"
        )

        # Let editor process the move
        yield

        # Wait for Lumen/Nanite to converge
        delay_ticks = max(1, int(CAPTURE_DELAY_SECONDS * 30))
        for _ in range(delay_ticks):
            yield

        # Build filename: 0001_elev030_azim090.png  (elevN10 for negative)
        elev = metadata['elevation_deg']
        if elev < 0:
            elev_str = f"elevN{abs(elev):02.0f}"
        else:
            elev_str = f"elev{elev:03.0f}"
        azim_str = f"azim{metadata['azimuth_deg']:03.0f}"
        screenshot_name = f"{idx:04d}_{elev_str}_{azim_str}"

        task = unreal.AutomationLibrary.take_high_res_screenshot(
            SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT, screenshot_name, camera=camera_actor
        )

        if task.is_valid_task():
            while not task.is_task_done():
                yield
            unreal.log(f"  -> Saved: {screenshot_name}.png")
        else:
            unreal.log_warning(f"  -> FAILED: {screenshot_name}")

        metadata["filename"] = screenshot_name + ".png"
        capture_state["all_metadata"].append(metadata)

        yield  # extra tick between shots


@unreal.AutomationScheduler.add_latent_command
def finalize_capture():
    """Move screenshots, export camera_transforms.json and NeRF transforms."""
    output_dir         = capture_state["output_dir"]
    screenshot_src_dir = get_screenshot_source_dir()

    # ------------------------------------------------------------------
    # Move screenshots from UE default folder to our organized folder
    # (normals go to output_dir/normals/, RGB goes to output_dir/)
    # ------------------------------------------------------------------
    dest_dir = os.path.join(output_dir, "normals") if CAPTURE_NORMALS else output_dir
    moved_count = 0
    for shot_meta in capture_state["all_metadata"]:
        src_filename = shot_meta["filename"]
        src_pattern  = os.path.join(
            screenshot_src_dir, src_filename.replace(".png", "") + "*"
        )
        matching_files = glob.glob(src_pattern)
        if not matching_files:
            src_path = os.path.join(screenshot_src_dir, src_filename)
            if os.path.exists(src_path):
                matching_files = [src_path]
        for src_path in matching_files:
            dst_path = os.path.join(dest_dir, os.path.basename(src_path))
            try:
                shutil.move(src_path, dst_path)
                moved_count += 1
            except Exception as e:
                unreal.log_warning(f"Could not move {src_path}: {e}")
    unreal.log(f"Moved {moved_count} screenshots to: {dest_dir}")

    # ------------------------------------------------------------------
    # Build camera intrinsics
    # ------------------------------------------------------------------
    fov_y_rad = math.radians(CAMERA_FOV)
    aspect    = SCREENSHOT_WIDTH / SCREENSHOT_HEIGHT
    fov_x_rad = 2.0 * math.atan(math.tan(fov_y_rad / 2.0) * aspect)
    fy = (SCREENSHOT_HEIGHT / 2.0) / math.tan(fov_y_rad / 2.0)
    fx = (SCREENSHOT_WIDTH  / 2.0) / math.tan(fov_x_rad / 2.0)
    cx = SCREENSHOT_WIDTH  / 2.0
    cy = SCREENSHOT_HEIGHT / 2.0

    cam_data = {
        "fov_y_deg":    CAMERA_FOV,
        "fov_x_deg":    math.degrees(fov_x_rad),
        "aspect_ratio": aspect,
        "resolution":   [SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT],
        "intrinsics": {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "note": "pinhole model, no distortion",
        },
        "intrinsic_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    }

    # ------------------------------------------------------------------
    # Export camera_transforms.json and NeRF transforms (RGB pass only)
    # ------------------------------------------------------------------
    if CAPTURE_NORMALS:
        unreal.log("Normal capture mode: skipping camera JSON / NeRF transforms export.")
    elif EXPORT_CAMERA_JSON:
        tgt = capture_state["target"]
        json_data = {
            "level_name":        get_level_name(),
            "capture_timestamp": os.path.basename(output_dir),
            "target":            [tgt.x, tgt.y, tgt.z],
            "orbit_radius":      ORBIT_RADIUS,
            "base_radius":       BASE_ORBIT_RADIUS,
            "radius_multiplier": RADIUS_MULTIPLIER,
            "camera":            cam_data,
            "coordinate_system": {
                "note":  "Unreal Engine: X=forward, Y=right, Z=up, left-handed",
                "units": "centimeters",
            },
            "num_shots": len(capture_state["all_metadata"]),
            "shots":     capture_state["all_metadata"],
        }
        json_path = os.path.join(output_dir, "camera_transforms.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        unreal.log(f"Unreal camera transforms -> {json_path}")

    # ------------------------------------------------------------------
    # Export NeRF transforms (train/test split, ready for train.py)
    # ------------------------------------------------------------------
    if not CAPTURE_NORMALS:
        export_nerf_transforms(output_dir, capture_state["all_metadata"], cam_data)

    # ------------------------------------------------------------------
    # Clean up spawned camera actor
    # ------------------------------------------------------------------
    camera_actor = capture_state["camera_actor"]
    if camera_actor is not None:
        try:
            camera_actor.destroy_actor()
            unreal.log("Capture camera removed.")
        except Exception as e:
            unreal.log_warning(f"Could not destroy camera: {e}")

    total = len(capture_state["all_metadata"])
    unreal.log(f"=== Capture Complete: {total} shots ===")
    unreal.log(f"Output folder: {output_dir}")
    unreal.log(f'Train command: python train.py -s "{output_dir}" -m <output_path> --eval')