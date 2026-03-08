"""
Converts Unreal Engine camera_transforms.json to NeRF transforms.json format
Usage: python convert_unreal_to_nerf.py <path_to_folder>
The folder should contain camera_transforms.json and all the .png images.
Output: transforms.json in the same folder (train/test split included)
"""

import json
import math
import numpy as np
import os
import random
import sys

def unreal_to_nerf_matrix(position, rotation_pyr):
    """
    Convert Unreal Engine camera pose to NeRF c2w matrix.
    Unreal: X=forward, Y=right, Z=up, left-handed, units=cm
    NeRF:   X=right, Y=up, Z=back, right-handed, units=meters (or arbitrary)
    rotation_pyr = [pitch, yaw, roll] in degrees
    """
    pitch = math.radians(rotation_pyr[0])
    yaw   = math.radians(rotation_pyr[1])
    roll  = math.radians(rotation_pyr[2])

    # Rotation matrices (Unreal convention)
    Rz = np.array([
        [ math.cos(yaw), -math.sin(yaw), 0],
        [ math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [ math.cos(pitch), 0, -math.sin(pitch)],   # UE left-handed: positive pitch = nose up
        [0, 1, 0],
        [ math.sin(pitch), 0,  math.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0,  math.cos(roll), -math.sin(roll)],
        [0,  math.sin(roll),  math.cos(roll)]
    ])
    R_ue = Rz @ Ry @ Rx

    # Unreal forward=X, right=Y, up=Z → NeRF right=X, up=Y, back=-Z
    # Coordinate change matrix
    M = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0]
    ])
    R_nerf = M @ R_ue @ M.T

    # Position: convert cm to m, apply coordinate change
    pos_ue = np.array(position) / 100.0  # cm to m
    pos_nerf = M @ pos_ue

    # Build 4x4 c2w matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R_nerf
    c2w[:3, 3]  = pos_nerf

    return c2w.tolist()


def convert(folder):
    json_path = os.path.join(folder, "camera_transforms.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    cam = data["camera"]
    fov_x = math.radians(cam["fov_x_deg"])
    fov_y = math.radians(cam["fov_y_deg"])
    W, H  = cam["resolution"]
    fx    = cam["intrinsics"]["fx"]
    fy    = cam["intrinsics"]["fy"]
    cx    = cam["intrinsics"]["cx"]
    cy    = cam["intrinsics"]["cy"]

    frames = []
    for shot in data["shots"]:
        c2w = unreal_to_nerf_matrix(shot["position"], shot["rotation_pyr"])
        frame = {
            "file_path": shot["filename"],
            "transform_matrix": c2w
        }
        frames.append(frame)

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

    # Train/test split (deterministic)
    random.seed(42)
    shuffled = frames[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    train_frames = shuffled[:split]
    test_frames  = shuffled[split:]

    for filename, subset in [
        ("transforms.json",       frames),
        ("transforms_train.json", train_frames),
        ("transforms_test.json",  test_frames),
    ]:
        out = dict(base)
        out["frames"] = subset
        out_path = os.path.join(folder, filename)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    print(f"Done! {len(frames)} total | {len(train_frames)} train | {len(test_frames)} test")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_unreal_to_nerf.py <folder_path>")
        sys.exit(1)
    convert(sys.argv[1])
