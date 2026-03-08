import re

with open('train.py', 'r') as f:
    src = f.read()

# Step 1: Restore any mangled lines from previous patch attempts
src = src.replace(
    'gt_normal = viewpoint_cam.normal_map.cuda() if viewpoint_cam.normal_map is not None else None',
    'gt_normal = viewpoint_cam.normal_map.cuda()'
)
src = src.replace(
    'seg_hr = gt_normal.unsqueeze(0) if gt_normal is not None else None if gt_normal is not None else None  # -> [1, 3, H, W]',
    'seg_hr = gt_normal.unsqueeze(0)  # -> [1, 3, H, W]'
)
src = src.replace(
    'seg_ds_area = F.interpolate(seg_hr, if seg_hr is not None else None # size=(gt_image.shape[1], gt_image.shape[2]), mode="area")  # [1, 3, H0, W0]',
    'seg_ds_area = F.interpolate(seg_hr, size=(gt_image.shape[1], gt_image.shape[2]), mode="area")  # [1, 3, H0, W0]'
)

# Step 2: Replace the 4-line normal block with a guarded version
old_block = (
    '        gt_normal = viewpoint_cam.normal_map.cuda()\n'
    '        seg_hr = gt_normal.unsqueeze(0)  # -> [1, 3, H, W]\n'
    '        seg_ds_area = F.interpolate(seg_hr, size=(gt_image.shape[1], gt_image.shape[2]), mode="area")  # [1, 3, H0, W0]\n'
    '        gt_normal = seg_ds_area.squeeze(0)  # -> [3, H0, W0]\n'
)
new_block = (
    '        _nm = viewpoint_cam.normal_map\n'
    '        if _nm is not None:\n'
    '            _nm = _nm.cuda()\n'
    '            _nm = F.interpolate(_nm.unsqueeze(0), size=(gt_image.shape[1], gt_image.shape[2]), mode="area").squeeze(0)\n'
    '        gt_normal = _nm\n'
)

if old_block in src:
    src = src.replace(old_block, new_block)
    print("OK: main block patched.")
else:
    print("WARNING: main block not found - printing lines 130-160:")
    for i, line in enumerate(src.splitlines()[130:160], start=131):
        print(f"  {i}: {line}")

# Step 3: Guard any normal_loss lines
src = re.sub(
    r'^(        )((?!if gt_normal).*normal_loss.*\n)',
    r'\1if gt_normal is not None:\n\1    \2',
    src, flags=re.MULTILINE
)

with open('train.py', 'w') as f:
    f.write(src)

print("Done.")
