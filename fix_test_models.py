import os
import torch
import torch.nn as nn

# Read the file
with open('src/test_models.py', 'r') as f:
    lines = f.readlines()

# Replace the build_model_from_state_dict function and enhance discover_models
new_lines = []
skip = False
for line in lines:
    if 'def build_model_from_state_dict(state_dict):' in line:
        new_lines.append('def build_model_from_state_dict(state_dict, filters=None, kernel_size=5):\n')
        new_lines.append('    model = DeepCFDCompat(filters=filters, kernel_size=kernel_size)\n')
        new_lines.append('    model.load_state_dict(state_dict, strict=True)\n')
        new_lines.append('    return model\n')
        skip = True
        continue
    if skip and line.startswith('def '):
        skip = False
    if skip:
        continue
        
    if 'discovered["deepcfd_data"] = build_model_from_state_dict(ckpt["state_dict"]).to(device)' in line:
        new_lines.append('            discovered["deepcfd_data"] = build_model_from_state_dict(\n')
        new_lines.append('                ckpt["state_dict"], \n')
        new_lines.append('                filters=ckpt.get("filters"), \n')
        new_lines.append('                kernel_size=ckpt.get("kernel_size", 5)\n')
        new_lines.append('            ).to(device)\n')
        continue

    if 'discovered[label] = build_model_from_state_dict(ab_results[key]["state_dict"]).to(device)' in line:
        new_lines.append('                discovered[label] = build_model_from_state_dict(\n')
        new_lines.append('                    ab_results[key]["state_dict"],\n')
        new_lines.append('                    filters=ab_results[key].get("filters"),\n')
        new_lines.append('                    kernel_size=ab_results[key].get("kernel_size", 5)\n')
        new_lines.append('                ).to(device)\n')
        continue

    new_lines.append(line)

with open('src/test_models.py', 'w') as f:
    f.writelines(new_lines)
