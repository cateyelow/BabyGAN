#!/usr/bin/env python3
"""Simple StyleGAN2 to ONNX converter"""

import torch
import sys
sys.path.append('stylegan2-ada-pytorch')

# For NVIDIA official pkl files
import legacy

# Load model
with open('models/stylegan2-ada-ffhq.pkl', 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to('cpu').eval()

# Create wrapper for 256x256 output
class Generator256(torch.nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G
        
    def forward(self, z):
        # Generate latent codes
        w = self.G.mapping(z, None)
        # Use only layers needed for 256x256
        w_256 = w[:, :14, :]
        # Generate image
        img = self.G.synthesis(w_256, noise_mode='const', force_fp32=True)
        # Ensure 256x256 output
        if img.shape[2] != 256:
            img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bilinear')
        return img

# Wrap model
model = Generator256(G)

# Export to ONNX
dummy_input = torch.randn(1, 512)
torch.onnx.export(model, dummy_input, "stylegan2_256.onnx",
                  input_names=['z'], output_names=['image'],
                  opset_version=12, do_constant_folding=True)

print("Exported to stylegan2_256.onnx")
