#!/usr/bin/env python3
"""
Convert StyleGAN2 to TFLite - Working Solution
Using the successfully downloaded StyleGAN2-ADA model
"""

import os
import sys
import subprocess

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def convert_to_onnx():
    """Convert StyleGAN2-ADA to ONNX"""
    
    print("Converting StyleGAN2-ADA to ONNX...")
    
    # Create conversion script
    script = '''
import torch
import sys
sys.path.append('stylegan2-ada-pytorch')
import legacy

# Load the model
print("Loading StyleGAN2-ADA model...")
with open('models/stylegan2-ada-ffhq.pkl', 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to('cpu').eval()

print(f"Model loaded! Architecture: {G.z_dim}D latent -> {G.img_resolution}x{G.img_resolution} image")

# Create wrapper for 256x256 mobile output
class MobileGenerator(torch.nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G
        
    def forward(self, z):
        # Map to W space
        w = self.G.mapping(z, None)
        
        # For 256x256, we only need first 14 layers (instead of 18 for 1024x1024)
        # StyleGAN2 layer mapping: 4->0-1, 8->2-3, 16->4-5, 32->6-7, 64->8-9, 128->10-11, 256->12-13
        w_256 = w[:, :14, :]
        
        # Generate image at 256x256
        img = self.G.synthesis(w_256, noise_mode='const', force_fp32=True)
        
        # Ensure correct size
        if img.shape[2] != 256 or img.shape[3] != 256:
            img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Convert from [-1, 1] to [0, 1] for mobile
        img = (img + 1) * 0.5
        
        return img

# Create mobile model
mobile_model = MobileGenerator(G)
mobile_model.eval()

# Test the model
print("Testing model...")
test_z = torch.randn(1, 512)
with torch.no_grad():
    test_output = mobile_model(test_z)
print(f"Test output shape: {test_output.shape}, range: [{test_output.min():.3f}, {test_output.max():.3f}]")

# Export to ONNX
print("Exporting to ONNX...")
torch.onnx.export(
    mobile_model,
    test_z,
    "stylegan2_mobile_256.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['latent'],
    output_names=['image'],
    dynamic_axes={'latent': {0: 'batch'}, 'image': {0: 'batch'}}
)

print("Successfully exported to stylegan2_mobile_256.onnx!")
'''
    
    with open('convert_to_onnx.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    # Run conversion
    try:
        subprocess.run([sys.executable, 'convert_to_onnx.py'], check=True)
        return True
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        return False


def convert_onnx_to_tflite():
    """Convert ONNX to TFLite using onnx2tf"""
    
    print("\nConverting ONNX to TFLite...")
    
    # Check if onnx2tf is installed
    try:
        import onnx2tf
    except ImportError:
        print("Installing onnx2tf...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnx2tf', 'onnxsim', 'onnxruntime'])
    
    # First simplify the ONNX model
    print("Simplifying ONNX model...")
    subprocess.run([sys.executable, '-m', 'onnxsim', 'stylegan2_mobile_256.onnx', 'stylegan2_mobile_256_sim.onnx'])
    
    # Convert to TFLite
    cmd = [
        sys.executable, '-m', 'onnx2tf',
        '-i', 'stylegan2_mobile_256_sim.onnx',
        '-o', 'tflite_output',
        '-oiqt',  # INT8 quantization
        '-cotof',  # Custom ops to TFLite ops
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Find the TFLite file
        tflite_files = []
        if os.path.exists('tflite_output'):
            for file in os.listdir('tflite_output'):
                if file.endswith('.tflite'):
                    tflite_files.append(os.path.join('tflite_output', file))
        
        if tflite_files:
            for tflite_path in tflite_files:
                size_mb = os.path.getsize(tflite_path) / 1024 / 1024
                print(f"Generated: {tflite_path} ({size_mb:.2f} MB)")
            return True
        else:
            print("No TFLite files found!")
            return False
            
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        return False


def create_test_app():
    """Create a test application"""
    
    print("\nCreating test application...")
    
    test_code = '''
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite_output/stylegan2_mobile_256_full_integer_quant.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# Generate random face
latent = np.random.randn(1, 512).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], latent)
interpreter.invoke()

# Get output
image = interpreter.get_tensor(output_details[0]['index'])
print(f"Generated image shape: {image.shape}")
print(f"Value range: [{image.min():.3f}, {image.max():.3f}]")

# Save as image
from PIL import Image
img_array = (image[0] * 255).astype(np.uint8)
img = Image.fromarray(img_array)
img.save("generated_face.png")
print("Saved generated_face.png")
'''
    
    with open('test_tflite_model.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("Created test_tflite_model.py")


def main():
    """Main conversion pipeline"""
    
    print("StyleGAN2 to TFLite Converter")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('models/stylegan2-ada-ffhq.pkl'):
        print("Error: Please run download_working_stylegan2_models.py first!")
        return
    
    # Check if stylegan2-ada-pytorch exists
    if not os.path.exists('stylegan2-ada-pytorch'):
        print("stylegan2-ada-pytorch not found. Please ensure it was cloned.")
        return
    
    # Step 1: Convert to ONNX
    if convert_to_onnx():
        print("\nStep 1 Complete: ONNX conversion successful!")
        
        # Step 2: Convert to TFLite
        if convert_onnx_to_tflite():
            print("\nStep 2 Complete: TFLite conversion successful!")
            
            # Step 3: Create test app
            create_test_app()
            
            print("\n" + "="*50)
            print("CONVERSION SUCCESSFUL!")
            print("="*50)
            print("\nGenerated files:")
            print("- stylegan2_mobile_256.onnx")
            print("- tflite_output/*.tflite")
            print("- test_tflite_model.py")
            
            print("\nNext steps:")
            print("1. Test the model: python test_tflite_model.py")
            print("2. Copy .tflite file to your mobile app")
            print("3. Use the provided mobile integration examples")
            
        else:
            print("\nTFLite conversion failed. Trying alternative approach...")
            # Alternative approach would go here
    else:
        print("\nONNX conversion failed.")
    
    # StyleGAN3 info
    print("\n" + "="*50)
    print("About StyleGAN3:")
    print("- More complex architecture than StyleGAN2")
    print("- Requires custom CUDA kernels")
    print("- Not suitable for mobile deployment")
    print("- Recommend StyleGAN2 for mobile use")


if __name__ == '__main__':
    main()