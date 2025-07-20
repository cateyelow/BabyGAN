#!/usr/bin/env python3
"""
Download working StyleGAN2 models from official sources
"""

import os
import urllib.request
import sys

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def download_with_progress(url, output_path):
    """Download file with progress bar"""
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\nError: {e}")
        return False


def download_stylegan2_models():
    """Download official StyleGAN2 models"""
    
    print("Download Working StyleGAN2 Models")
    print("=" * 50)
    
    # Official NVIDIA models
    models = {
        "StyleGAN2 FFHQ 256x256": {
            "url": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl",
            "path": "models/stylegan2-ffhq-256x256.pkl",
            "size_mb": 357
        },
        "StyleGAN2 FFHQ 512x512": {
            "url": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl", 
            "path": "models/stylegan2-ffhq-512x512.pkl",
            "size_mb": 361
        },
        "StyleGAN2-ADA FFHQ": {
            "url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
            "path": "models/stylegan2-ada-ffhq.pkl",
            "size_mb": 324
        },
        "StyleGAN2-ADA MetFaces": {
            "url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl",
            "path": "models/stylegan2-ada-metfaces.pkl", 
            "size_mb": 324
        }
    }
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    print("\nAvailable models:")
    for i, (name, info) in enumerate(models.items(), 1):
        print(f"{i}. {name} (~{info['size_mb']}MB)")
    
    # Download StyleGAN2-ADA FFHQ (works with PyTorch)
    print(f"\nDownloading StyleGAN2-ADA FFHQ model...")
    model_info = models["StyleGAN2-ADA FFHQ"]
    
    if os.path.exists(model_info["path"]):
        print(f"Model already exists: {model_info['path']}")
    else:
        print(f"URL: {model_info['url']}")
        print(f"Downloading {model_info['size_mb']}MB...")
        
        if download_with_progress(model_info["url"], model_info["path"]):
            print(f"Successfully downloaded: {model_info['path']}")
            print(f"File size: {os.path.getsize(model_info['path']) / 1024 / 1024:.2f} MB")
        else:
            print("Download failed!")
    
    # Alternative download locations
    print("\n\nAlternative download sources:")
    print("1. Hugging Face Models:")
    print("   - https://huggingface.co/changwh5/Stylegan2-ada")
    print("   - https://huggingface.co/ZeqiangLai/StyleGAN2-pkl")
    
    print("\n2. Direct PyTorch models (smaller):")
    print("   - Use stylegan2-pytorch repository")
    print("   - Pre-converted PyTorch weights available")
    
    print("\n3. For your existing pkl file:")
    print("   - Use NVIDIA's official legacy.py loader")
    print("   - Convert with stylegan2-ada-pytorch tools")
    
    return model_info["path"] if os.path.exists(model_info["path"]) else None


def create_simple_converter():
    """Create a simple converter script"""
    
    converter = '''#!/usr/bin/env python3
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
'''
    
    with open('simple_converter.py', 'w', encoding='utf-8') as f:
        f.write(converter)
    
    print("\nCreated simple_converter.py")


def main():
    """Main function"""
    
    # Download working models
    model_path = download_stylegan2_models()
    
    if model_path:
        # Create converter
        create_simple_converter()
        
        print("\n\nNext steps:")
        print("1. Run the converter:")
        print("   python simple_converter.py")
        print("\n2. Convert ONNX to TFLite:")
        print("   onnx2tf -i stylegan2_256.onnx -o tflite_output")
        print("\n3. Or use your existing pkl file with the converter script")
    
    print("\n\nFor StyleGAN3:")
    print("- More complex architecture, harder to convert")
    print("- Requires custom ops not supported by TFLite") 
    print("- Recommend StyleGAN2 for mobile deployment")


if __name__ == '__main__':
    main()