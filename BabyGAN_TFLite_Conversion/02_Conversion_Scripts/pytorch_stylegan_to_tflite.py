#!/usr/bin/env python3
"""
PyTorch StyleGAN2 to TFLite Converter
Most reliable method for getting StyleGAN2 on mobile
"""

import os
import sys
import subprocess
import urllib.request

def setup_pytorch_environment():
    """Set up PyTorch and required dependencies"""
    
    print("🔧 Setting up PyTorch environment...")
    
    # Check if PyTorch is installed
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} already installed")
    except ImportError:
        print("📦 Installing PyTorch...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu'])
    
    # Clone StyleGAN2-PyTorch repo if not exists
    if not os.path.exists('stylegan2-pytorch'):
        print("\n📥 Cloning StyleGAN2-PyTorch repository...")
        subprocess.run(['git', 'clone', 'https://github.com/rosinality/stylegan2-pytorch.git'])
    
    return True


def download_pretrained_weights():
    """Download pre-trained StyleGAN2 weights"""
    
    print("\n📥 Downloading pre-trained weights...")
    
    weights_dir = 'stylegan2-pytorch/weights'
    os.makedirs(weights_dir, exist_ok=True)
    
    # Weight files to download
    weight_files = {
        '256px': {
            'url': 'https://github.com/rosinality/stylegan2-pytorch/releases/download/weights/stylegan2-ffhq-256px.pt',
            'path': os.path.join(weights_dir, 'stylegan2-ffhq-256px.pt'),
            'size_mb': 160
        },
        '512px': {
            'url': 'https://github.com/rosinality/stylegan2-pytorch/releases/download/weights/stylegan2-ffhq-512px.pt',
            'path': os.path.join(weights_dir, 'stylegan2-ffhq-512px.pt'),
            'size_mb': 370
        }
    }
    
    # Download 256px model (best for mobile)
    model_key = '256px'
    weight_info = weight_files[model_key]
    
    if os.path.exists(weight_info['path']):
        print(f"✅ Weights already exist: {weight_info['path']}")
        return weight_info['path']
    
    print(f"Downloading {model_key} model (~{weight_info['size_mb']}MB)...")
    print(f"URL: {weight_info['url']}")
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(weight_info['url'], weight_info['path'], download_progress)
        print(f"\n✅ Downloaded: {weight_info['path']}")
        return weight_info['path']
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return None


def convert_pytorch_to_onnx(weight_path):
    """Convert PyTorch model to ONNX"""
    
    print("\n🔄 Converting PyTorch to ONNX...")
    
    # Create conversion script
    conversion_script = '''
import torch
import sys
sys.path.append('stylegan2-pytorch')
from model import Generator

# Load model
print("Loading StyleGAN2 model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
g = Generator(256, 512, 8)  # 256px model

# Load weights
checkpoint = torch.load('{}', map_location=device)
g.load_state_dict(checkpoint['g_ema'], strict=False)
g.eval()

# Move to CPU for ONNX export
g = g.to('cpu')

# Create dummy input
dummy_input = torch.randn(1, 512)

# Export to ONNX
print("Exporting to ONNX...")
torch.onnx.export(
    g,
    dummy_input,
    "stylegan2_256px.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['latent'],
    output_names=['image'],
    dynamic_axes={
        'latent': {0: 'batch_size'},
        'image': {0: 'batch_size'}
    }
)

print("✅ ONNX export complete: stylegan2_256px.onnx")
'''.format(weight_path)
    
    # Save and run conversion script
    with open('convert_to_onnx.py', 'w') as f:
        f.write(conversion_script)
    
    try:
        subprocess.run([sys.executable, 'convert_to_onnx.py'], check=True)
        return 'stylegan2_256px.onnx'
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        return None


def convert_onnx_to_tflite(onnx_path):
    """Convert ONNX model to TFLite"""
    
    print("\n📱 Converting ONNX to TFLite...")
    
    # Install onnx2tf if not available
    try:
        import onnx2tf
    except ImportError:
        print("Installing onnx2tf...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnx2tf'])
    
    # Run conversion
    cmd = [
        sys.executable, '-m', 'onnx2tf',
        '-i', onnx_path,
        '-o', 'tflite_output',
        '-oiqt'  # Quantize to INT8
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Find the generated TFLite file
        for file in os.listdir('tflite_output'):
            if file.endswith('.tflite'):
                tflite_path = os.path.join('tflite_output', file)
                print(f"✅ TFLite model created: {tflite_path}")
                
                # Get file size
                size_mb = os.path.getsize(tflite_path) / 1024 / 1024
                print(f"📏 Model size: {size_mb:.2f} MB")
                
                return tflite_path
        
        print("❌ No TFLite file found in output")
        return None
        
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        return None


def create_mobile_demo():
    """Create a demo app for mobile"""
    
    print("\n📱 Creating mobile demo code...")
    
    demo_code = '''# Android Demo (Kotlin)
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

class StyleGAN2Mobile(private val modelPath: String) {
    private lateinit var interpreter: Interpreter
    
    fun initialize() {
        val model = loadModelFile(modelPath)
        interpreter = Interpreter(model)
    }
    
    fun generateRandomFace(): FloatArray {
        // Generate random latent vector
        val latent = FloatArray(512) { Random.nextGaussian().toFloat() }
        
        // Prepare buffers
        val inputBuffer = ByteBuffer.allocateDirect(512 * 4)
            .order(ByteOrder.nativeOrder())
        inputBuffer.asFloatBuffer().put(latent)
        
        val outputBuffer = ByteBuffer.allocateDirect(256 * 256 * 3 * 4)
            .order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        // Convert to array
        val output = FloatArray(256 * 256 * 3)
        outputBuffer.asFloatBuffer().get(output)
        
        return output
    }
    
    fun interpolateFaces(latent1: FloatArray, latent2: FloatArray, alpha: Float): FloatArray {
        // Linear interpolation between two latent vectors
        val interpolated = FloatArray(512) { i ->
            latent1[i] * (1 - alpha) + latent2[i] * alpha
        }
        
        // Generate face from interpolated latent
        return generateFromLatent(interpolated)
    }
}
'''
    
    with open('mobile_demo.kt', 'w') as f:
        f.write(demo_code)
    
    print("✅ Created mobile_demo.kt")


def main():
    """Main conversion pipeline"""
    
    print("🚀 PyTorch StyleGAN2 to TFLite Converter")
    print("=" * 50)
    
    # Step 1: Setup environment
    if not setup_pytorch_environment():
        print("❌ Failed to setup environment")
        return
    
    # Step 2: Download weights
    weight_path = download_pretrained_weights()
    if not weight_path:
        print("❌ Failed to download weights")
        return
    
    # Step 3: Convert to ONNX
    onnx_path = convert_pytorch_to_onnx(weight_path)
    if not onnx_path:
        print("❌ Failed to convert to ONNX")
        return
    
    # Step 4: Convert to TFLite
    tflite_path = convert_onnx_to_tflite(onnx_path)
    if not tflite_path:
        print("❌ Failed to convert to TFLite")
        return
    
    # Step 5: Create demo
    create_mobile_demo()
    
    print("\n🎉 Conversion complete!")
    print(f"\n📋 Summary:")
    print(f"  - PyTorch weights: {weight_path}")
    print(f"  - ONNX model: {onnx_path}")
    print(f"  - TFLite model: {tflite_path}")
    print(f"  - Mobile demo: mobile_demo.kt")
    
    print("\n📱 Next steps:")
    print("  1. Copy the .tflite file to your mobile app")
    print("  2. Use the demo code as a starting point")
    print("  3. Test on real devices")
    print("  4. Optimize further if needed")
    
    # Create summary file
    with open('CONVERSION_SUCCESS.txt', 'w') as f:
        f.write(f"StyleGAN2 TFLite Conversion Summary\n")
        f.write(f"==================================\n")
        f.write(f"TFLite Model: {tflite_path}\n")
        f.write(f"Model Size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB\n")
        f.write(f"Resolution: 256x256\n")
        f.write(f"Input Shape: [1, 512]\n")
        f.write(f"Output Shape: [1, 256, 256, 3]\n")


if __name__ == '__main__':
    main()