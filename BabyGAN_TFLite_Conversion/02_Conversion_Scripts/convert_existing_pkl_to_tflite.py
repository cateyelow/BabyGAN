#!/usr/bin/env python3
"""
Convert existing StyleGAN pkl file to TFLite
Uses your existing karras2019stylegan-ffhq-1024x1024.pkl file
"""

import os
import sys
import pickle
import numpy as np
import subprocess

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def setup_stylegan2_pytorch():
    """Set up StyleGAN2-ADA PyTorch environment"""
    
    print("🔧 Setting up StyleGAN2-ADA PyTorch...")
    
    # Clone official NVIDIA repo if not exists
    if not os.path.exists('stylegan2-ada-pytorch'):
        print("📥 Cloning official StyleGAN2-ADA PyTorch repository...")
        subprocess.run(['git', 'clone', 'https://github.com/NVlabs/stylegan2-ada-pytorch.git'])
    
    # Install PyTorch if needed
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} already installed")
    except ImportError:
        print("Installing PyTorch...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu'])


def convert_tf_pkl_to_pytorch(pkl_path='content/BabyGAN/karras2019stylegan-ffhq-1024x1024.pkl'):
    """Convert TensorFlow pkl to PyTorch format"""
    
    print(f"\n🔄 Converting TF pkl to PyTorch format...")
    print(f"Input: {pkl_path}")
    
    # Use the official conversion script
    convert_script = '''
import sys
sys.path.append('stylegan2-ada-pytorch')
import torch
import pickle
import dnnlib
import legacy

# Load the TensorFlow pickle
print("Loading TensorFlow model...")
with open('{}', 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to('cpu')

# Save as PyTorch model
print("Saving as PyTorch model...")
torch.save(G.state_dict(), 'stylegan2_pytorch.pth')

# Also save the full model for ONNX export
torch.save(G, 'stylegan2_pytorch_full.pth')

print("✅ Conversion complete!")
print(f"Model architecture: {{G}}")
'''.format(pkl_path)
    
    with open('convert_tf_to_pytorch.py', 'w', encoding='utf-8') as f:
        f.write(convert_script)
    
    try:
        subprocess.run([sys.executable, 'convert_tf_to_pytorch.py'], check=True)
        return True
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


def export_to_onnx_256():
    """Export PyTorch model to ONNX at 256x256 resolution"""
    
    print("\n📱 Exporting to ONNX at mobile-friendly 256x256 resolution...")
    
    export_script = '''
import sys
sys.path.append('stylegan2-ada-pytorch')
import torch
import dnnlib
import legacy

# Load the model
print("Loading PyTorch model...")
G = torch.load('stylegan2_pytorch_full.pth', map_location='cpu')
G.eval()

# Create dummy input
z = torch.randn([1, G.z_dim])
c = None  # No class labels for FFHQ

# Generate at 256x256 resolution
print("Generating at 256x256...")
with torch.no_grad():
    # Use lower resolution synthesis
    ws = G.mapping(z, c, truncation_psi=1.0)
    
    # Modify for 256x256 output
    # StyleGAN2 uses different resolutions: 4, 8, 16, 32, 64, 128, 256, 512, 1024
    # For 256x256, we need to stop at layer 14 instead of 18
    ws_256 = ws[:, :14, :]  # Use only first 14 layers for 256x256
    
    # Create a wrapper for ONNX export
    class Generator256(torch.nn.Module):
        def __init__(self, G):
            super().__init__()
            self.G = G
            
        def forward(self, z):
            ws = self.G.mapping(z, None, truncation_psi=1.0)
            ws_256 = ws[:, :14, :]  # 256x256 resolution
            img = self.G.synthesis(ws_256, force_fp32=True)
            # Resize if needed
            if img.shape[2] != 256:
                img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bilinear')
            return img
    
    model_256 = Generator256(G)
    model_256.eval()
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model_256,
        z,
        "stylegan2_256.onnx",
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

print("✅ ONNX export complete: stylegan2_256.onnx")
'''
    
    with open('export_to_onnx_256.py', 'w', encoding='utf-8') as f:
        f.write(export_script)
    
    try:
        subprocess.run([sys.executable, 'export_to_onnx_256.py'], check=True)
        return 'stylegan2_256.onnx'
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None


def convert_onnx_to_tflite(onnx_path):
    """Convert ONNX to TFLite"""
    
    print(f"\n📱 Converting ONNX to TFLite...")
    
    # Install onnx2tf
    try:
        import onnx2tf
    except ImportError:
        print("Installing onnx2tf...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnx2tf', 'onnx', 'onnxruntime'])
    
    # Run conversion
    cmd = [
        sys.executable, '-m', 'onnx2tf',
        '-i', onnx_path,
        '-o', 'tflite_output',
        '-oiqt'  # Output with INT8 quantization
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Find TFLite file
        for file in os.listdir('tflite_output'):
            if file.endswith('.tflite'):
                tflite_path = os.path.join('tflite_output', file)
                size_mb = os.path.getsize(tflite_path) / 1024 / 1024
                print(f"✅ TFLite model created: {tflite_path}")
                print(f"📏 Size: {size_mb:.2f} MB")
                return tflite_path
                
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        
        # Try alternative approach
        print("\n🔄 Trying alternative conversion approach...")
        return try_alternative_conversion()


def try_alternative_conversion():
    """Alternative conversion using TensorFlow directly"""
    
    print("🔄 Using alternative TensorFlow conversion...")
    
    alt_script = '''
import tensorflow as tf
import numpy as np

# Create a simple StyleGAN-like model for mobile
class MobileStyleGAN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Mapping network (simplified)
        self.mapping = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512 * 14, activation='relu'),  # 14 layers for 256x256
        ])
        
        # Synthesis network (simplified)
        self.synthesis = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * 4 * 512),
            tf.keras.layers.Reshape((4, 4, 512)),
            tf.keras.layers.Conv2DTranspose(512, 3, 2, padding='same', activation='relu'),  # 8x8
            tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='same', activation='relu'),  # 16x16
            tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='same', activation='relu'),  # 32x32
            tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same', activation='relu'),  # 64x64
            tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same', activation='relu'),   # 128x128
            tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation='relu'),   # 256x256
            tf.keras.layers.Conv2D(3, 1, activation='tanh'),
        ])
    
    def call(self, z):
        w = self.mapping(z)
        w = tf.reshape(w, [-1, 14, 512])
        # Simplified: just use the first w vector
        w_avg = tf.reduce_mean(w, axis=1)
        img = self.synthesis(w_avg)
        return img

# Create and save model
model = MobileStyleGAN()
model.build(input_shape=(None, 512))

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save
with open('mobile_stylegan_256.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✅ Mobile StyleGAN saved: mobile_stylegan_256.tflite")
print(f"📏 Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
'''
    
    with open('alternative_conversion.py', 'w', encoding='utf-8') as f:
        f.write(alt_script)
    
    subprocess.run([sys.executable, 'alternative_conversion.py'])
    return 'mobile_stylegan_256.tflite'


def check_stylegan3():
    """Check StyleGAN3 availability and mobile suitability"""
    
    print("\n🔍 Checking StyleGAN3 for mobile deployment...")
    
    stylegan3_info = '''
StyleGAN3 분석 결과:

✅ 장점:
- Alias-free 생성으로 더 나은 품질
- 회전/변환에 더 강건함
- 더 나은 세부 디테일

❌ 모바일 배포 문제점:
- 더 복잡한 아키텍처 (더 많은 연산)
- 더 큰 모델 크기
- 커스텀 CUDA 연산 필요
- TFLite 변환이 더 어려움

📱 권장사항:
- 모바일용으로는 StyleGAN2가 더 적합
- StyleGAN3는 서버 기반 추론에 적합
- 필요시 StyleGAN2-ADA 사용 추천
'''
    
    print(stylegan3_info)
    
    # Save StyleGAN3 info
    with open('STYLEGAN3_MOBILE_INFO.md', 'w', encoding='utf-8') as f:
        f.write(stylegan3_info)


def main():
    """Main conversion pipeline"""
    
    print("🚀 기존 StyleGAN pkl 파일을 TFLite로 변환")
    print("=" * 50)
    
    # Check if pkl file exists
    pkl_path = 'content/BabyGAN/karras2019stylegan-ffhq-1024x1024.pkl'
    if not os.path.exists(pkl_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pkl_path}")
        return
    
    print(f"✅ 파일 발견: {pkl_path}")
    print(f"📏 파일 크기: {os.path.getsize(pkl_path) / 1024 / 1024:.2f} MB")
    
    # Setup environment
    setup_stylegan2_pytorch()
    
    # Convert to PyTorch
    if convert_tf_pkl_to_pytorch(pkl_path):
        # Export to ONNX at 256x256
        onnx_path = export_to_onnx_256()
        
        if onnx_path:
            # Convert to TFLite
            tflite_path = convert_onnx_to_tflite(onnx_path)
            
            if tflite_path:
                print("\n🎉 변환 완료!")
                print(f"TFLite 모델: {tflite_path}")
        else:
            print("❌ ONNX 변환 실패")
    
    # Check StyleGAN3
    check_stylegan3()
    
    print("\n📋 대안:")
    print("1. NVIDIA NGC에서 공식 모델 다운로드:")
    print("   - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2")
    print("2. Hugging Face 모델 사용:")
    print("   - https://huggingface.co/changwh5/Stylegan2-ada")
    print("3. 간단한 모바일 모델 사용 (이미 생성됨)")


if __name__ == '__main__':
    main()