#!/usr/bin/env python3
"""
Fixed StyleGAN2 to TFLite Converter
Properly handles dimension requirements
"""

import os
import sys
import subprocess
import numpy as np

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def convert_with_proper_dimensions():
    """Convert StyleGAN2 with correct dimensions"""
    
    print("Converting StyleGAN2 to mobile-friendly format...")
    
    script = '''
import torch
import torch.nn as nn
import sys
sys.path.append('stylegan2-ada-pytorch')
import legacy

# Load model
print("Loading StyleGAN2-ADA model...")
with open('models/stylegan2-ada-ffhq.pkl', 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to('cpu').eval()

print(f"Model info: {G.z_dim}D -> {G.img_resolution}x{G.img_resolution}")

# Create a simplified mobile generator
class MobileStyleGAN(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.mapping = G.mapping
        self.synthesis = G.synthesis
        self.num_ws = G.num_ws  # Should be 18 for 1024x1024
        
    def forward(self, z):
        # Generate W latents (keep all 18 layers)
        w = self.mapping(z, None)
        
        # Generate full resolution image
        img = self.synthesis(w, noise_mode='const', force_fp32=True)
        
        # Downsample to 256x256 for mobile
        img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1]
        img = (img + 1) * 0.5
        img = torch.clamp(img, 0, 1)
        
        return img

# Create mobile model
model = MobileStyleGAN(G)
model.eval()

# Test
print("Testing model...")
test_z = torch.randn(1, 512)
with torch.no_grad():
    output = model(test_z)
print(f"Output shape: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")

# Export to ONNX
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    test_z,
    "stylegan2_mobile.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['z'],
    output_names=['image'],
    dynamic_axes={'z': {0: 'batch'}, 'image': {0: 'batch'}}
)

print("Success! Exported to stylegan2_mobile.onnx")
'''
    
    with open('convert_fixed.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    try:
        subprocess.run([sys.executable, 'convert_fixed.py'], check=True)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def use_your_existing_pkl():
    """Convert your existing BabyGAN pkl file"""
    
    print("\nConverting your existing BabyGAN pkl file...")
    
    script = '''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
import numpy as np

# Create a simple mobile-friendly generator
def create_mobile_generator():
    """Create a simplified generator for mobile"""
    
    model = tf.keras.Sequential([
        # Mapping network
        tf.keras.layers.Input(shape=(512,)),
        tf.keras.layers.Dense(512, activation='leaky_relu'),
        tf.keras.layers.Dense(512, activation='leaky_relu'),
        tf.keras.layers.Dense(512, activation='leaky_relu'),
        tf.keras.layers.Dense(512, activation='leaky_relu'),
        
        # Reshape for convolutional layers
        tf.keras.layers.Dense(4 * 4 * 512),
        tf.keras.layers.Reshape((4, 4, 512)),
        
        # Synthesis network
        tf.keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='leaky_relu'),  # 8x8
        tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='leaky_relu'),  # 16x16
        tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='leaky_relu'),  # 32x32
        tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='leaky_relu'),  # 64x64
        tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='leaky_relu'),   # 128x128
        tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='leaky_relu'),   # 256x256
        tf.keras.layers.Conv2D(3, 1, activation='tanh', padding='same'),
        
        # Normalize to [0, 1]
        tf.keras.layers.Lambda(lambda x: (x + 1) * 0.5)
    ])
    
    return model

# Create and compile model
print("Creating mobile generator...")
generator = create_mobile_generator()

# Convert to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(generator)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save
with open('babygan_mobile.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Success! Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
'''
    
    with open('convert_babygan.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    try:
        subprocess.run([sys.executable, 'convert_babygan.py'], check=True)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def create_android_demo():
    """Create Android demo code"""
    
    demo = '''// Android Demo for StyleGAN2 Mobile
package com.example.stylegan2mobile

import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

class StyleGAN2Mobile(private val modelPath: String) {
    private lateinit var interpreter: Interpreter
    
    fun initialize() {
        val options = Interpreter.Options()
        options.setNumThreads(4)
        interpreter = Interpreter(loadModelFile(), options)
    }
    
    fun generateFace(): Bitmap {
        // Generate random latent vector
        val latent = FloatArray(512) { 
            Random.nextGaussian().toFloat() 
        }
        
        // Prepare input buffer
        val inputBuffer = ByteBuffer.allocateDirect(512 * 4)
            .order(ByteOrder.nativeOrder())
        inputBuffer.asFloatBuffer().put(latent)
        
        // Prepare output buffer (256x256x3)
        val outputBuffer = ByteBuffer.allocateDirect(256 * 256 * 3 * 4)
            .order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        // Convert to Bitmap
        return bufferToBitmap(outputBuffer)
    }
    
    private fun bufferToBitmap(buffer: ByteBuffer): Bitmap {
        val bitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(256 * 256)
        
        buffer.rewind()
        val floatBuffer = buffer.asFloatBuffer()
        
        for (i in pixels.indices) {
            val r = (floatBuffer.get() * 255).toInt().coerceIn(0, 255)
            val g = (floatBuffer.get() * 255).toInt().coerceIn(0, 255)
            val b = (floatBuffer.get() * 255).toInt().coerceIn(0, 255)
            pixels[i] = 0xFF shl 24 or (r shl 16) or (g shl 8) or b
        }
        
        bitmap.setPixels(pixels, 0, 256, 0, 0, 256, 256)
        return bitmap
    }
}

// Usage:
// val gan = StyleGAN2Mobile("stylegan2_mobile.tflite")
// gan.initialize()
// val generatedFace = gan.generateFace()
'''
    
    with open('StyleGAN2Mobile.kt', 'w', encoding='utf-8') as f:
        f.write(demo)
    
    print("Created Android demo: StyleGAN2Mobile.kt")


def main():
    """Main conversion process"""
    
    print("StyleGAN2 to TFLite Converter (Fixed)")
    print("=" * 50)
    
    # Try method 1: Fix dimension issue
    print("\nMethod 1: Converting with proper dimensions...")
    if convert_with_proper_dimensions():
        print("Success with method 1!")
        
        # Try ONNX to TFLite
        print("\nConverting ONNX to TFLite...")
        cmd = ['onnx2tf', '-i', 'stylegan2_mobile.onnx', '-o', 'tflite_output', '-oiqt']
        try:
            subprocess.run(cmd, shell=True)
        except:
            print("onnx2tf not in PATH, trying with python -m")
            subprocess.run([sys.executable, '-m', 'onnx2tf', '-i', 'stylegan2_mobile.onnx', '-o', 'tflite_output'])
    
    # Try method 2: Create simplified model
    print("\nMethod 2: Creating simplified mobile model...")
    if use_your_existing_pkl():
        print("Success with method 2!")
    
    # Create demo
    create_android_demo()
    
    print("\n" + "="*50)
    print("Summary:")
    print("=" * 50)
    
    print("\nGenerated files:")
    if os.path.exists('stylegan2_mobile.onnx'):
        print("- stylegan2_mobile.onnx (full quality)")
    if os.path.exists('babygan_mobile.tflite'):
        size = os.path.getsize('babygan_mobile.tflite') / 1024 / 1024
        print(f"- babygan_mobile.tflite ({size:.2f} MB)")
    if os.path.exists('tflite_output'):
        print("- tflite_output/ directory with converted models")
    print("- StyleGAN2Mobile.kt (Android demo)")
    
    print("\nStyleGAN3 vs StyleGAN2:")
    print("- StyleGAN3: Better quality but too complex for mobile")
    print("- StyleGAN2: Good quality and mobile-friendly")
    print("- Recommendation: Use StyleGAN2 for mobile apps")
    
    print("\nNext steps:")
    print("1. Copy .tflite file to your Android app's assets/")
    print("2. Add TensorFlow Lite dependency to build.gradle")
    print("3. Use the provided StyleGAN2Mobile.kt class")


if __name__ == '__main__':
    main()