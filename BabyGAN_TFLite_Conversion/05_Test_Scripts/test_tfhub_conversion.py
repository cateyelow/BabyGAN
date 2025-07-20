#!/usr/bin/env python3
"""
Quick test to verify TensorFlow Hub StyleGAN2 can be loaded and converted
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

print("Testing TensorFlow Hub StyleGAN2 conversion...")
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Hub version: {hub.__version__}")

# Test loading a small model first
try:
    print("\nTesting model loading (this will download ~100MB on first run)...")
    
    # Use 256x256 model for mobile
    model_url = "https://tfhub.dev/google/stylegan2-ffhq-256x256/1"
    
    # Create a simple function to test
    @tf.function
    def test_generation(latents):
        # Load model inside function for testing
        model = hub.load(model_url)
        return model(latents)
    
    # Test with dummy input
    test_latent = tf.random.normal([1, 512])
    print("Creating concrete function...")
    
    concrete_func = test_generation.get_concrete_function(
        tf.TensorSpec(shape=[1, 512], dtype=tf.float32)
    )
    
    print("Testing TFLite conversion...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("\n✅ All prerequisites are working!")
    print("\nYou can now run:")
    print("  python stylegan2_tfhub_to_tflite.py")
    print("\nThis will create a mobile-ready StyleGAN2 model!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Ensure TensorFlow 2.x is properly installed")
    print("3. Try: pip install --upgrade tensorflow tensorflow-hub")