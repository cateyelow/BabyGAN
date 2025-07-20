#!/usr/bin/env python3
"""
Fix for TensorFlow Hub download issues
Alternative methods to download and convert StyleGAN2 models
"""

import os
import sys
import tempfile
import shutil
import urllib.request
import tensorflow as tf
import numpy as np

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def clear_tfhub_cache():
    """Clear TensorFlow Hub cache to fix corrupted downloads"""
    
    print("🧹 Clearing TensorFlow Hub cache...")
    
    # Get cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "tfhub_modules")
    alt_cache_dir = os.path.join(tempfile.gettempdir(), "tfhub_modules")
    
    # Clear both possible cache locations
    for cache in [cache_dir, alt_cache_dir]:
        if os.path.exists(cache):
            print(f"Found cache at: {cache}")
            try:
                shutil.rmtree(cache)
                print(f"✅ Cleared cache: {cache}")
            except Exception as e:
                print(f"⚠️ Could not clear {cache}: {e}")


def test_tfhub_connectivity():
    """Test if we can reach TensorFlow Hub"""
    
    print("\n🌐 Testing TensorFlow Hub connectivity...")
    
    test_urls = [
        "https://tfhub.dev",
        "https://storage.googleapis.com/tfhub-modules/google/stylegan2-ffhq-256x256/1.tar.gz"
    ]
    
    for url in test_urls:
        try:
            response = urllib.request.urlopen(url, timeout=10)
            print(f"✅ Can reach: {url} (Status: {response.status})")
        except Exception as e:
            print(f"❌ Cannot reach: {url}")
            print(f"   Error: {e}")


def download_model_manually(model_name="stylegan2-ffhq-256x256"):
    """Manually download the model if TF Hub fails"""
    
    print(f"\n📥 Attempting manual download of {model_name}...")
    
    # Direct download URLs for StyleGAN2 models
    model_urls = {
        "stylegan2-ffhq-256x256": "https://storage.googleapis.com/tfhub-modules/google/stylegan2-ffhq-256x256/1.tar.gz",
        "stylegan2-ffhq-512x512": "https://storage.googleapis.com/tfhub-modules/google/stylegan2-ffhq-512x512/1.tar.gz",
        "stylegan2-ffhq-1024x1024": "https://storage.googleapis.com/tfhub-modules/google/stylegan2-ffhq-1024x1024/1.tar.gz"
    }
    
    if model_name not in model_urls:
        print(f"❌ Unknown model: {model_name}")
        return None
    
    url = model_urls[model_name]
    output_dir = f"downloaded_models/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    tarball_path = os.path.join(output_dir, "model.tar.gz")
    
    try:
        print(f"Downloading from: {url}")
        print("This may take a few minutes...")
        
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, tarball_path, download_progress)
        print("\n✅ Download complete!")
        
        # Extract the model
        print("📦 Extracting model...")
        import tarfile
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(output_dir)
        
        print(f"✅ Model extracted to: {output_dir}")
        
        # Clean up tarball
        os.remove(tarball_path)
        
        return output_dir
        
    except Exception as e:
        print(f"\n❌ Manual download failed: {e}")
        return None


def use_alternative_model():
    """Use an alternative pre-trained model that's easier to convert"""
    
    print("\n🔄 Using alternative approach: TensorFlow SavedModel...")
    
    # Create a simple StyleGAN-like model for demonstration
    print("Creating a simplified model for TFLite conversion...")
    
    class SimpleGenerator(tf.keras.Model):
        def __init__(self):
            super(SimpleGenerator, self).__init__()
            
            # Simple generator architecture
            self.dense1 = tf.keras.layers.Dense(4 * 4 * 512, use_bias=False)
            self.bn1 = tf.keras.layers.BatchNormalization()
            
            self.conv1 = tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding='same', use_bias=False)
            self.bn2 = tf.keras.layers.BatchNormalization()
            
            self.conv2 = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False)
            self.bn3 = tf.keras.layers.BatchNormalization()
            
            self.conv3 = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False)
            self.bn4 = tf.keras.layers.BatchNormalization()
            
            self.conv4 = tf.keras.layers.Conv2DTranspose(32, 5, strides=2, padding='same', use_bias=False)
            self.bn5 = tf.keras.layers.BatchNormalization()
            
            self.conv5 = tf.keras.layers.Conv2DTranspose(3, 5, strides=2, padding='same', use_bias=False, activation='tanh')
        
        def call(self, inputs, training=False):
            x = self.dense1(inputs)
            x = self.bn1(x, training=training)
            x = tf.nn.leaky_relu(x)
            
            x = tf.reshape(x, [-1, 4, 4, 512])
            
            x = self.conv1(x)
            x = self.bn2(x, training=training)
            x = tf.nn.leaky_relu(x)
            
            x = self.conv2(x)
            x = self.bn3(x, training=training)
            x = tf.nn.leaky_relu(x)
            
            x = self.conv3(x)
            x = self.bn4(x, training=training)
            x = tf.nn.leaky_relu(x)
            
            x = self.conv4(x)
            x = self.bn5(x, training=training)
            x = tf.nn.leaky_relu(x)
            
            x = self.conv5(x)
            
            # Resize to 256x256
            x = tf.image.resize(x, [256, 256])
            
            return x
    
    # Create and save model
    generator = SimpleGenerator()
    
    # Build the model
    generator(tf.random.normal([1, 512]))
    
    # Save as SavedModel
    tf.saved_model.save(generator, "simple_stylegan_savedmodel")
    
    print("✅ Created alternative model")
    
    # Convert to TFLite
    print("\n📱 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model("simple_stylegan_savedmodel")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = "simple_stylegan.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model saved: {tflite_path}")
    print(f"📏 Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    return tflite_path


def fix_and_retry_tfhub():
    """Try to fix TF Hub issues and retry the download"""
    
    print("\n🔧 Attempting to fix TensorFlow Hub issues...")
    
    # Set environment variables for TF Hub
    os.environ['TFHUB_CACHE_DIR'] = os.path.join(os.getcwd(), 'tfhub_cache')
    os.environ['TFHUB_DOWNLOAD_PROGRESS'] = '1'
    
    # Try with a simple model first
    print("\n🧪 Testing with a simple TF Hub model...")
    
    try:
        import tensorflow_hub as hub
        
        # Try a smaller, simpler model first
        test_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
        print(f"Testing with: {test_url}")
        
        model = hub.load(test_url)
        print("✅ TF Hub is working with simple models!")
        
        # Now try StyleGAN2 with KerasLayer
        print("\n📥 Trying alternative loading method for StyleGAN2...")
        
        # Create a wrapper model
        @tf.function
        def stylegan_wrapper(latents):
            hub_layer = hub.KerasLayer("https://tfhub.dev/google/stylegan2-ffhq-256x256/1", 
                                      trainable=False)
            return hub_layer(latents)
        
        # Get concrete function
        concrete_func = stylegan_wrapper.get_concrete_function(
            tf.TensorSpec(shape=[1, 512], dtype=tf.float32)
        )
        
        print("✅ Alternative loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Alternative method also failed: {e}")
        return False


def create_pytorch_alternative():
    """Provide PyTorch alternative instructions"""
    
    print("\n🐍 PyTorch Alternative (Recommended)")
    print("=" * 50)
    
    instructions = """
Since TensorFlow Hub is having issues, here's the PyTorch alternative:

1. Install PyTorch and download pre-trained model:
   ```bash
   pip install torch torchvision
   git clone https://github.com/rosinality/stylegan2-pytorch.git
   cd stylegan2-pytorch
   ```

2. Download pre-trained weights:
   ```bash
   # For 256x256 model (recommended for mobile)
   wget https://github.com/rosinality/stylegan2-pytorch/releases/download/weights/stylegan2-ffhq-256px.pt
   ```

3. Export to ONNX:
   ```python
   import torch
   from model import Generator
   
   # Load model
   g = Generator(256, 512, 8)
   g.load_state_dict(torch.load('stylegan2-ffhq-256px.pt')['g'])
   g.eval()
   
   # Export to ONNX
   dummy_input = torch.randn(1, 512)
   torch.onnx.export(g, dummy_input, "stylegan2.onnx", 
                     input_names=['latent'], 
                     output_names=['image'],
                     dynamic_axes={'latent': {0: 'batch'}, 'image': {0: 'batch'}})
   ```

4. Convert ONNX to TFLite:
   ```bash
   pip install onnx2tf
   onnx2tf -i stylegan2.onnx -o tflite_output -oiqt
   ```

This approach is more reliable and gives you more control!
"""
    
    print(instructions)
    
    # Save instructions
    with open("PYTORCH_ALTERNATIVE.md", 'w') as f:
        f.write(instructions)
    
    print("\n📄 Instructions saved to: PYTORCH_ALTERNATIVE.md")


def main():
    """Main troubleshooting function"""
    
    print("🔧 TensorFlow Hub Troubleshooting Tool")
    print("=" * 50)
    
    # Step 1: Clear cache
    clear_tfhub_cache()
    
    # Step 2: Test connectivity
    test_tfhub_connectivity()
    
    # Step 3: Try manual download
    model_path = download_model_manually()
    
    if not model_path:
        # Step 4: Try fixing TF Hub
        if not fix_and_retry_tfhub():
            # Step 5: Use alternative model
            print("\n⚠️ TF Hub is not working properly. Using alternatives...")
            
            # Create simple model for testing
            tflite_path = use_alternative_model()
            
            # Provide PyTorch instructions
            create_pytorch_alternative()
            
            print("\n📋 Summary:")
            print("1. ❌ TensorFlow Hub download failed")
            print("2. ✅ Created simple alternative model for testing")
            print("3. ✅ Provided PyTorch conversion instructions")
            print("\n🎯 Recommendation: Use the PyTorch approach for best results!")
    
    else:
        print(f"\n✅ Model downloaded successfully to: {model_path}")
        print("You can now use this model for conversion!")


if __name__ == '__main__':
    main()