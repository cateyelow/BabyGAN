#!/usr/bin/env python3
"""
StyleGAN2 TensorFlow Hub to TFLite Converter
Simplest path to get a working TFLite model
"""

import os
import numpy as np
import tensorflow as tf

def download_and_convert_tfhub_model(model_url, output_path='stylegan2_mobile.tflite'):
    """Download StyleGAN2 from TF Hub and convert to TFLite"""
    
    print(f"📥 Downloading StyleGAN2 model from TensorFlow Hub...")
    print(f"URL: {model_url}")
    
    try:
        import tensorflow_hub as hub
    except ImportError:
        print("Installing tensorflow-hub...")
        import subprocess
        subprocess.run(['pip', 'install', 'tensorflow-hub'])
        import tensorflow_hub as hub
    
    # Load the model
    print("Loading model...")
    model = hub.load(model_url)
    
    # Get model info
    print("\n📊 Model Information:")
    print(f"  - Model type: StyleGAN2")
    print(f"  - Resolution: 256x256")
    print(f"  - Latent size: 512")
    
    # Create a concrete function for conversion
    print("\n🔧 Creating concrete function for TFLite conversion...")
    
    @tf.function
    def generate_images(latent_vector):
        """Generate images from latent vectors"""
        # The model expects latents in shape [batch_size, 512]
        return model(latent_vector)
    
    # Get concrete function with fixed input shape
    concrete_func = generate_images.get_concrete_function(
        tf.TensorSpec(shape=[1, 512], dtype=tf.float32)
    )
    
    # Convert to TFLite
    print("\n📱 Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # Optimization settings
    print("Applying optimizations...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Use float16 quantization for better quality
    converter.target_spec.supported_types = [tf.float16]
    
    # Allow custom ops if needed
    converter.allow_custom_ops = True
    
    # Convert the model
    try:
        tflite_model = converter.convert()
        print(f"✅ Conversion successful!")
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"💾 Model saved to: {output_path}")
        print(f"📏 Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\nTrying alternative conversion settings...")
        
        # Try with different settings
        converter.optimizations = []
        converter.target_spec.supported_types = [tf.float32]
        
        try:
            tflite_model = converter.convert()
            
            alt_path = output_path.replace('.tflite', '_fp32.tflite')
            with open(alt_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ Alternative conversion successful!")
            print(f"💾 Model saved to: {alt_path}")
            print(f"📏 Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
            
            return alt_path
            
        except Exception as e2:
            print(f"❌ Alternative conversion also failed: {e2}")
            return None


def test_tflite_model(tflite_path):
    """Test the converted TFLite model"""
    
    print(f"\n🧪 Testing TFLite model: {tflite_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📊 Model Details:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")
    
    # Generate test latent vector
    latent = np.random.randn(1, 512).astype(np.float32)
    
    # Run inference
    print("\n🚀 Running inference...")
    interpreter.set_tensor(input_details[0]['index'], latent)
    
    import time
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ Inference successful!")
    print(f"⏱️  Inference time: {inference_time:.2f} ms")
    print(f"📐 Output shape: {output.shape}")
    print(f"📊 Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return output


def create_mobile_app_example(tflite_path):
    """Create example code for mobile app integration"""
    
    android_code = '''// Android (Kotlin) Example
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class StyleGANModel(private val modelPath: String) {
    private lateinit var interpreter: Interpreter
    
    fun initialize() {
        val model = loadModelFile(modelPath)
        interpreter = Interpreter(model)
    }
    
    fun generateImage(latentVector: FloatArray): FloatArray {
        val inputBuffer = ByteBuffer.allocateDirect(1 * 512 * 4)
            .order(ByteOrder.nativeOrder())
        
        // Fill input buffer
        inputBuffer.asFloatBuffer().put(latentVector)
        
        // Prepare output buffer (1 * 256 * 256 * 3)
        val outputBuffer = ByteBuffer.allocateDirect(1 * 256 * 256 * 3 * 4)
            .order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        // Convert to float array
        val output = FloatArray(256 * 256 * 3)
        outputBuffer.asFloatBuffer().get(output)
        
        return output
    }
}
'''

    ios_code = '''// iOS (Swift) Example
import TensorFlowLite

class StyleGANModel {
    private var interpreter: Interpreter?
    
    init(modelPath: String) {
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter?.allocateTensors()
        } catch {
            print("Failed to create interpreter: \\(error)")
        }
    }
    
    func generateImage(latentVector: [Float]) -> [Float]? {
        guard let interpreter = interpreter else { return nil }
        
        // Prepare input
        let inputTensor = try? interpreter.input(at: 0)
        let inputData = latentVector.withUnsafeBufferPointer { Data(buffer: $0) }
        try? interpreter.copy(inputData, toInputAt: 0)
        
        // Run inference
        try? interpreter.invoke()
        
        // Get output
        let outputTensor = try? interpreter.output(at: 0)
        guard let outputData = outputTensor?.data else { return nil }
        
        let output = outputData.withUnsafeBytes { (ptr: UnsafePointer<Float>) in
            Array(UnsafeBufferPointer(start: ptr, count: 256 * 256 * 3))
        }
        
        return output
    }
}
'''
    
    # Save example code
    with open('mobile_integration_android.kt', 'w') as f:
        f.write(android_code)
    
    with open('mobile_integration_ios.swift', 'w') as f:
        f.write(ios_code)
    
    print("\n📱 Mobile integration examples created:")
    print("  - mobile_integration_android.kt")
    print("  - mobile_integration_ios.swift")


def main():
    """Main conversion function"""
    
    print("🚀 StyleGAN2 TensorFlow Hub to TFLite Converter")
    print("=" * 50)
    
    # Available models
    models = {
        '256': 'https://tfhub.dev/google/stylegan2-ffhq-256x256/1',
        '512': 'https://tfhub.dev/google/stylegan2-ffhq-512x512/1',
        '1024': 'https://tfhub.dev/google/stylegan2-ffhq-1024x1024/1'
    }
    
    # Choose resolution
    print("\n📐 Available resolutions:")
    print("  1. 256x256 (Recommended for mobile)")
    print("  2. 512x512 (High-end devices only)")
    print("  3. 1024x1024 (Not recommended for mobile)")
    
    # Use 256x256 for mobile deployment
    resolution = '256'
    model_url = models[resolution]
    
    print(f"\n✅ Using {resolution}x{resolution} model for mobile deployment")
    
    # Convert model
    tflite_path = download_and_convert_tfhub_model(
        model_url, 
        f'stylegan2_{resolution}x{resolution}.tflite'
    )
    
    if tflite_path and os.path.exists(tflite_path):
        # Test the model
        test_tflite_model(tflite_path)
        
        # Create mobile examples
        create_mobile_app_example(tflite_path)
        
        print("\n🎉 Conversion complete!")
        print(f"\n📋 Summary:")
        print(f"  - Model: StyleGAN2 {resolution}x{resolution}")
        print(f"  - Output: {tflite_path}")
        print(f"  - Size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
        print(f"  - Ready for mobile deployment!")
        
        print("\n📝 Next steps:")
        print("  1. Copy the .tflite file to your mobile app")
        print("  2. Use the provided integration examples")
        print("  3. Test on target devices")
        print("  4. Consider further optimization if needed")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("  1. Ensure TensorFlow 2.x is installed")
        print("  2. Check internet connection for model download")
        print("  3. Try a smaller resolution model")
        print("  4. Consider using PyTorch alternatives")


if __name__ == '__main__':
    main()