"""
Analyze TFLite model for Android compatibility issues
"""

import tensorflow as tf
import numpy as np
import os

def analyze_tflite_model(model_path):
    """Analyze TFLite model for potential Android issues"""
    
    print("=" * 60)
    print(f"Analyzing TFLite Model for Android Compatibility")
    print(f"Model: {model_path}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n📊 Model Information:")
    print(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    print("\n📥 Input Details:")
    for i, input_detail in enumerate(input_details):
        print(f"Input {i}:")
        print(f"  - Shape: {input_detail['shape']}")
        print(f"  - Type: {input_detail['dtype']}")
        print(f"  - Quantization: {input_detail['quantization']}")
    
    print("\n📤 Output Details:")
    for i, output_detail in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  - Shape: {output_detail['shape']}")
        print(f"  - Type: {output_detail['dtype']}")
        print(f"  - Quantization: {output_detail['quantization']}")
    
    # Test inference with different input patterns
    print("\n🧪 Testing Inference Patterns:")
    
    input_shape = input_details[0]['shape']
    
    # Test 1: Zero input
    print("\n1. Zero Input Test:")
    zero_input = np.zeros(input_shape, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], zero_input)
    interpreter.invoke()
    zero_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"   - Output range: [{np.min(zero_output):.4f}, {np.max(zero_output):.4f}]")
    print(f"   - Output mean: {np.mean(zero_output):.4f}")
    print(f"   - Unique values: {len(np.unique(zero_output))}")
    
    # Test 2: Random normal input
    print("\n2. Random Normal Input Test:")
    random_input = np.random.randn(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], random_input)
    interpreter.invoke()
    random_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"   - Output range: [{np.min(random_output):.4f}, {np.max(random_output):.4f}]")
    print(f"   - Output mean: {np.mean(random_output):.4f}")
    print(f"   - Unique values: {len(np.unique(random_output))}")
    print(f"   - Negative values: {np.sum(random_output < 0)} ({np.sum(random_output < 0) / random_output.size * 100:.1f}%)")
    
    # Test 3: Truncated normal input
    print("\n3. Truncated Normal Input Test (±2σ):")
    truncated_input = np.clip(np.random.randn(*input_shape), -2, 2).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], truncated_input)
    interpreter.invoke()
    truncated_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"   - Output range: [{np.min(truncated_output):.4f}, {np.max(truncated_output):.4f}]")
    print(f"   - Output mean: {np.mean(truncated_output):.4f}")
    print(f"   - Variance: {np.var(truncated_output):.6f}")
    
    # Analyze output characteristics
    print("\n🔍 Output Analysis:")
    
    # Determine likely activation function
    if np.min(random_output) < -0.5 and np.max(random_output) > 0.5:
        print("   ✓ Likely uses tanh activation (output in [-1, 1] range)")
        print("   → Use conversion: (value + 1) * 127.5")
    elif np.min(random_output) >= -0.1 and np.max(random_output) <= 1.1:
        print("   ✓ Likely uses sigmoid activation (output in [0, 1] range)")
        print("   → Use conversion: value * 255")
    else:
        print("   ⚠ Unusual output range detected")
        print("   → Use auto-scaling normalization")
    
    # Check for Android-specific issues
    print("\n⚠️  Android Compatibility Warnings:")
    
    warnings = []
    
    # Check model size
    if os.path.getsize(model_path) > 50 * 1024 * 1024:
        warnings.append("Model size >50MB may cause memory issues on some devices")
    
    # Check output variance
    if np.var(random_output) < 0.0001:
        warnings.append("Low output variance detected - model may not be responding to input")
    
    # Check for numerical issues
    if np.any(np.isnan(random_output)) or np.any(np.isinf(random_output)):
        warnings.append("NaN or Inf values detected in output")
    
    # Check quantization
    if output_details[0]['quantization'] != (0.0, 0):
        warnings.append("Model uses quantization - may behave differently on Android")
    
    if warnings:
        for warning in warnings:
            print(f"   ⚠ {warning}")
    else:
        print("   ✓ No major compatibility issues detected")
    
    # Recommendations
    print("\n💡 Recommendations for Android:")
    print("1. Use CPU-only execution (disable GPU delegate)")
    print("2. Use explicit Float32List buffers")
    print("3. Apply output normalization based on detected range")
    print("4. Use truncated normal distribution for input (±2σ)")
    print("5. Test with different thread counts (1-4)")

def compare_with_keras_model():
    """Compare TFLite output with original Keras model if available"""
    print("\n" + "=" * 60)
    print("Comparing TFLite with Original Model")
    print("=" * 60)
    
    # This would compare outputs if we had the original model
    print("Note: Original model comparison not available")
    print("If you have the original .h5 or .pb model, you can add comparison here")

if __name__ == "__main__":
    # Analyze the TFLite model
    tflite_path = "babygan_tflite_app/assets/models/stylegan_mobile_working.tflite"
    
    if os.path.exists(tflite_path):
        analyze_tflite_model(tflite_path)
        compare_with_keras_model()
    else:
        print(f"❌ TFLite model not found at: {tflite_path}")
        print("Please ensure the model is in the correct location")
        
        # Try alternative path
        alt_path = "07_Flutter_Test/babygan_flutter_test/assets/models/stylegan_mobile_working.tflite"
        if os.path.exists(alt_path):
            print(f"\n✓ Found model at alternative location: {alt_path}")
            analyze_tflite_model(alt_path)