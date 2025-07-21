#!/usr/bin/env python3
"""
Analyze TFLite model for Android-specific issues
"""

import tensorflow as tf
import numpy as np
import os
import json

def analyze_tflite_model(model_path):
    """Comprehensive analysis of TFLite model for Android compatibility"""
    
    print(f"\n{'='*60}")
    print("TFLite Model Analysis for Android")
    print(f"{'='*60}\n")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("1. MODEL STRUCTURE")
    print("-" * 40)
    print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    print(f"Number of operations: {len(interpreter.get_tensor_details())}")
    
    print("\n2. INPUT TENSOR ANALYSIS")
    print("-" * 40)
    for i, input_detail in enumerate(input_details):
        print(f"Input {i}:")
        print(f"  Name: {input_detail['name']}")
        print(f"  Shape: {input_detail['shape']}")
        print(f"  Type: {input_detail['dtype']}")
        print(f"  Quantization: {input_detail['quantization']}")
        print(f"  Index: {input_detail['index']}")
    
    print("\n3. OUTPUT TENSOR ANALYSIS")
    print("-" * 40)
    for i, output_detail in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  Name: {output_detail['name']}")
        print(f"  Shape: {output_detail['shape']}")
        print(f"  Type: {output_detail['dtype']}")
        print(f"  Quantization: {output_detail['quantization']}")
        print(f"  Index: {output_detail['index']}")
    
    print("\n4. ANDROID-SPECIFIC CHECKS")
    print("-" * 40)
    
    # Check data types
    input_dtype = input_details[0]['dtype']
    output_dtype = output_details[0]['dtype']
    
    if input_dtype == np.float32 and output_dtype == np.float32:
        print("✓ Data types: Float32 (optimal for Android)")
    else:
        print(f"⚠ Data types: Input={input_dtype}, Output={output_dtype}")
        print("  Consider using Float32 for better Android compatibility")
    
    # Check tensor shapes
    output_shape = output_details[0]['shape']
    if len(output_shape) == 4 and output_shape[-1] == 3:
        print("✓ Output format: NHWC (channels-last, Android-friendly)")
    else:
        print(f"⚠ Output shape: {output_shape}")
        print("  Android prefers NHWC (batch, height, width, channels) format")
    
    print("\n5. INFERENCE TEST")
    print("-" * 40)
    
    try:
        # Create test input
        input_shape = input_details[0]['shape']
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✓ Inference successful!")
        print(f"  Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
        print(f"  Output range: [{output_data.min():.3f}, {output_data.max():.3f}]")
        print(f"  Output mean: {output_data.mean():.3f}")
        print(f"  Output std: {output_data.std():.3f}")
        
        # Check for common issues
        if np.all(output_data == output_data.flat[0]):
            print("\n⚠ WARNING: All output values are identical!")
            print("  This suggests the model may not be working correctly")
        
        if output_data.min() >= -1.1 and output_data.max() <= 1.1:
            print("\n✓ Output appears to use tanh activation (range ~[-1, 1])")
        elif output_data.min() >= -0.1 and output_data.max() <= 1.1:
            print("\n✓ Output appears to use sigmoid activation (range ~[0, 1])")
        else:
            print(f"\n⚠ Unusual output range: [{output_data.min():.3f}, {output_data.max():.3f}]")
        
        # Test multiple inferences
        print("\n6. CONSISTENCY TEST")
        print("-" * 40)
        outputs = []
        for i in range(5):
            test_input = np.random.randn(*input_shape).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            outputs.append(output)
            print(f"  Run {i+1}: range [{output.min():.3f}, {output.max():.3f}], mean {output.mean():.3f}")
        
        # Check if outputs vary
        all_same = True
        for i in range(1, len(outputs)):
            if not np.allclose(outputs[0], outputs[i], rtol=0.1):
                all_same = False
                break
        
        if all_same:
            print("\n⚠ WARNING: All outputs are similar despite different inputs!")
        else:
            print("\n✓ Model produces varied outputs for different inputs")
        
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
    
    print("\n7. ANDROID OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    
    # Model size recommendations
    model_size_mb = os.path.getsize(model_path) / 1024 / 1024
    if model_size_mb > 50:
        print(f"⚠ Model size ({model_size_mb:.1f} MB) may be large for some Android devices")
        print("  Consider quantization or pruning to reduce size")
    else:
        print(f"✓ Model size ({model_size_mb:.1f} MB) is reasonable for Android")
    
    # Performance recommendations
    print("\nFor optimal Android performance:")
    print("- Use CPU backend for compatibility (GPU delegate can be problematic)")
    print("- Use 4-8 threads based on device capabilities")
    print("- Consider INT8 quantization for 4x size reduction")
    print("- Test on multiple Android versions (especially 8.0+)")
    
    print("\n8. KNOWN ANDROID ISSUES AND SOLUTIONS")
    print("-" * 40)
    print("Common issues on Android:")
    print("1. Gray/uniform output → Check buffer alignment and data format")
    print("2. Crashes → Disable GPU delegate, use CPU only")
    print("3. Slow performance → Reduce threads or use quantization")
    print("4. Memory issues → Use smaller batch size (1) and optimize buffer usage")
    
    # Generate Android-specific code snippet
    print("\n9. RECOMMENDED ANDROID CODE")
    print("-" * 40)
    print("""
// Recommended Android/Flutter code for this model:

// 1. Load with CPU-only options
final options = InterpreterOptions()
  ..threads = 4; // Adjust based on device
  
final interpreter = await Interpreter.fromAsset(
  'your_model.tflite',
  options: options,
);

// 2. Use explicit buffer management
final input = Float32List(512); // Your input size
final outputSize = 256 * 256 * 3; // Your output size
final output = Float32List(outputSize);

// 3. Run inference with proper buffers
final inputs = <int, Object>{0: input.buffer.asFloat32List()};
final outputs = <int, Object>{0: output.buffer.asFloat32List()};
interpreter.runForMultipleInputs(inputs, outputs);

// 4. Convert output with auto-scaling
double min = output.reduce(math.min);
double max = output.reduce(math.max);
// Normalize based on actual range
""")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default path
        model_path = "E:/GitHub/BabyGAN/BabyGAN_TFLite_Conversion/04_Final_Models/stylegan_mobile_working.tflite"
    
    analyze_tflite_model(model_path)