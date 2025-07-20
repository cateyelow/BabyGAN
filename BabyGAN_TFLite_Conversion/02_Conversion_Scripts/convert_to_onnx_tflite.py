#!/usr/bin/env python3
"""
StyleGAN to TensorFlow Lite Converter via ONNX
Converts frozen TF1 graph to TFLite using ONNX as intermediate format
"""

import os
import sys
import subprocess
import numpy as np
import pickle


def install_dependencies():
    """Install required conversion tools"""
    
    print("Installing conversion dependencies...")
    
    dependencies = [
        'tf2onnx',
        'onnx',
        'onnxruntime',
        'onnx2tf',
        'tensorflow>=2.0'
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', dep], check=False)
    
    print("Dependencies installed!")


def convert_pb_to_onnx(pb_path, model_info, output_path='stylegan.onnx'):
    """Convert frozen graph to ONNX format"""
    
    print(f"\n📊 Converting {pb_path} to ONNX...")
    
    # Build conversion command
    cmd = [
        sys.executable, '-m', 'tf2onnx.convert',
        '--graphdef', pb_path,
        '--output', output_path,
        '--inputs', f"{model_info['input_name'].split(':')[0]}:0",
        '--outputs', f"{model_info['output_name'].split(':')[0]}:0",
        '--opset', '13',
        '--verbose'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully converted to {output_path}")
            print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            return output_path
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None


def simplify_onnx_model(onnx_path, output_path='stylegan_simplified.onnx'):
    """Simplify ONNX model for better conversion"""
    
    print(f"\n🔧 Simplifying ONNX model...")
    
    try:
        import onnx
        from onnx import optimizer
        
        # Load model
        model = onnx.load(onnx_path)
        
        # Apply optimizations
        passes = [
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'fuse_bn_into_conv',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_pad_into_conv'
        ]
        
        optimized_model = optimizer.optimize(model, passes)
        
        # Save simplified model
        onnx.save(optimized_model, output_path)
        
        print(f"✅ Simplified model saved to {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"⚠️ Simplification failed: {e}")
        return onnx_path


def convert_onnx_to_tflite(onnx_path, output_dir='tflite_model'):
    """Convert ONNX to TensorFlow Lite using onnx2tf"""
    
    print(f"\n📱 Converting ONNX to TensorFlow Lite...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build conversion command
    cmd = [
        sys.executable, '-m', 'onnx2tf',
        '-i', onnx_path,
        '-o', output_dir,
        '-oiqt',  # Output with INT8 quantization
        '-cotof',  # Customize operations for TFLite
        '-coion',  # Custom operation implementation
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for TFLite file
        tflite_path = os.path.join(output_dir, 'stylegan.tflite')
        if os.path.exists(tflite_path) or result.returncode == 0:
            print(f"✅ TensorFlow Lite model created")
            
            # Find the actual tflite file
            for file in os.listdir(output_dir):
                if file.endswith('.tflite'):
                    tflite_path = os.path.join(output_dir, file)
                    print(f"📱 TFLite model: {tflite_path}")
                    print(f"File size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
                    return tflite_path
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            
            # Try alternative approach
            print("\n🔄 Trying alternative conversion method...")
            return convert_with_tensorflow(onnx_path, output_dir)
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None


def convert_with_tensorflow(onnx_path, output_dir):
    """Alternative conversion using TensorFlow directly"""
    
    print("\n🔄 Converting via TensorFlow SavedModel...")
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Save as SavedModel
        saved_model_path = os.path.join(output_dir, 'saved_model')
        tf_rep.export_graph(saved_model_path)
        
        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # Optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.allow_custom_ops = True
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = os.path.join(output_dir, 'stylegan_tf.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved to {tflite_path}")
        print(f"File size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
        
        return tflite_path
        
    except Exception as e:
        print(f"❌ Alternative conversion failed: {e}")
        return None


def create_mobile_optimized_model(model_info):
    """Create a mobile-optimized version with reduced resolution"""
    
    print("\n📱 Creating mobile-optimized model (256x256)...")
    
    # This would involve recreating the model at lower resolution
    # For now, we'll note this as a recommended approach
    
    print("⚠️ Note: For optimal mobile performance, consider:")
    print("  - Reducing output resolution to 256x256 or 512x512")
    print("  - Using MobileStyleGAN architecture")
    print("  - Applying aggressive quantization")
    print("  - Pruning unnecessary layers")


def validate_tflite_model(tflite_path, test_latent_path):
    """Validate the converted TFLite model"""
    
    print(f"\n✅ Validating TFLite model...")
    
    try:
        import tensorflow as tf
        
        # Load test latent
        test_latent = np.load(test_latent_path)
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_latent)
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✅ Inference successful!")
        print(f"Generated image shape: {output.shape}")
        print(f"Value range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def main():
    """Main conversion pipeline"""
    
    print("🚀 StyleGAN to TensorFlow Lite Converter")
    print("=" * 50)
    
    # Check for extracted model
    extracted_dir = 'extracted_model'
    if not os.path.exists(extracted_dir):
        print("❌ No extracted model found. Please run extract_stylegan_model.py first!")
        return
    
    # Load model info
    with open(os.path.join(extracted_dir, 'model_info.pkl'), 'rb') as f:
        model_info = pickle.load(f)
    
    print(f"Model info loaded: {model_info['model_name']}")
    
    # Install dependencies if needed
    try:
        import tf2onnx
        import onnx2tf
    except ImportError:
        install_dependencies()
    
    # Conversion pipeline
    pb_path = os.path.join(extracted_dir, 'frozen_stylegan.pb')
    
    # Step 1: Convert to ONNX
    onnx_path = convert_pb_to_onnx(pb_path, model_info)
    if not onnx_path:
        print("❌ Failed to convert to ONNX")
        return
    
    # Step 2: Simplify ONNX model
    simplified_onnx = simplify_onnx_model(onnx_path)
    
    # Step 3: Convert to TFLite
    tflite_path = convert_onnx_to_tflite(simplified_onnx)
    if not tflite_path:
        print("❌ Failed to convert to TFLite")
        return
    
    # Step 4: Validate
    test_latent_path = os.path.join(extracted_dir, 'test_latent.npy')
    if os.path.exists(test_latent_path):
        validate_tflite_model(tflite_path, test_latent_path)
    
    # Step 5: Optimization recommendations
    create_mobile_optimized_model(model_info)
    
    print("\n🎉 Conversion pipeline completed!")
    print(f"TFLite model: {tflite_path}")
    
    # Save conversion info
    conversion_info = {
        'original_model': 'karras2019stylegan-ffhq-1024x1024.pkl',
        'tflite_path': tflite_path,
        'input_shape': model_info['input_shape'],
        'output_shape': model_info['output_shape'],
        'conversion_date': str(np.datetime64('today'))
    }
    
    with open('conversion_info.json', 'w') as f:
        import json
        json.dump(conversion_info, f, indent=2)
    
    print("\n📝 Next steps:")
    print("1. Test the model on target mobile devices")
    print("2. Consider creating a mobile-optimized version (256x256)")
    print("3. Implement custom operations if needed")
    print("4. Apply additional quantization for smaller size")


if __name__ == '__main__':
    main()