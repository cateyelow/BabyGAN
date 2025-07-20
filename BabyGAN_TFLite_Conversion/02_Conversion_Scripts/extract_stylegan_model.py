#!/usr/bin/env python3
"""
StyleGAN Model Extraction for TensorFlow Lite Conversion
Extracts the synthesis network from StyleGAN pickle and prepares it for conversion
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf

# Add dnnlib to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import dnnlib
import dnnlib.tflib as tflib


def extract_synthesis_network(pkl_path, output_dir='extracted_model'):
    """Extract synthesis network from StyleGAN pickle file"""
    
    print(f"Loading StyleGAN from {pkl_path}...")
    
    # Initialize TensorFlow
    tflib.init_tf()
    
    # Load the model
    with open(pkl_path, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    
    print("Model loaded successfully!")
    print(f"Synthesis network: {Gs}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model info
    print("\nModel Information:")
    print(f"- Output shape: {Gs.output_shape}")
    print(f"- Input shape: {Gs.input_shape}")
    print(f"- Latent size: {Gs.input_shape[1]}")
    
    # Create a session and initialize
    sess = tf.get_default_session()
    
    # Define fixed input shape for mobile (reduced resolution)
    batch_size = 1
    latent_size = Gs.input_shape[1]
    
    # Create placeholder for latent input
    latent_placeholder = tf.placeholder(
        tf.float32, 
        shape=[batch_size, latent_size], 
        name='latent_input'
    )
    
    # Generate output using the synthesis network
    with tf.variable_scope('Gs'):
        images = Gs.get_output_for(latent_placeholder, is_training=False, randomize_noise=False)
    
    # Get output tensor name
    output_tensor = tf.identity(images, name='generated_image')
    
    print(f"\nInput tensor: {latent_placeholder.name}")
    print(f"Output tensor: {output_tensor.name}")
    
    # Save the graph definition
    graph_def_path = os.path.join(output_dir, 'stylegan_graph.pb')
    tf.train.write_graph(sess.graph_def, output_dir, 'stylegan_graph.pb', as_text=False)
    print(f"Saved graph definition to {graph_def_path}")
    
    # Freeze the graph
    print("\nFreezing graph...")
    freeze_graph(sess, output_dir, output_tensor.name.split(':')[0])
    
    # Save model info
    save_model_info(output_dir, Gs, latent_placeholder.name, output_tensor.name)
    
    return output_dir


def freeze_graph(sess, output_dir, output_node_name):
    """Freeze the graph with variables converted to constants"""
    
    from tensorflow.python.framework import graph_util
    
    # Get the graph definition
    graph_def = sess.graph.as_graph_def()
    
    # Get all variable names
    variable_names = [v.name for v in tf.global_variables()]
    print(f"Found {len(variable_names)} variables to freeze")
    
    # Convert variables to constants
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        [output_node_name]
    )
    
    # Save frozen graph
    frozen_path = os.path.join(output_dir, 'frozen_stylegan.pb')
    with tf.gfile.GFile(frozen_path, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
    
    print(f"Saved frozen graph to {frozen_path}")
    print(f"Graph size: {len(frozen_graph_def.SerializeToString()) / 1024 / 1024:.2f} MB")
    
    # Count operations
    print(f"Total operations: {len(frozen_graph_def.node)}")
    
    # List operation types
    op_types = set(node.op for node in frozen_graph_def.node)
    print(f"Unique operation types: {len(op_types)}")
    
    return frozen_path


def save_model_info(output_dir, model, input_name, output_name):
    """Save model information for later use"""
    
    info = {
        'input_name': input_name,
        'output_name': output_name,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'resolution': model.output_shape[2],
        'latent_size': model.input_shape[1],
        'num_layers': len(model.vars),
        'model_name': 'stylegan-ffhq'
    }
    
    info_path = os.path.join(output_dir, 'model_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(info, f)
    
    # Also save as text for reference
    info_txt_path = os.path.join(output_dir, 'model_info.txt')
    with open(info_txt_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Saved model info to {info_path}")


def create_test_latent(latent_size, output_path):
    """Create a test latent vector for validation"""
    
    # Create random latent
    latent = np.random.randn(1, latent_size).astype(np.float32)
    
    # Save for testing
    np.save(output_path, latent)
    print(f"Saved test latent to {output_path}")
    
    return latent


def test_extracted_model(frozen_graph_path, latent_vector, input_name, output_name):
    """Test the extracted frozen graph"""
    
    print("\nTesting extracted model...")
    
    # Create new graph
    with tf.Graph().as_default():
        # Load frozen graph
        with tf.gfile.GFile(frozen_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        # Import graph
        tf.import_graph_def(graph_def, name='')
        
        # Get input and output tensors
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name(input_name)
        output_tensor = graph.get_tensor_by_name(output_name)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")
        
        # Run inference
        with tf.Session() as sess:
            output = sess.run(output_tensor, feed_dict={input_tensor: latent_vector})
            print(f"Generated image shape: {output.shape}")
            print(f"Value range: [{output.min():.3f}, {output.max():.3f}]")
            
            return output


def main():
    """Main extraction function"""
    
    # Check if model exists
    pkl_path = 'content/BabyGAN/karras2019stylegan-ffhq-1024x1024.pkl'
    if not os.path.exists(pkl_path):
        print(f"Error: Model file not found at {pkl_path}")
        print("Please download the model from NVIDIA and place it in the project directory")
        return
    
    # Extract model
    output_dir = extract_synthesis_network(pkl_path)
    
    # Load model info
    with open(os.path.join(output_dir, 'model_info.pkl'), 'rb') as f:
        info = pickle.load(f)
    
    # Create test latent
    test_latent = create_test_latent(
        info['latent_size'], 
        os.path.join(output_dir, 'test_latent.npy')
    )
    
    # Test frozen graph
    frozen_path = os.path.join(output_dir, 'frozen_stylegan.pb')
    test_extracted_model(
        frozen_path, 
        test_latent, 
        info['input_name'], 
        info['output_name']
    )
    
    print("\n✅ Model extraction completed successfully!")
    print(f"Next steps: Convert {frozen_path} to ONNX/TFLite")


if __name__ == '__main__':
    main()