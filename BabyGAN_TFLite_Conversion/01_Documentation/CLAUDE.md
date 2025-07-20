# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BabyGAN is a StyleGAN-based face generation system that can:
- Generate baby faces by mixing parent faces
- Manipulate face attributes (age, gender, smile, etc.)
- Encode real images into latent space for editing

**Current Stack**: TensorFlow 1.10.0 (GPU), NVIDIA StyleGAN architecture, 1024x1024 resolution

## Common Development Commands

```bash
# Install dependencies (Python 3.6 64-bit required)
pip install tensorflow-gpu==1.10.0 h5py==2.10.0 opencv-python pillow imageio moviepy

# Align face images for encoding
python align_images.py raw_images/ aligned_images/

# Encode aligned images to latent representations
python encode_images.py aligned_images/ generated_images/ latent_representations/

# Run the main notebook (English version)
jupyter notebook BabyGAN_(ENG).ipynb

# Train ResNet for latent approximation
python train_resnet.py data/ffhq_dataset/tfrecords/ffhq
```

## Architecture & Key Components

### Core Model Architecture
- **Base**: NVIDIA StyleGAN (karras2019stylegan-ffhq-1024x1024.pkl)
- **Latent Space**: W+ space with shape [batch_size, 18, 512]
- **Resolution**: 1024x1024 (18 layers for this resolution)
- **Framework**: Custom dnnlib with TensorFlow 1.x sessions

### Key Files
- `encoder/generator_model.py`: Generator wrapper, synthesis network interface
- `encoder/perceptual_model.py`: Image encoding with perceptual loss
- `dnnlib/tflib/network.py`: Dynamic network loading from pickle files
- `config.py`: Global settings (GPU, paths, model configuration)
- `ffhq_dataset/latent_directions/*.npy`: Pre-computed attribute vectors

### Critical Implementation Details
- Model uses dynamic graph construction from pickled Python functions
- Heavy reliance on tf.contrib operations (deprecated in TF2)
- Session-based execution with placeholders and feed_dict
- Custom NCHW data format for performance

## TensorFlow Lite Conversion Plan

### ⚠️ Major Challenges
1. **Dynamic Graph Construction**: Model is built from pickled Python code at runtime
2. **TF1.x Architecture**: Session-based, uses deprecated tf.contrib modules
3. **Unsupported Operations**: ScipyOptimizerInterface, dynamic shapes, complex control flow
4. **Model Size**: ~500MB+ model, 1024x1024 resolution too large for mobile

### 📋 Conversion Strategy

#### Option 1: Direct Conversion Path (Difficult)
```bash
# Step 1: Extract frozen graph from TF1 session
python freeze_graph.py \
    --input_checkpoint=model.ckpt \
    --output_graph=frozen_model.pb \
    --output_node_names="G_synthesis_1/_Run/concat:0"

# Step 2: Convert to TF2 SavedModel
tf_upgrade_v2 --infile=frozen_model.pb --outfile=model_v2.pb

# Step 3: Convert to TFLite (will likely fail due to unsupported ops)
tflite_convert \
    --saved_model_dir=saved_model \
    --output_file=model.tflite \
    --allow_custom_ops
```

#### Option 2: ONNX Intermediate Path (Recommended)
```bash
# Step 1: Export TF1 model to ONNX
python -m tf2onnx.convert \
    --graphdef frozen_model.pb \
    --output stylegan.onnx \
    --inputs "Gs_network/dlatents_in:0" \
    --outputs "Gs_network/images_out:0"

# Step 2: Use onnx2tf for direct TFLite conversion
pip install onnx2tf
onnx2tf -i stylegan.onnx -o tflite_model -oiqt

# Step 3: Quantize for mobile deployment
python quantize_model.py --input=model.tflite --output=model_quant.tflite
```

#### Option 3: Re-implementation Strategy (Most Viable)
1. Use existing PyTorch StyleGAN2 implementations
2. Export PyTorch → ONNX → TFLite
3. Reduce resolution to 256x256 or 512x512
4. Use MobileStyleGAN architecture for efficiency

### 🔧 Implementation Steps for TFLite Conversion

1. **Model Extraction**
   ```python
   # Extract synthesis network from pickle
   import pickle
   with open('karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
       G, D, Gs = pickle.load(f)
   # Save Gs network graph definition
   ```

2. **Graph Simplification**
   - Remove discriminator components
   - Extract only synthesis network
   - Replace tf.contrib operations
   - Convert NCHW to NHWC format

3. **Resolution Reduction**
   - Target 256x256 or 512x512 for mobile
   - Adjust layer count (18 → 14 for 512x512)
   - Reduce latent dimensions if needed

4. **Operation Replacement**
   ```python
   # Replace unsupported ops
   replacements = {
       'tf.contrib.opt.ScipyOptimizerInterface': 'tf.keras.optimizers.Adam',
       'tf.nn.fused_batch_norm': 'tf.nn.batch_normalization',
       'tf.contrib.layers.xavier_initializer': 'tf.keras.initializers.GlorotUniform'
   }
   ```

5. **Mobile Optimization**
   - Post-training quantization (INT8)
   - Pruning unnecessary layers
   - Model compression techniques

### 🚀 Alternative: Use Pre-converted Models

Consider using existing mobile-friendly alternatives:
- **MobileStyleGAN**: Designed for mobile deployment
- **StyleGAN2-ADA PyTorch**: Better conversion support
- **TFLite Model Zoo**: Pre-converted generative models

### 📊 Expected Results

| Aspect | Original | TFLite Target |
|--------|----------|---------------|
| Resolution | 1024x1024 | 256x256 or 512x512 |
| Model Size | ~500MB | <50MB (quantized) |
| Inference Time | ~2s (GPU) | <500ms (mobile GPU) |
| Quality | High | Medium (acceptable) |

### 🛠️ Useful Tools & Resources

```bash
# Install conversion tools
pip install tf2onnx onnx2tf tensorflow==2.x onnx onnxruntime

# Visualization tools
pip install netron  # View model architecture
netron model.onnx  # Open in browser

# TFLite tools
pip install tflite-support tflite-runtime
```

### 💡 Development Tips

1. **Start Small**: Test conversion with a smaller StyleGAN model (256x256)
2. **Validate Each Step**: Ensure outputs match at each conversion stage
3. **Profile Performance**: Use TFLite benchmark tool for mobile testing
4. **Consider Alternatives**: If conversion fails, consider training a mobile-specific model

### 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Op not supported" | Use `--allow_custom_ops` or replace with supported ops |
| Dynamic shapes | Set fixed input shapes during conversion |
| Memory overflow | Reduce batch size to 1, lower resolution |
| NCHW/NHWC mismatch | Use onnx2tf which handles this automatically |

### 📝 Validation Script

```python
# validate_conversion.py
import numpy as np
import tensorflow as tf

def compare_outputs(original_model, tflite_path, test_latent):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_latent)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Compare with original
    original_output = original_model.run(test_latent)
    
    # Calculate similarity
    mse = np.mean((original_output - tflite_output) ** 2)
    print(f"MSE: {mse}, PSNR: {20 * np.log10(255.0 / np.sqrt(mse))}")
```

## Next Steps

1. **Attempt ONNX conversion** first as it's the most promising path
2. **Consider PyTorch alternatives** if TF1→TFLite proves too difficult
3. **Reduce model complexity** for mobile deployment
4. **Test on target devices** early in the process