
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
