import 'package:flutter/material.dart';
import 'dart:typed_data';
import 'dart:math';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BabyGAN TFLite Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'BabyGAN TFLite - Corrected'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  Interpreter? _interpreter;
  bool _isLoading = false;
  String _status = 'Tap button to load model';
  Uint8List? _generatedImage;
  String _conversionMethod = 'tanh'; // Default to tanh range

  Future<void> _loadModel() async {
    setState(() {
      _isLoading = true;
      _status = 'Loading model...';
    });

    try {
      // Load TFLite model
      _interpreter = await Interpreter.fromAsset(
        'assets/models/stylegan_mobile_working.tflite',
      );
      
      // Print model info
      print('Input shape: ${_interpreter!.getInputTensor(0).shape}');
      print('Output shape: ${_interpreter!.getOutputTensor(0).shape}');
      
      setState(() {
        _status = 'Model loaded successfully!';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error loading model: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _generateFace() async {
    if (_interpreter == null) {
      setState(() {
        _status = 'Please load model first';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _status = 'Generating face...';
    });

    try {
      // Generate random latent vector with truncated normal distribution
      final random = Random();
      final latent = List.generate(512, (_) {
        double value;
        do {
          value = random.nextGaussian();
        } while (value.abs() > 2.0); // Truncate at 2 standard deviations
        return value;
      });
      
      // Prepare input tensor as Float32List with shape [1, 512]
      final input = Float32List(512);
      for (int i = 0; i < 512; i++) {
        input[i] = latent[i].toDouble();
      }
      
      // Get output shape
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      print('Output tensor shape: $outputShape');
      
      // Calculate total output size
      int outputSize = 1;
      for (var dim in outputShape) {
        outputSize *= dim;
      }
      
      // Prepare output tensor as flat Float32List
      final output = Float32List(outputSize);
      
      // Create properly shaped input and output for interpreter
      var inputBuffer = input.buffer.asFloat32List();
      var outputBuffer = output.buffer.asFloat32List();
      
      // Run inference with proper shapes
      Map<int, Object> inputs = {0: inputBuffer};
      Map<int, Object> outputs = {0: outputBuffer};
      
      _interpreter!.runForMultipleInputs(inputs, outputs);
      
      // Analyze output range for debugging
      double minVal = double.infinity;
      double maxVal = double.negativeInfinity;
      for (int i = 0; i < outputBuffer.length; i++) {
        minVal = min(minVal, outputBuffer[i]);
        maxVal = max(maxVal, outputBuffer[i]);
      }
      print('Output range: Min=$minVal, Max=$maxVal');
      
      // Convert output to image with proper normalization
      final imageData = _convertOutputToImage(outputBuffer);
      
      setState(() {
        _generatedImage = imageData;
        _status = 'Face generated! (Range: [${minVal.toStringAsFixed(2)}, ${maxVal.toStringAsFixed(2)}])';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error: ${e.toString()}';
        _isLoading = false;
      });
      print('Error details: $e');
      print('Stack trace: ${StackTrace.current}');
    }
  }

  Uint8List _convertOutputToImage(Float32List output) {
    // Create image from flat output array
    final image = img.Image(width: 256, height: 256);
    
    // The output is in format [batch, height, width, channels]
    // Since batch=1, we can treat it as [height, width, channels]
    int idx = 0;
    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        int r, g, b;
        
        if (_conversionMethod == 'tanh') {
          // Convert from [-1, 1] to [0, 255] (most StyleGAN models use tanh)
          r = ((output[idx] + 1) * 127.5).clamp(0, 255).toInt();
          g = ((output[idx + 1] + 1) * 127.5).clamp(0, 255).toInt();
          b = ((output[idx + 2] + 1) * 127.5).clamp(0, 255).toInt();
        } else {
          // Convert from [0, 1] to [0, 255] (sigmoid)
          r = (output[idx] * 255).clamp(0, 255).toInt();
          g = (output[idx + 1] * 255).clamp(0, 255).toInt();
          b = (output[idx + 2] * 255).clamp(0, 255).toInt();
        }
        
        image.setPixelRgb(x, y, r, g, b);
        idx += 3;
      }
    }
    
    // Encode to PNG
    return Uint8List.fromList(img.encodePng(image));
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                Text(
                  _status,
                  style: Theme.of(context).textTheme.headlineSmall,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 10),
                // Conversion method selector
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text('Conversion: '),
                    Radio<String>(
                      value: 'tanh',
                      groupValue: _conversionMethod,
                      onChanged: (value) {
                        setState(() {
                          _conversionMethod = value!;
                        });
                      },
                    ),
                    const Text('[-1,1] → [0,255]'),
                    const SizedBox(width: 20),
                    Radio<String>(
                      value: 'sigmoid',
                      groupValue: _conversionMethod,
                      onChanged: (value) {
                        setState(() {
                          _conversionMethod = value!;
                        });
                      },
                    ),
                    const Text('[0,1] → [0,255]'),
                  ],
                ),
                const SizedBox(height: 20),
                if (_generatedImage != null)
                  Container(
                    width: 256,
                    height: 256,
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey, width: 2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: Image.memory(
                        _generatedImage!,
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
                const SizedBox(height: 20),
                if (_isLoading)
                  const CircularProgressIndicator()
                else
                  Column(
                    children: [
                      ElevatedButton.icon(
                        onPressed: _interpreter == null ? _loadModel : null,
                        icon: const Icon(Icons.download),
                        label: const Text('Load Model'),
                      ),
                      const SizedBox(height: 10),
                      ElevatedButton.icon(
                        onPressed: _interpreter != null ? _generateFace : null,
                        icon: const Icon(Icons.face),
                        label: const Text('Generate Face'),
                      ),
                    ],
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// Extension for Random to generate Gaussian distribution
extension RandomGaussian on Random {
  double nextGaussian() {
    // Box-Muller transform
    double u1 = 0.0;
    double u2 = 0.0;
    while (u1 == 0.0) u1 = nextDouble(); // Avoid log(0)
    u2 = nextDouble();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
  }
}