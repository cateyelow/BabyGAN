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
      home: const MyHomePage(title: 'BabyGAN TFLite Test'),
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
      // Get input and output shapes
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      
      print('Input shape: $inputShape');
      print('Output shape: $outputShape');
      
      // Generate random latent vector
      final random = Random();
      final latent = List.generate(512, (_) => random.nextGaussian());
      
      // Prepare input tensor - reshape to [1, 512]
      final input = [latent];
      
      // Prepare output tensor - shape [1, 256, 256, 3]
      final output = List.generate(
        1,
        (_) => List.generate(
          256,
          (_) => List.generate(
            256,
            (_) => List.generate(3, (_) => 0.0),
          ),
        ),
      );
      
      // Run inference
      _interpreter!.run(input, output);
      
      // Convert output to image
      final imageData = _convertToImage(output[0]);
      
      setState(() {
        _generatedImage = imageData;
        _status = 'Face generated successfully!';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error generating face: $e';
        _isLoading = false;
      });
      print('Error details: $e');
    }
  }

  Uint8List _convertToImage(List<List<List<double>>> output) {
    // Create image from output tensor
    final image = img.Image(width: 256, height: 256);
    
    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        final r = (output[y][x][0] * 255).clamp(0, 255).toInt();
        final g = (output[y][x][1] * 255).clamp(0, 255).toInt();
        final b = (output[y][x][2] * 255).clamp(0, 255).toInt();
        
        image.setPixelRgb(x, y, r, g, b);
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
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              _status,
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            if (_generatedImage != null)
              Container(
                width: 256,
                height: 256,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                ),
                child: Image.memory(
                  _generatedImage!,
                  fit: BoxFit.contain,
                ),
              ),
            const SizedBox(height: 20),
            if (_isLoading)
              const CircularProgressIndicator()
            else
              Column(
                children: [
                  ElevatedButton(
                    onPressed: _loadModel,
                    child: const Text('Load Model'),
                  ),
                  const SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: _interpreter != null ? _generateFace : null,
                    child: const Text('Generate Face'),
                  ),
                ],
              ),
          ],
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