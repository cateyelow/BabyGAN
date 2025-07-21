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
      title: 'BabyGAN TFLite Debug',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'BabyGAN Debug - Gray Output Fix'),
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
  String _status = 'Tap button to load model and diagnose';
  String _debugInfo = '';
  List<Uint8List> _generatedImages = [];
  List<String> _methodNames = [];

  Future<void> _loadModelAndDiagnose() async {
    setState(() {
      _isLoading = true;
      _status = 'Loading model and running diagnostics...';
      _debugInfo = '';
      _generatedImages.clear();
      _methodNames.clear();
    });

    try {
      // Load TFLite model
      _interpreter = await Interpreter.fromAsset(
        'assets/models/stylegan_mobile_working.tflite',
      );
      
      // Get tensor info
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      
      String debug = 'Model Info:\n';
      debug += 'Input shape: $inputShape\n';
      debug += 'Output shape: $outputShape\n\n';
      
      // Test with a simple input
      debug += 'Testing model response:\n';
      
      // Test 1: All zeros
      final zeroInput = Float32List(512);
      final zeroOutput = Float32List(1 * 256 * 256 * 3);
      
      Map<int, Object> inputs = {0: zeroInput};
      Map<int, Object> outputs = {0: zeroOutput};
      _interpreter!.runForMultipleInputs(inputs, outputs);
      
      // Analyze zero input output
      double minVal = double.infinity;
      double maxVal = double.negativeInfinity;
      double sum = 0;
      
      for (int i = 0; i < zeroOutput.length; i++) {
        minVal = min(minVal, zeroOutput[i]);
        maxVal = max(maxVal, zeroOutput[i]);
        sum += zeroOutput[i];
      }
      
      double avgVal = sum / zeroOutput.length;
      debug += 'Zero input → Min: ${minVal.toStringAsFixed(4)}, Max: ${maxVal.toStringAsFixed(4)}, Avg: ${avgVal.toStringAsFixed(4)}\n';
      
      // Test 2: Random input
      final random = Random();
      final randomInput = Float32List(512);
      for (int i = 0; i < 512; i++) {
        randomInput[i] = random.nextGaussian();
      }
      
      final randomOutput = Float32List(1 * 256 * 256 * 3);
      inputs = {0: randomInput};
      outputs = {0: randomOutput};
      _interpreter!.runForMultipleInputs(inputs, outputs);
      
      // Analyze random input output
      minVal = double.infinity;
      maxVal = double.negativeInfinity;
      sum = 0;
      int negCount = 0;
      
      for (int i = 0; i < randomOutput.length; i++) {
        minVal = min(minVal, randomOutput[i]);
        maxVal = max(maxVal, randomOutput[i]);
        sum += randomOutput[i];
        if (randomOutput[i] < 0) negCount++;
      }
      
      avgVal = sum / randomOutput.length;
      double negPercent = (negCount / randomOutput.length) * 100;
      
      debug += 'Random input → Min: ${minVal.toStringAsFixed(4)}, Max: ${maxVal.toStringAsFixed(4)}, Avg: ${avgVal.toStringAsFixed(4)}\n';
      debug += 'Negative values: $negCount (${negPercent.toStringAsFixed(1)}%)\n\n';
      
      // Determine likely range
      debug += 'Diagnosis:\n';
      if (minVal < -0.5 && maxVal > 0.5) {
        debug += '✓ Output appears to be in [-1, 1] range (tanh activation)\n';
        debug += '→ Solution: Use (value + 1) * 127.5 conversion\n';
      } else if (minVal >= 0 && maxVal <= 1) {
        debug += '✓ Output appears to be in [0, 1] range (sigmoid activation)\n';
        debug += '→ Solution: Use value * 255 conversion\n';
      } else {
        debug += '⚠ Unusual output range detected\n';
        debug += '→ May need custom normalization\n';
      }
      
      // Check variance
      double variance = 0;
      for (int i = 0; i < 100; i++) {
        variance += pow(randomOutput[i] - avgVal, 2);
      }
      variance /= 100;
      
      if (variance < 0.0001) {
        debug += '⚠ WARNING: Low variance detected - model may not be responding to input\n';
      }
      
      setState(() {
        _status = 'Model loaded! Diagnostics complete.';
        _debugInfo = debug;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error: $e';
        _debugInfo = 'Stack trace: ${StackTrace.current}';
        _isLoading = false;
      });
    }
  }

  Future<void> _generateFaceAllMethods() async {
    if (_interpreter == null) {
      setState(() {
        _status = 'Please load model first';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _status = 'Generating faces with different methods...';
      _generatedImages.clear();
      _methodNames.clear();
    });

    try {
      final random = Random();
      
      // Method 1: Assume [0, 1] range
      final latent1 = List.generate(512, (_) => random.nextGaussian());
      final output1 = await _generateWithMethod(latent1, 'sigmoid');
      _generatedImages.add(output1);
      _methodNames.add('[0,1] → ×255');
      
      // Method 2: Assume [-1, 1] range
      final latent2 = List.generate(512, (_) => random.nextGaussian());
      final output2 = await _generateWithMethod(latent2, 'tanh');
      _generatedImages.add(output2);
      _methodNames.add('[-1,1] → +1 → ×127.5');
      
      // Method 3: Auto-normalize based on actual range
      final latent3 = List.generate(512, (_) => random.nextGaussian());
      final output3 = await _generateWithMethod(latent3, 'auto');
      _generatedImages.add(output3);
      _methodNames.add('Auto-normalize');
      
      // Method 4: Truncated normal input
      final latent4 = List.generate(512, (_) {
        double val;
        do {
          val = random.nextGaussian();
        } while (val.abs() > 2.0);
        return val;
      });
      final output4 = await _generateWithMethod(latent4, 'tanh');
      _generatedImages.add(output4);
      _methodNames.add('Truncated + [-1,1]');
      
      setState(() {
        _status = 'Generated ${_generatedImages.length} images with different methods';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error: $e';
        _isLoading = false;
      });
    }
  }

  Future<Uint8List> _generateWithMethod(List<double> latent, String method) async {
    // Prepare input
    final input = Float32List(512);
    for (int i = 0; i < 512; i++) {
      input[i] = latent[i].toDouble();
    }
    
    // Run inference
    final output = Float32List(1 * 256 * 256 * 3);
    Map<int, Object> inputs = {0: input};
    Map<int, Object> outputs = {0: output};
    _interpreter!.runForMultipleInputs(inputs, outputs);
    
    // Convert based on method
    return _convertOutputToImage(output, method);
  }

  Uint8List _convertOutputToImage(Float32List output, String method) {
    final image = img.Image(width: 256, height: 256);
    
    if (method == 'auto') {
      // Find actual min/max for auto-normalization
      double minVal = double.infinity;
      double maxVal = double.negativeInfinity;
      
      for (int i = 0; i < output.length; i++) {
        minVal = min(minVal, output[i]);
        maxVal = max(maxVal, output[i]);
      }
      
      double range = maxVal - minVal;
      if (range < 0.0001) range = 1.0; // Prevent division by zero
      
      int idx = 0;
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          final r = ((output[idx] - minVal) / range * 255).clamp(0, 255).toInt();
          final g = ((output[idx + 1] - minVal) / range * 255).clamp(0, 255).toInt();
          final b = ((output[idx + 2] - minVal) / range * 255).clamp(0, 255).toInt();
          
          image.setPixelRgb(x, y, r, g, b);
          idx += 3;
        }
      }
    } else {
      int idx = 0;
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          int r, g, b;
          
          if (method == 'sigmoid') {
            // Assume [0, 1] range
            r = (output[idx] * 255).clamp(0, 255).toInt();
            g = (output[idx + 1] * 255).clamp(0, 255).toInt();
            b = (output[idx + 2] * 255).clamp(0, 255).toInt();
          } else { // tanh
            // Assume [-1, 1] range
            r = ((output[idx] + 1) * 127.5).clamp(0, 255).toInt();
            g = ((output[idx + 1] + 1) * 127.5).clamp(0, 255).toInt();
            b = ((output[idx + 2] + 1) * 127.5).clamp(0, 255).toInt();
          }
          
          image.setPixelRgb(x, y, r, g, b);
          idx += 3;
        }
      }
    }
    
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
                const SizedBox(height: 20),
                if (_debugInfo.isNotEmpty)
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.grey[100],
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.grey),
                    ),
                    child: Text(
                      _debugInfo,
                      style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                    ),
                  ),
                const SizedBox(height: 20),
                if (_generatedImages.isNotEmpty)
                  Column(
                    children: [
                      const Text(
                        'Results with different conversion methods:',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 10),
                      ...List.generate(_generatedImages.length, (index) => 
                        Column(
                          children: [
                            Text(_methodNames[index]),
                            const SizedBox(height: 5),
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
                                  _generatedImages[index],
                                  fit: BoxFit.contain,
                                ),
                              ),
                            ),
                            const SizedBox(height: 15),
                          ],
                        ),
                      ),
                    ],
                  ),
                const SizedBox(height: 20),
                if (_isLoading)
                  const CircularProgressIndicator()
                else
                  Column(
                    children: [
                      ElevatedButton.icon(
                        onPressed: _loadModelAndDiagnose,
                        icon: const Icon(Icons.bug_report),
                        label: const Text('Load Model & Diagnose'),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                        ),
                      ),
                      const SizedBox(height: 10),
                      ElevatedButton.icon(
                        onPressed: _interpreter != null ? _generateFaceAllMethods : null,
                        icon: const Icon(Icons.auto_fix_high),
                        label: const Text('Generate Face (Try All Methods)'),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                        ),
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
    while (u1 == 0.0) u1 = nextDouble();
    u2 = nextDouble();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
  }
}