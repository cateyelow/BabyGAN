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
      title: 'BabyGAN Android Fixed',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'BabyGAN Android Solution'),
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
  String _debugInfo = '';
  
  // Android-specific fixes
  bool _useCpuOnly = true;
  bool _useExplicitBuffers = true;
  bool _autoScaleOutput = true;
  int _numThreads = 1;

  Future<void> _loadModel() async {
    setState(() {
      _isLoading = true;
      _status = 'Loading model with Android fixes...';
      _debugInfo = '';
    });

    try {
      // Close existing interpreter
      _interpreter?.close();
      
      // Configure interpreter options for Android
      final options = InterpreterOptions();
      
      if (_useCpuOnly) {
        // Disable NNAPI and GPU delegates which cause issues on Android
        options.useNnApiForAndroid = false;
        // Don't add any delegates
      }
      
      // Set number of threads
      options.threads = _numThreads;
      
      // Load model
      _interpreter = await Interpreter.fromAsset(
        'assets/models/stylegan_mobile_working.tflite',
        options: options,
      );
      
      // Get model info
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      
      String debug = 'Model Configuration:\n';
      debug += '• CPU Only: $_useCpuOnly\n';
      debug += '• Threads: $_numThreads\n';
      debug += '• Input shape: $inputShape\n';
      debug += '• Output shape: $outputShape\n';
      
      setState(() {
        _status = 'Model loaded successfully!';
        _debugInfo = debug;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error loading model: $e';
        _debugInfo = 'Stack trace: ${StackTrace.current}';
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
      _status = 'Generating face with Android fixes...';
    });

    try {
      // Generate latent vector with truncated normal distribution
      final random = Random();
      final latent = List<double>.generate(512, (_) {
        double value;
        do {
          value = random.nextGaussian();
        } while (value.abs() > 2.0); // Truncate at 2 standard deviations
        return value;
      });
      
      // Prepare input with explicit buffer management
      Float32List input;
      if (_useExplicitBuffers) {
        // Create aligned buffer
        input = Float32List(512);
        for (int i = 0; i < 512; i++) {
          input[i] = latent[i];
        }
      } else {
        input = Float32List.fromList(latent);
      }
      
      // Get output shape and calculate size
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      int outputSize = 1;
      for (var dim in outputShape) {
        outputSize *= dim;
      }
      
      // Prepare output with explicit buffer management
      final output = Float32List(outputSize);
      
      // Create new buffers for inference (Android workaround)
      final inputBuffer = input.buffer.asFloat32List();
      final outputBuffer = output.buffer.asFloat32List();
      
      // Run inference
      Map<int, Object> inputs = {0: inputBuffer};
      Map<int, Object> outputs = {0: outputBuffer};
      
      _interpreter!.runForMultipleInputs(inputs, outputs);
      
      // Analyze output for debugging
      double minVal = double.infinity;
      double maxVal = double.negativeInfinity;
      double sum = 0;
      int zeroCount = 0;
      Set<double> uniqueValues = {};
      
      for (int i = 0; i < outputBuffer.length; i++) {
        final val = outputBuffer[i];
        minVal = min(minVal, val);
        maxVal = max(maxVal, val);
        sum += val;
        if (val == 0.0) zeroCount++;
        if (i < 1000) uniqueValues.add(val); // Sample first 1000 values
      }
      
      double avgVal = sum / outputBuffer.length;
      
      String debug = 'Generation Results:\n';
      debug += '• Output range: [${minVal.toStringAsFixed(4)}, ${maxVal.toStringAsFixed(4)}]\n';
      debug += '• Average value: ${avgVal.toStringAsFixed(4)}\n';
      debug += '• Zero values: $zeroCount (${(zeroCount * 100.0 / outputBuffer.length).toStringAsFixed(1)}%)\n';
      debug += '• Unique values (sample): ${uniqueValues.length}\n';
      
      // Detect output type
      String outputType = 'Unknown';
      if (minVal >= -1.1 && maxVal <= 1.1 && minVal < -0.5) {
        outputType = 'Tanh [-1, 1]';
      } else if (minVal >= -0.1 && maxVal <= 1.1) {
        outputType = 'Sigmoid [0, 1]';
      } else if (minVal >= -0.1 && maxVal >= 200) {
        outputType = 'Raw [0, 255]';
      }
      debug += '• Detected output type: $outputType\n';
      
      // Convert output to image
      final imageData = _convertOutputToImage(outputBuffer, outputType);
      
      setState(() {
        _generatedImage = imageData;
        _status = 'Face generated successfully!';
        _debugInfo = debug;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Error: ${e.toString()}';
        _debugInfo = 'Stack trace: ${StackTrace.current}';
        _isLoading = false;
      });
    }
  }

  Uint8List _convertOutputToImage(Float32List output, String detectedType) {
    final image = img.Image(width: 256, height: 256);
    
    int idx = 0;
    
    if (_autoScaleOutput) {
      // Auto-scale based on actual range
      double min = double.infinity;
      double max = double.negativeInfinity;
      
      for (var val in output) {
        if (val < min) min = val;
        if (val > max) max = val;
      }
      
      double range = max - min;
      if (range < 0.0001) {
        // If range is too small, assume it's a constant color
        range = 1.0;
      }
      
      idx = 0;
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          final r = ((output[idx] - min) / range * 255).clamp(0, 255).toInt();
          final g = ((output[idx + 1] - min) / range * 255).clamp(0, 255).toInt();
          final b = ((output[idx + 2] - min) / range * 255).clamp(0, 255).toInt();
          
          image.setPixelRgb(x, y, r, g, b);
          idx += 3;
        }
      }
    } else {
      // Use detected type for conversion
      idx = 0;
      for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
          int r, g, b;
          
          if (detectedType == 'Tanh [-1, 1]') {
            // Convert from [-1, 1] to [0, 255]
            r = ((output[idx] + 1) * 127.5).clamp(0, 255).toInt();
            g = ((output[idx + 1] + 1) * 127.5).clamp(0, 255).toInt();
            b = ((output[idx + 2] + 1) * 127.5).clamp(0, 255).toInt();
          } else if (detectedType == 'Raw [0, 255]') {
            // Already in [0, 255] range
            r = output[idx].clamp(0, 255).toInt();
            g = output[idx + 1].clamp(0, 255).toInt();
            b = output[idx + 2].clamp(0, 255).toInt();
          } else {
            // Assume [0, 1] range
            r = (output[idx] * 255).clamp(0, 255).toInt();
            g = (output[idx + 1] * 255).clamp(0, 255).toInt();
            b = (output[idx + 2] * 255).clamp(0, 255).toInt();
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
                
                // Android-specific settings
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Android Fixes:',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        SwitchListTile(
                          title: const Text('CPU Only Mode'),
                          subtitle: const Text('Disable GPU/NNAPI (fixes gray output)'),
                          value: _useCpuOnly,
                          onChanged: (value) {
                            setState(() {
                              _useCpuOnly = value;
                            });
                          },
                        ),
                        SwitchListTile(
                          title: const Text('Explicit Buffer Management'),
                          subtitle: const Text('Android memory alignment fix'),
                          value: _useExplicitBuffers,
                          onChanged: (value) {
                            setState(() {
                              _useExplicitBuffers = value;
                            });
                          },
                        ),
                        SwitchListTile(
                          title: const Text('Auto-Scale Output'),
                          subtitle: const Text('Automatically normalize output range'),
                          value: _autoScaleOutput,
                          onChanged: (value) {
                            setState(() {
                              _autoScaleOutput = value;
                            });
                          },
                        ),
                        ListTile(
                          title: const Text('CPU Threads'),
                          subtitle: Text('Current: $_numThreads'),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              IconButton(
                                icon: const Icon(Icons.remove),
                                onPressed: _numThreads > 1
                                    ? () => setState(() => _numThreads--)
                                    : null,
                              ),
                              Text('$_numThreads'),
                              IconButton(
                                icon: const Icon(Icons.add),
                                onPressed: _numThreads < 8
                                    ? () => setState(() => _numThreads++)
                                    : null,
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
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
                        onPressed: _loadModel,
                        icon: const Icon(Icons.download),
                        label: const Text('Load Model (with fixes)'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blue,
                          foregroundColor: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 10),
                      ElevatedButton.icon(
                        onPressed: _interpreter != null ? _generateFace : null,
                        icon: const Icon(Icons.face),
                        label: const Text('Generate Face'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          foregroundColor: Colors.white,
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

// Extension for Random
extension RandomGaussian on Random {
  double nextGaussian() {
    double u1 = 0.0;
    double u2 = 0.0;
    while (u1 == 0.0) u1 = nextDouble();
    u2 = nextDouble();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
  }
}