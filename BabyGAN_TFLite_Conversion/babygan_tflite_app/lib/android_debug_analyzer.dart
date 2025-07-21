import 'package:flutter/material.dart';
import 'dart:typed_data';
import 'dart:math';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:io' show Platform;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BabyGAN Android Debug',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Android Gray Background Analyzer'),
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
  String _status = 'Tap to start Android-specific diagnosis';
  String _debugInfo = '';
  List<Uint8List> _testImages = [];
  List<String> _testNames = [];

  Future<void> _runAndroidDiagnosis() async {
    setState(() {
      _isLoading = true;
      _status = 'Running Android-specific diagnosis...';
      _debugInfo = '';
      _testImages.clear();
      _testNames.clear();
    });

    String debug = '=== Android TFLite Diagnosis ===\n\n';
    
    try {
      // Test 1: Basic model loading
      debug += '1. Model Loading Test:\n';
      try {
        final options = InterpreterOptions();
        _interpreter = await Interpreter.fromAsset(
          'assets/models/stylegan_mobile_working.tflite',
          options: options,
        );
        debug += '✓ Model loaded successfully\n';
        debug += 'Input: ${_interpreter!.getInputTensor(0).shape}\n';
        debug += 'Output: ${_interpreter!.getOutputTensor(0).shape}\n\n';
      } catch (e) {
        debug += '✗ Model loading failed: $e\n\n';
        throw e;
      }

      // Test 2: CPU-only execution
      debug += '2. CPU-Only Execution Test:\n';
      try {
        final cpuResult = await _testWithDelegate('CPU', null);
        _testImages.add(cpuResult);
        _testNames.add('CPU Only');
        debug += '✓ CPU execution completed\n';
      } catch (e) {
        debug += '✗ CPU execution failed: $e\n';
      }

      // Test 3: GPU Delegate test (Android specific)
      debug += '\n3. GPU Delegate Test:\n';
      try {
        final gpuOptions = InterpreterOptions()..useNnApiForAndroid = false;
        gpuOptions.addDelegate(GpuDelegateV2());
        
        final gpuInterpreter = await Interpreter.fromAsset(
          'assets/models/stylegan_mobile_working.tflite',
          options: gpuOptions,
        );
        
        final gpuResult = await _runInference(gpuInterpreter, 'GPU');
        _testImages.add(gpuResult);
        _testNames.add('GPU Delegate');
        debug += '✓ GPU delegate execution completed\n';
        
        gpuInterpreter.close();
      } catch (e) {
        debug += '✗ GPU delegate not supported or failed: $e\n';
        debug += '→ This is often the cause of gray outputs!\n';
      }

      // Test 4: NNAPI test (Android specific)
      debug += '\n4. NNAPI Test:\n';
      try {
        final nnapiOptions = InterpreterOptions()..useNnApiForAndroid = true;
        
        final nnapiInterpreter = await Interpreter.fromAsset(
          'assets/models/stylegan_mobile_working.tflite',
          options: nnapiOptions,
        );
        
        final nnapiResult = await _runInference(nnapiInterpreter, 'NNAPI');
        _testImages.add(nnapiResult);
        _testNames.add('NNAPI');
        debug += '✓ NNAPI execution completed\n';
        
        nnapiInterpreter.close();
      } catch (e) {
        debug += '✗ NNAPI failed: $e\n';
      }

      // Test 5: Memory alignment test
      debug += '\n5. Memory Alignment Test:\n';
      final alignmentIssues = await _testMemoryAlignment();
      debug += alignmentIssues;

      // Test 6: Float precision test
      debug += '\n6. Float Precision Test:\n';
      final precisionInfo = await _testFloatPrecision();
      debug += precisionInfo;

      // Test 7: Threading test
      debug += '\n7. Threading Configuration:\n';
      try {
        final threadOptions = InterpreterOptions()..threads = 4;
        final threadInterpreter = await Interpreter.fromAsset(
          'assets/models/stylegan_mobile_working.tflite',
          options: threadOptions,
        );
        
        final threadResult = await _runInference(threadInterpreter, 'Multi-thread');
        _testImages.add(threadResult);
        _testNames.add('4 Threads');
        debug += '✓ Multi-threading test completed\n';
        
        threadInterpreter.close();
      } catch (e) {
        debug += '✗ Threading test failed: $e\n';
      }

      // Final diagnosis
      debug += '\n=== DIAGNOSIS SUMMARY ===\n';
      debug += _generateDiagnosis();

      setState(() {
        _status = 'Diagnosis complete!';
        _debugInfo = debug;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = 'Diagnosis failed: $e';
        _debugInfo = debug + '\n\nError: $e\n${StackTrace.current}';
        _isLoading = false;
      });
    }
  }

  Future<Uint8List> _testWithDelegate(String name, Delegate? delegate) async {
    final options = InterpreterOptions();
    if (delegate != null) {
      options.addDelegate(delegate);
    }
    
    final interpreter = await Interpreter.fromAsset(
      'assets/models/stylegan_mobile_working.tflite',
      options: options,
    );
    
    final result = await _runInference(interpreter, name);
    interpreter.close();
    return result;
  }

  Future<Uint8List> _runInference(Interpreter interpreter, String configName) async {
    // Create test input
    final random = Random(42); // Fixed seed for consistency
    final input = Float32List(512);
    for (int i = 0; i < 512; i++) {
      input[i] = random.nextGaussian() * 0.5; // Smaller values for testing
    }
    
    // Prepare output
    final outputShape = interpreter.getOutputTensor(0).shape;
    int outputSize = 1;
    for (var dim in outputShape) {
      outputSize *= dim;
    }
    final output = Float32List(outputSize);
    
    // Run inference
    Map<int, Object> inputs = {0: input};
    Map<int, Object> outputs = {0: output};
    interpreter.runForMultipleInputs(inputs, outputs);
    
    // Analyze output
    double min = double.infinity;
    double max = double.negativeInfinity;
    double sum = 0;
    Set<double> uniqueValues = {};
    
    for (int i = 0; i < min(1000, output.length); i++) {
      min = min > output[i] ? output[i] : min;
      max = max < output[i] ? output[i] : max;
      sum += output[i];
      uniqueValues.add(output[i]);
    }
    
    print('$configName - Range: [$min, $max], Unique values: ${uniqueValues.length}');
    
    // Convert to image with auto-scaling
    return _convertWithAutoScale(output, configName);
  }

  Uint8List _convertWithAutoScale(Float32List output, String label) {
    final image = img.Image(width: 256, height: 256);
    
    // Find actual range
    double min = double.infinity;
    double max = double.negativeInfinity;
    for (var val in output) {
      if (val < min) min = val;
      if (val > max) max = val;
    }
    
    // Add label
    img.drawString(image, img.arial_14, 5, 5, label);
    img.drawString(image, img.arial_14, 5, 20, 'Range: [${min.toStringAsFixed(2)}, ${max.toStringAsFixed(2)}]');
    
    // Auto-scale to full range
    double range = max - min;
    if (range < 0.0001) range = 1.0;
    
    int idx = 0;
    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        final r = ((output[idx] - min) / range * 255).clamp(0, 255).toInt();
        final g = ((output[idx + 1] - min) / range * 255).clamp(0, 255).toInt();
        final b = ((output[idx + 2] - min) / range * 255).clamp(0, 255).toInt();
        
        image.setPixelRgb(x, y, r, g, b);
        idx += 3;
      }
    }
    
    return Uint8List.fromList(img.encodePng(image));
  }

  Future<String> _testMemoryAlignment() async {
    String result = '';
    
    // Test different buffer alignments
    final input = Float32List(512);
    final output = Float32List(1 * 256 * 256 * 3);
    
    // Check alignment
    final inputAddr = input.buffer.lengthInBytes;
    final outputAddr = output.buffer.lengthInBytes;
    
    result += 'Input buffer size: $inputAddr bytes\n';
    result += 'Output buffer size: $outputAddr bytes\n';
    result += 'Input alignment: ${inputAddr % 16 == 0 ? "✓ 16-byte aligned" : "✗ Not 16-byte aligned"}\n';
    result += 'Output alignment: ${outputAddr % 16 == 0 ? "✓ 16-byte aligned" : "✗ Not 16-byte aligned"}\n';
    
    return result;
  }

  Future<String> _testFloatPrecision() async {
    String result = '';
    
    // Test float precision
    final testValue = 0.123456789;
    final float32 = Float32List(1);
    float32[0] = testValue;
    
    result += 'Original: $testValue\n';
    result += 'Float32: ${float32[0]}\n';
    result += 'Precision loss: ${(testValue - float32[0]).abs()}\n';
    
    // Test denormalized numbers
    final tiny = 1e-38;
    float32[0] = tiny;
    result += 'Denormalized test: ${float32[0] == 0 ? "✗ Flushed to zero" : "✓ Preserved"}\n';
    
    return result;
  }

  String _generateDiagnosis() {
    String diagnosis = '';
    
    if (_testImages.isEmpty) {
      diagnosis += '✗ No successful inference runs\n';
      diagnosis += '→ Check model file and permissions\n';
    } else {
      diagnosis += '✓ Inference successful with ${_testImages.length} configurations\n';
      
      // Check for common issues
      diagnosis += '\nLikely Issues:\n';
      diagnosis += '1. GPU Delegate: Often produces gray output on Android\n';
      diagnosis += '   → Solution: Use CPU-only execution\n';
      diagnosis += '2. Memory Alignment: Android ARM requires specific alignment\n';
      diagnosis += '   → Solution: Use explicit Float32List buffers\n';
      diagnosis += '3. Float Precision: Different handling on Android\n';
      diagnosis += '   → Solution: Ensure proper normalization\n';
    }
    
    return diagnosis;
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
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Text(
                  _status,
                  style: Theme.of(context).textTheme.headlineSmall,
                  textAlign: TextAlign.center,
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
                  child: SelectableText(
                    _debugInfo,
                    style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
                  ),
                ),
              const SizedBox(height: 20),
              if (_testImages.isNotEmpty)
                Column(
                  children: [
                    const Text(
                      'Test Results:',
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                    ),
                    const SizedBox(height: 10),
                    GridView.builder(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 2,
                        crossAxisSpacing: 10,
                        mainAxisSpacing: 10,
                      ),
                      itemCount: _testImages.length,
                      itemBuilder: (context, index) {
                        return Column(
                          children: [
                            Text(_testNames[index], style: const TextStyle(fontSize: 12)),
                            Expanded(
                              child: Container(
                                decoration: BoxDecoration(
                                  border: Border.all(color: Colors.grey),
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: Image.memory(
                                  _testImages[index],
                                  fit: BoxFit.contain,
                                ),
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                  ],
                ),
              const SizedBox(height: 20),
              Center(
                child: _isLoading
                  ? const CircularProgressIndicator()
                  : ElevatedButton.icon(
                      onPressed: _runAndroidDiagnosis,
                      icon: const Icon(Icons.android),
                      label: const Text('Run Android Diagnosis'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                      ),
                    ),
              ),
            ],
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