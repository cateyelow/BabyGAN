import 'dart:typed_data';
import 'dart:math';
import 'package:tflite_flutter/tflite_flutter.dart';

class DebugHelper {
  /// Comprehensive debugging for TFLite gray output issue
  static Future<Map<String, dynamic>> diagnoseModel(Interpreter interpreter) async {
    final diagnostics = <String, dynamic>{};
    
    try {
      // 1. Model Information
      diagnostics['model_info'] = _getModelInfo(interpreter);
      
      // 2. Test with different input patterns
      diagnostics['input_tests'] = await _testInputPatterns(interpreter);
      
      // 3. Analyze output statistics
      diagnostics['output_analysis'] = await _analyzeOutputs(interpreter);
      
      // 4. Check tensor quantization
      diagnostics['quantization'] = _checkQuantization(interpreter);
      
      // 5. Memory layout analysis
      diagnostics['memory_layout'] = _analyzeMemoryLayout(interpreter);
      
    } catch (e) {
      diagnostics['error'] = e.toString();
    }
    
    return diagnostics;
  }
  
  static Map<String, dynamic> _getModelInfo(Interpreter interpreter) {
    final info = <String, dynamic>{};
    
    try {
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);
      
      info['input'] = {
        'shape': inputTensor.shape,
        'type': inputTensor.type,
        'name': inputTensor.name,
        'quantization_params': inputTensor.params,
      };
      
      info['output'] = {
        'shape': outputTensor.shape,
        'type': outputTensor.type,
        'name': outputTensor.name,
        'quantization_params': outputTensor.params,
      };
      
    } catch (e) {
      info['error'] = e.toString();
    }
    
    return info;
  }
  
  static Future<Map<String, dynamic>> _testInputPatterns(Interpreter interpreter) async {
    final results = <String, dynamic>{};
    
    // Test patterns
    final patterns = {
      'zeros': Float32List(512),
      'ones': Float32List.fromList(List.filled(512, 1.0)),
      'gaussian': _generateGaussian(512),
      'uniform': _generateUniform(512),
      'small_gaussian': _generateGaussian(512, scale: 0.1),
      'large_gaussian': _generateGaussian(512, scale: 2.0),
      'truncated_gaussian': _generateTruncatedGaussian(512),
    };
    
    for (final entry in patterns.entries) {
      final output = await _runInference(interpreter, entry.value);
      results[entry.key] = _analyzeOutput(output);
    }
    
    return results;
  }
  
  static Future<Float32List> _runInference(Interpreter interpreter, Float32List input) async {
    final outputShape = interpreter.getOutputTensor(0).shape;
    final outputSize = outputShape.reduce((a, b) => a * b);
    final output = Float32List(outputSize);
    
    final inputs = {0: input};
    final outputs = {0: output};
    
    interpreter.runForMultipleInputs(inputs, outputs);
    
    return output;
  }
  
  static Map<String, dynamic> _analyzeOutput(Float32List output) {
    final stats = <String, dynamic>{};
    
    // Basic statistics
    double min = double.infinity;
    double max = double.negativeInfinity;
    double sum = 0;
    double sumSquared = 0;
    
    // Value distribution
    int zeros = 0;
    int ones = 0;
    int negatives = 0;
    int inRange = 0; // values in [0, 1]
    
    for (final value in output) {
      min = value < min ? value : min;
      max = value > max ? value : max;
      sum += value;
      sumSquared += value * value;
      
      if (value == 0) zeros++;
      if (value == 1) ones++;
      if (value < 0) negatives++;
      if (value >= 0 && value <= 1) inRange++;
    }
    
    final mean = sum / output.length;
    final variance = (sumSquared / output.length) - (mean * mean);
    final stdDev = sqrt(variance.abs());
    
    // Check for common patterns
    final allSame = output.every((v) => v == output[0]);
    final mostlyGray = _checkMostlyGray(output);
    
    stats['min'] = min;
    stats['max'] = max;
    stats['mean'] = mean;
    stats['std_dev'] = stdDev;
    stats['zeros_count'] = zeros;
    stats['ones_count'] = ones;
    stats['negatives_count'] = negatives;
    stats['in_range_count'] = inRange;
    stats['all_same'] = allSame;
    stats['mostly_gray'] = mostlyGray;
    stats['sample_values'] = output.take(10).toList();
    
    return stats;
  }
  
  static bool _checkMostlyGray(Float32List output) {
    // Check if output represents a gray image
    // Assuming NHWC format: [1, 256, 256, 3]
    const tolerance = 0.1;
    int grayPixels = 0;
    
    for (int i = 0; i < output.length; i += 3) {
      if (i + 2 < output.length) {
        final r = output[i];
        final g = output[i + 1];
        final b = output[i + 2];
        
        // Check if RGB values are similar (gray)
        if ((r - g).abs() < tolerance && 
            (g - b).abs() < tolerance && 
            (r - b).abs() < tolerance) {
          grayPixels++;
        }
      }
    }
    
    final totalPixels = output.length ~/ 3;
    return grayPixels > totalPixels * 0.9; // 90% gray pixels
  }
  
  static Future<Map<String, dynamic>> _analyzeOutputs(Interpreter interpreter) async {
    final analysis = <String, dynamic>{};
    
    // Run multiple inferences
    const numTests = 5;
    final outputs = <Float32List>[];
    
    for (int i = 0; i < numTests; i++) {
      final input = _generateGaussian(512);
      final output = await _runInference(interpreter, input);
      outputs.add(output);
    }
    
    // Check consistency
    analysis['consistent'] = _checkConsistency(outputs);
    
    // Check variation
    analysis['variation'] = _checkVariation(outputs);
    
    return analysis;
  }
  
  static bool _checkConsistency(List<Float32List> outputs) {
    // Check if outputs are suspiciously similar
    if (outputs.length < 2) return true;
    
    final first = outputs[0];
    for (int i = 1; i < outputs.length; i++) {
      double diff = 0;
      for (int j = 0; j < first.length; j++) {
        diff += (first[j] - outputs[i][j]).abs();
      }
      if (diff > 0.01 * first.length) {
        return false; // Outputs vary enough
      }
    }
    return true; // Outputs too similar
  }
  
  static double _checkVariation(List<Float32List> outputs) {
    // Calculate average pixel-wise variation
    if (outputs.isEmpty) return 0;
    
    final pixelVariances = List<double>.filled(outputs[0].length, 0);
    
    // Calculate mean for each pixel
    final pixelMeans = List<double>.filled(outputs[0].length, 0);
    for (final output in outputs) {
      for (int i = 0; i < output.length; i++) {
        pixelMeans[i] += output[i];
      }
    }
    for (int i = 0; i < pixelMeans.length; i++) {
      pixelMeans[i] /= outputs.length;
    }
    
    // Calculate variance for each pixel
    for (final output in outputs) {
      for (int i = 0; i < output.length; i++) {
        final diff = output[i] - pixelMeans[i];
        pixelVariances[i] += diff * diff;
      }
    }
    
    double totalVariance = 0;
    for (final variance in pixelVariances) {
      totalVariance += variance / outputs.length;
    }
    
    return totalVariance / pixelVariances.length;
  }
  
  static Map<String, dynamic> _checkQuantization(Interpreter interpreter) {
    final quantInfo = <String, dynamic>{};
    
    try {
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);
      
      quantInfo['input_quantized'] = inputTensor.params != null;
      quantInfo['output_quantized'] = outputTensor.params != null;
      
      if (inputTensor.params != null) {
        quantInfo['input_scale'] = inputTensor.params!['scale'];
        quantInfo['input_zero_point'] = inputTensor.params!['zero_point'];
      }
      
      if (outputTensor.params != null) {
        quantInfo['output_scale'] = outputTensor.params!['scale'];
        quantInfo['output_zero_point'] = outputTensor.params!['zero_point'];
      }
      
    } catch (e) {
      quantInfo['error'] = e.toString();
    }
    
    return quantInfo;
  }
  
  static Map<String, dynamic> _analyzeMemoryLayout(Interpreter interpreter) {
    final layout = <String, dynamic>{};
    
    try {
      final outputShape = interpreter.getOutputTensor(0).shape;
      
      // Expected: [1, 256, 256, 3]
      layout['expected_shape'] = [1, 256, 256, 3];
      layout['actual_shape'] = outputShape;
      layout['shape_matches'] = outputShape.toString() == [1, 256, 256, 3].toString();
      
      // Calculate strides
      if (outputShape.length == 4) {
        layout['channel_stride'] = 1;
        layout['width_stride'] = outputShape[3];
        layout['height_stride'] = outputShape[2] * outputShape[3];
        layout['batch_stride'] = outputShape[1] * outputShape[2] * outputShape[3];
      }
      
    } catch (e) {
      layout['error'] = e.toString();
    }
    
    return layout;
  }
  
  // Helper functions for generating different input patterns
  static Float32List _generateGaussian(int size, {double scale = 1.0}) {
    final random = Random();
    return Float32List.fromList(
      List.generate(size, (_) => _nextGaussian(random) * scale),
    );
  }
  
  static Float32List _generateUniform(int size, {double min = -1, double max = 1}) {
    final random = Random();
    final range = max - min;
    return Float32List.fromList(
      List.generate(size, (_) => random.nextDouble() * range + min),
    );
  }
  
  static Float32List _generateTruncatedGaussian(int size, {double limit = 2.0}) {
    final random = Random();
    return Float32List.fromList(
      List.generate(size, (_) {
        double value;
        do {
          value = _nextGaussian(random);
        } while (value.abs() > limit);
        return value;
      }),
    );
  }
  
  static double _nextGaussian(Random random) {
    // Box-Muller transform
    double u1 = 0.0;
    double u2 = 0.0;
    while (u1 == 0.0) u1 = random.nextDouble();
    u2 = random.nextDouble();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
  }
  
  /// Generate a diagnostic report
  static String generateReport(Map<String, dynamic> diagnostics) {
    final buffer = StringBuffer();
    
    buffer.writeln('=== BabyGAN TFLite Diagnostic Report ===\n');
    
    // Model Information
    buffer.writeln('1. MODEL INFORMATION:');
    final modelInfo = diagnostics['model_info'] as Map<String, dynamic>?;
    if (modelInfo != null) {
      buffer.writeln('   Input: ${modelInfo['input']}');
      buffer.writeln('   Output: ${modelInfo['output']}');
    }
    
    // Input Pattern Tests
    buffer.writeln('\n2. INPUT PATTERN TESTS:');
    final inputTests = diagnostics['input_tests'] as Map<String, dynamic>?;
    if (inputTests != null) {
      for (final entry in inputTests.entries) {
        buffer.writeln('   ${entry.key}: ${entry.value}');
      }
    }
    
    // Output Analysis
    buffer.writeln('\n3. OUTPUT ANALYSIS:');
    final outputAnalysis = diagnostics['output_analysis'] as Map<String, dynamic>?;
    if (outputAnalysis != null) {
      buffer.writeln('   Consistent outputs: ${outputAnalysis['consistent']}');
      buffer.writeln('   Variation: ${outputAnalysis['variation']}');
    }
    
    // Quantization
    buffer.writeln('\n4. QUANTIZATION:');
    final quantization = diagnostics['quantization'] as Map<String, dynamic>?;
    if (quantization != null) {
      buffer.writeln('   ${quantization}');
    }
    
    // Memory Layout
    buffer.writeln('\n5. MEMORY LAYOUT:');
    final memoryLayout = diagnostics['memory_layout'] as Map<String, dynamic>?;
    if (memoryLayout != null) {
      buffer.writeln('   Shape matches expected: ${memoryLayout['shape_matches']}');
      buffer.writeln('   Layout: ${memoryLayout}');
    }
    
    // Diagnosis
    buffer.writeln('\n6. DIAGNOSIS:');
    buffer.writeln(_diagnoseIssue(diagnostics));
    
    return buffer.toString();
  }
  
  static String _diagnoseIssue(Map<String, dynamic> diagnostics) {
    final issues = <String>[];
    
    // Check input tests
    final inputTests = diagnostics['input_tests'] as Map<String, dynamic>?;
    if (inputTests != null) {
      bool allGray = true;
      bool allSameRange = true;
      double? firstMean;
      
      for (final entry in inputTests.entries) {
        final stats = entry.value as Map<String, dynamic>;
        if (stats['mostly_gray'] == false) allGray = false;
        
        if (firstMean == null) {
          firstMean = stats['mean'] as double;
        } else {
          if ((stats['mean'] as double - firstMean).abs() > 0.1) {
            allSameRange = false;
          }
        }
      }
      
      if (allGray) {
        issues.add('All outputs are gray regardless of input - likely model weights issue');
      }
      
      if (allSameRange) {
        issues.add('Output mean is constant - model may not be processing input');
      }
    }
    
    // Check output consistency
    final outputAnalysis = diagnostics['output_analysis'] as Map<String, dynamic>?;
    if (outputAnalysis != null && outputAnalysis['consistent'] == true) {
      issues.add('Outputs are too consistent - model may be stuck');
    }
    
    // Check quantization
    final quantization = diagnostics['quantization'] as Map<String, dynamic>?;
    if (quantization != null && quantization['output_quantized'] == true) {
      issues.add('Output is quantized - may need dequantization');
    }
    
    if (issues.isEmpty) {
      return 'No obvious issues detected. Check model conversion process.';
    }
    
    return issues.join('\n   ');
  }
}

// Extension to use in main app
extension InterpreterDiagnostics on Interpreter {
  Future<String> runDiagnostics() async {
    final diagnostics = await DebugHelper.diagnoseModel(this);
    return DebugHelper.generateReport(diagnostics);
  }
}