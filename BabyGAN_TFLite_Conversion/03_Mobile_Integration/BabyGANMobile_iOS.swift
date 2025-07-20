// iOS Solution for StyleGAN Mobile
import UIKit
import TensorFlowLite

class BabyGANMobile {
    private var interpreter: Interpreter?
    private let inputSize = 512
    private let outputWidth = 256
    private let outputHeight = 256
    
    init() {
        setupInterpreter()
    }
    
    private func setupInterpreter() {
        guard let modelPath = Bundle.main.path(forResource: "stylegan_mobile_working", 
                                               ofType: "tflite") else {
            print("Failed to load model")
            return
        }
        
        do {
            var options = Interpreter.Options()
            options.threadCount = 4
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            try interpreter?.allocateTensors()
        } catch {
            print("Failed to create interpreter: \(error)")
        }
    }
    
    func generateBabyFace(parent1Latent: [Float]? = nil,
                          parent2Latent: [Float]? = nil,
                          mixRatio: Float = 0.5) -> UIImage? {
        
        guard let interpreter = interpreter else { return nil }
        
        // Create or mix latent vectors
        var latent: [Float]
        if let p1 = parent1Latent, let p2 = parent2Latent {
            // Mix parents
            latent = (0..<inputSize).map { i in
                p1[i] * (1 - mixRatio) + p2[i] * mixRatio
            }
        } else {
            // Random latent
            latent = (0..<inputSize).map { _ in Float.random(in: -2...2) }
        }
        
        // Prepare input
        let inputData = latent.withUnsafeBufferPointer { Data(buffer: $0) }
        
        do {
            // Set input
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference
            try interpreter.invoke()
            
            // Get output
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            
            // Convert to UIImage
            return createImage(from: outputData)
            
        } catch {
            print("Inference failed: \(error)")
            return nil
        }
    }
    
    private func createImage(from data: Data) -> UIImage? {
        let width = outputWidth
        let height = outputHeight
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            let floatPointer = bytes.bindMemory(to: Float.self)
            
            for y in 0..<height {
                for x in 0..<width {
                    let idx = (y * width + x) * 3
                    let pixelIdx = (y * width + x) * 4
                    
                    pixelData[pixelIdx] = UInt8(floatPointer[idx] * 255)     // R
                    pixelData[pixelIdx + 1] = UInt8(floatPointer[idx + 1] * 255) // G
                    pixelData[pixelIdx + 2] = UInt8(floatPointer[idx + 2] * 255) // B
                    pixelData[pixelIdx + 3] = 255 // A
                }
            }
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let provider = CGDataProvider(data: NSData(bytes: pixelData, length: pixelData.count)) else {
            return nil
        }
        
        guard let cgImage = CGImage(width: width,
                                   height: height,
                                   bitsPerComponent: 8,
                                   bitsPerPixel: 32,
                                   bytesPerRow: bytesPerRow,
                                   space: colorSpace,
                                   bitmapInfo: bitmapInfo,
                                   provider: provider,
                                   decode: nil,
                                   shouldInterpolate: true,
                                   intent: .defaultIntent) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
}
