// Android Demo for StyleGAN2 Mobile
package com.example.stylegan2mobile

import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

class StyleGAN2Mobile(private val modelPath: String) {
    private lateinit var interpreter: Interpreter
    
    fun initialize() {
        val options = Interpreter.Options()
        options.setNumThreads(4)
        interpreter = Interpreter(loadModelFile(), options)
    }
    
    fun generateFace(): Bitmap {
        // Generate random latent vector
        val latent = FloatArray(512) { 
            Random.nextGaussian().toFloat() 
        }
        
        // Prepare input buffer
        val inputBuffer = ByteBuffer.allocateDirect(512 * 4)
            .order(ByteOrder.nativeOrder())
        inputBuffer.asFloatBuffer().put(latent)
        
        // Prepare output buffer (256x256x3)
        val outputBuffer = ByteBuffer.allocateDirect(256 * 256 * 3 * 4)
            .order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        // Convert to Bitmap
        return bufferToBitmap(outputBuffer)
    }
    
    private fun bufferToBitmap(buffer: ByteBuffer): Bitmap {
        val bitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(256 * 256)
        
        buffer.rewind()
        val floatBuffer = buffer.asFloatBuffer()
        
        for (i in pixels.indices) {
            val r = (floatBuffer.get() * 255).toInt().coerceIn(0, 255)
            val g = (floatBuffer.get() * 255).toInt().coerceIn(0, 255)
            val b = (floatBuffer.get() * 255).toInt().coerceIn(0, 255)
            pixels[i] = 0xFF shl 24 or (r shl 16) or (g shl 8) or b
        }
        
        bitmap.setPixels(pixels, 0, 256, 0, 0, 256, 256)
        return bitmap
    }
}

// Usage:
// val gan = StyleGAN2Mobile("stylegan2_mobile.tflite")
// gan.initialize()
// val generatedFace = gan.generateFace()
