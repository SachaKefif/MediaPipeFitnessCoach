package com.example.fitnesscoach

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.fitnesscoach.databinding.ActivityMainBinding
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var poseLandmarker: PoseLandmarker? = null
    private var ortSession: OrtSession? = null
    private val ortEnv = OrtEnvironment.getEnvironment()

    private val poseProcessor = PoseProcessor()
    private val frameWindow: Deque<FloatArray> = ArrayDeque(60)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
            setupModels()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun setupModels() {
        // Setup MediaPipe
        try {
            val baseOptionsBuilder = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker.task")
            
            val optionsBuilder = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener(this::returnResult)
                .setErrorListener(this::errorListener)
            
            val options = optionsBuilder.build()
            poseLandmarker = PoseLandmarker.createFromOptions(this, options)
        } catch (e: Exception) {
            Log.e(TAG, "MediaPipe Error: ${e.message}")
            Toast.makeText(this, "Place pose_landmarker.task in assets!", Toast.LENGTH_LONG).show()
        }

        // Setup ONNX Runtime
        try {
            val modelBytes = assets.open("fitness_coach_onnx.onnx").readBytes()
            ortSession = ortEnv.createSession(modelBytes)
        } catch (e: Exception) {
            Log.e(TAG, "ONNX Error: ${e.message}")
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        val mpImage = BitmapImageBuilder(bitmap).build()
        
        poseLandmarker?.detectAsync(mpImage, System.currentTimeMillis())
        
        imageProxy.close()
    }

    private fun returnResult(result: PoseLandmarkerResult, input: com.google.mediapipe.framework.image.MPImage) {
        val features = poseProcessor.cleanData(result, input.width, input.height)
        
        if (features != null) {
            synchronized(frameWindow) {
                if (frameWindow.size >= 60) {
                    frameWindow.removeFirst()
                }
                frameWindow.addLast(features)

                if (frameWindow.size == 60) {
                    runInference()
                }
            }
        } else {
            synchronized(frameWindow) {
                frameWindow.clear()
            }
            runOnUiThread {
                viewBinding.predictionText.text = "NO BODY DETECTED"
                viewBinding.predictionText.setTextColor(ContextCompat.getColor(this, android.R.color.holo_orange_dark))
            }
        }
    }

    private fun runInference() {
        val session = ortSession ?: return
        
        // Flatten the window: 60 frames * 70 features = 4200 features
        val inputData = FloatArray(60 * 70)
        var offset = 0
        synchronized(frameWindow) {
            for (frame in frameWindow) {
                System.arraycopy(frame, 0, inputData, offset, 70)
                offset += 70
            }
        }

        // Reshape to [1, 4200]
        val inputShape = longArrayOf(1, 4200)
        val inputTensor = OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(inputData), inputShape)
        
        try {
            val results = session.run(Collections.singletonMap("input", inputTensor)) // Check your model's input name!
            val output = results[0].value as Array<FloatArray> // Assuming output is [1, num_classes] probabilities
            
            // Labels: 0 = Nothing, 1 = Pushup, 2 = Squat
            val probabilities = output[0]
            val maxIdx = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[maxIdx]

            runOnUiThread {
                updateUI(maxIdx, confidence)
            }
            
            results.close()
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
        } finally {
            inputTensor.close()
        }
    }

    private fun updateUI(label: Int, confidence: Float) {
        val text = when (label) {
            1 -> "PUSHUP (${(confidence * 100).toInt()}%)"
            2 -> "SQUAT (${(confidence * 100).toInt()}%)"
            else -> "IDLE"
        }
        
        val color = if (label > 0) android.R.color.holo_green_dark else android.R.color.holo_red_dark
        
        viewBinding.predictionText.text = text
        viewBinding.predictionText.setTextColor(ContextCompat.getColor(this, color))
    }

    private fun errorListener(e: Exception) {
        Log.e(TAG, "MediaPipe Error Listener: ${e.message}")
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        poseLandmarker?.close()
        ortSession?.close()
        ortEnv.close()
    }

    companion object {
        private const val TAG = "FitnessCoach"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
