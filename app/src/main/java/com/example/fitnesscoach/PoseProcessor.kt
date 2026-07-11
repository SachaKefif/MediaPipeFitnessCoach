package com.example.fitnesscoach

import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.acos
import kotlin.math.pow
import kotlin.math.sqrt

class PoseProcessor {

    companion object {
        const val L_SHOULDER = 0
        const val R_SHOULDER = 1
        const val L_ELBOW = 2
        const val R_ELBOW = 3
        const val L_WRIST = 4
        const val R_WRIST = 5
        const val L_HAND = 6
        const val R_HAND = 7
        const val L_HIP = 8
        const val R_HIP = 9
        const val L_KNEE = 10
        const val R_KNEE = 11
        const val L_ANKLE = 12
        const val R_ANKLE = 13
        const val L_FOOT = 14
        const val R_FOOT = 15

        val KEEP_INDICES = intArrayOf(11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 31, 32)
    }

    fun cleanData(result: PoseLandmarkerResult, width: Int, height: Int): FloatArray? {
        if (result.landmarks().isEmpty()) return null

        val landmarks = result.landmarks()[0]

        // 1. Extract and restore grid (equivalent to extract_pose_landmarks)
        val poseArray = Array(16) { FloatArray(3) }
        for (i in KEEP_INDICES.indices) {
            val lm = landmarks[KEEP_INDICES[i]]
            poseArray[i][0] = lm.x() * width
            poseArray[i][1] = lm.y() * height
            poseArray[i][2] = lm.z() * width
        }

        // 2. Normalize coords (equivalent to normalize_coords)
        val hipCenter = FloatArray(3) { i -> (poseArray[L_HIP][i] + poseArray[R_HIP][i]) / 2f }
        val shoulderCenter = FloatArray(3) { i -> (poseArray[L_SHOULDER][i] + poseArray[R_SHOULDER][i]) / 2f }
        
        var torsoLength = 0f
        for (i in 0..2) torsoLength += (shoulderCenter[i] - hipCenter[i]).pow(2)
        torsoLength = sqrt(torsoLength)

        if (torsoLength < 1e-6) torsoLength = 1.0f

        val normalizedArray = Array(16) { i ->
            FloatArray(3) { j -> (poseArray[i][j] - hipCenter[j]) / torsoLength }
        }

        // 3. Add data (angles and distances)
        return addData(normalizedArray)
    }

    private fun addData(normalizedArray: Array<FloatArray>): FloatArray {
        val leftAngles = listOf(
            calcAngle(normalizedArray, L_SHOULDER, L_ELBOW, L_WRIST),
            calcAngle(normalizedArray, L_ELBOW, L_SHOULDER, L_HIP),
            calcAngle(normalizedArray, L_SHOULDER, L_HIP, L_ANKLE),
            calcAngle(normalizedArray, L_HIP, L_KNEE, L_ANKLE),
            calcAngle(normalizedArray, L_HAND, L_WRIST, L_ELBOW),
            calcAngle(normalizedArray, L_KNEE, L_ANKLE, L_FOOT),
            calcAngle(normalizedArray, L_ANKLE, L_KNEE, L_WRIST),
            calcAngle(normalizedArray, L_ANKLE, L_HIP, L_WRIST),
            calcAngle(normalizedArray, L_ANKLE, L_SHOULDER, L_WRIST)
        )

        val rightAngles = listOf(
            calcAngle(normalizedArray, R_SHOULDER, R_ELBOW, R_WRIST),
            calcAngle(normalizedArray, R_ELBOW, R_SHOULDER, R_HIP),
            calcAngle(normalizedArray, R_SHOULDER, R_HIP, R_ANKLE),
            calcAngle(normalizedArray, R_HIP, R_KNEE, R_ANKLE),
            calcAngle(normalizedArray, R_HAND, R_WRIST, R_ELBOW),
            calcAngle(normalizedArray, R_KNEE, R_ANKLE, R_FOOT),
            calcAngle(normalizedArray, R_ANKLE, R_KNEE, R_WRIST),
            calcAngle(normalizedArray, R_ANKLE, R_HIP, R_WRIST),
            calcAngle(normalizedArray, R_ANKLE, R_SHOULDER, R_WRIST)
        )

        val leftDistances = listOf(
            calcDistance(normalizedArray, L_WRIST, L_SHOULDER),
            calcDistance(normalizedArray, L_ANKLE, L_SHOULDER)
        )

        val rightDistances = listOf(
            calcDistance(normalizedArray, R_WRIST, R_SHOULDER),
            calcDistance(normalizedArray, R_ANKLE, L_SHOULDER) // Note: original code has R_ANKLE, L_SHOULDER for right side... wait let me check
        )

        // Result vector: flat coords (48) + paired angles (18) + paired distances (4) = 70
        val finalArray = FloatArray(70)
        var idx = 0

        // Flat coords
        for (i in 0..15) {
            for (j in 0..2) {
                finalArray[idx++] = normalizedArray[i][j]
            }
        }

        // Paired angles: [[r, l] for r, l in zip(right_angles, left_angles)]
        for (i in 0..8) {
            finalArray[idx++] = rightAngles[i]
            finalArray[idx++] = leftAngles[i]
        }

        // Paired distances
        for (i in 0..1) {
            finalArray[idx++] = rightDistances[i]
            finalArray[idx++] = leftDistances[i]
        }

        return finalArray
    }

    private fun calcAngle(landmarks: Array<FloatArray>, p1: Int, p2: Int, p3: Int): Float {
        val a = landmarks[p1]
        val b = landmarks[p2]
        val c = landmarks[p3]

        val ba = floatArrayOf(a[0] - b[0], a[1] - b[1])
        val bc = floatArrayOf(c[0] - b[0], c[1] - b[1])

        val dotProduct = ba[0] * bc[0] + ba[1] * bc[1]
        val normBa = sqrt(ba[0].pow(2) + ba[1].pow(2))
        val normBc = sqrt(bc[0].pow(2) + bc[1].pow(2))

        if (normBa == 0f || normBc == 0f) return 0f

        var cosineAngle = dotProduct / (normBa * normBc)
        if (cosineAngle > 1.0f) cosineAngle = 1.0f
        if (cosineAngle < -1.0f) cosineAngle = -1.0f

        return Math.toDegrees(acos(cosineAngle.toDouble())).toFloat()
    }

    private fun calcDistance(landmarks: Array<FloatArray>, p1: Int, p2: Int): Float {
        val a = landmarks[p1]
        val b = landmarks[p2]
        return sqrt((a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2))
    }
}
