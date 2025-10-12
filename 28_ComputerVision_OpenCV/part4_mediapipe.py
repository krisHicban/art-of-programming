import cv2
import mediapipe as mp
import numpy as np

# ====== HEALTH APP: Workout Form Analyzer ======
class WorkoutFormAnalyzer:
    """Real-time workout form analysis using MediaPipe"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.squat_count = 0
        self.prev_state = None

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points (landmarks)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def analyze_squat_form(self, landmarks):
        """Analyze squat form quality"""
        # Get key landmarks
        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)

        # Detect squat phases
        if knee_angle > 160:
            state = 'up'
        elif knee_angle < 90:
            state = 'down'
        else:
            state = 'middle'

        # Count rep
        if self.prev_state == 'down' and state == 'up':
            self.squat_count += 1

        self.prev_state = state

        # Form feedback
        feedback = []
        if state == 'down':
            if knee_angle > 100:
                feedback.append("⚠️ Go deeper!")
            else:
                feedback.append("✅ Good depth")

            if left_knee[0] > left_ankle[0]:
                feedback.append("⚠️ Knees too far forward")
            else:
                feedback.append("✅ Good knee position")

        return {
            'count': self.squat_count,
            'knee_angle': knee_angle,
            'state': state,
            'feedback': feedback
        }

# ====== HAND TRACKING: Finance Receipt Scanner ======
class ReceiptScanner:
    """Use hand tracking to guide receipt photo capture"""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )

    def detect_pinch_gesture(self, landmarks):
        """Detect pinch gesture (thumb + index finger close)"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        )

        return distance < 0.05

print("\n🚀 WHY MEDIAPIPE IS REVOLUTIONARY")
print("   Building pose estimation from scratch:")
print("     - Requires PhD-level ML expertise")
print("     - Years of training data collection")
print("     - Millions in compute resources")
print("   Using MediaPipe:")
print("     - 5 lines of code")
print("     - Works on ANY device")
print("     - Real-time performance (30+ FPS)")
print("\n💡 This is the POWER of standing on giants' shoulders!")