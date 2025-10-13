import cv2
import mediapipe as mp
import numpy as np
import time
import platform

# ====== INTERACTIVE HAND TRACKING LAB ======
# Learn hand tracking with REAL webcam feedback!

class HandTrackingLab:
    """
    Interactive hand tracking system using MediaPipe.
    Works on Mac, Windows, and Linux!
    """
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand detector with optimized settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture tracking
        self.pinch_detected = False
        self.gesture_history = []
        self.last_gesture_time = time.time()
        
        # Statistics
        self.frame_count = 0
        self.hands_detected_count = 0
        
        print(f"\n🖐️  MediaPipe Hand Tracking initialized!")
        print(f"💻 Platform: {platform.system()} {platform.release()}")
    
    def get_finger_status(self, hand_landmarks, handedness):
        """
        Determine which fingers are extended (up).
        Returns: [thumb, index, middle, ring, pinky] as booleans
        """
        # Landmark indices for finger tips and joints
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # Joints below tips
        
        fingers_up = []
        
        # Thumb (special case - horizontal comparison)
        if handedness == "Right":
            fingers_up.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
        else:
            fingers_up.append(hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x)
        
        # Other fingers (vertical comparison)
        for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
            fingers_up.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
        
        return fingers_up
    
    def calculate_distance(self, landmark1, landmark2):
        """Calculate Euclidean distance between two landmarks"""
        return np.sqrt(
            (landmark1.x - landmark2.x)**2 +
            (landmark1.y - landmark2.y)**2 +
            (landmark1.z - landmark2.z)**2
        )
    
    def detect_gestures(self, hand_landmarks, handedness):
        """
        Detect various hand gestures.
        Returns: gesture name and confidence
        """
        fingers = self.get_finger_status(hand_landmarks, handedness)
        
        # Get landmark positions
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        wrist = hand_landmarks.landmark[0]
        
        # Calculate distances
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        
        # GESTURE DETECTION LOGIC
        
        # 1. PINCH (Thumb + Index close)
        if pinch_distance < 0.05:
            return "PINCH 🤏", 0.95
        
        # 2. PEACE SIGN (Index + Middle up, others down)
        if fingers == [False, True, True, False, False]:
            return "PEACE ✌️", 0.90
        
        # 3. THUMBS UP (Only thumb up)
        if fingers == [True, False, False, False, False]:
            return "THUMBS UP 👍", 0.90
        
        # 4. ROCK ON (Index + Pinky up)
        if fingers == [False, True, False, False, True]:
            return "ROCK ON 🤘", 0.90
        
        # 5. POINTING (Only index up)
        if fingers == [False, True, False, False, False]:
            return "POINTING ☝️", 0.85
        
        # 6. OPEN PALM (All fingers up)
        if all(fingers):
            return "OPEN PALM 🖐️", 0.85
        
        # 7. FIST (All fingers down)
        if not any(fingers):
            return "FIST ✊", 0.85
        
        # 8. OK SIGN (Thumb + Index circle, others up)
        if pinch_distance < 0.08 and fingers[1] == False and fingers[2:] == [True, True, True]:
            return "OK SIGN 👌", 0.80
        
        return "UNKNOWN", 0.0
    
    def draw_hand_info(self, frame, hand_landmarks, handedness, hand_index):
        """Draw detailed information about detected hand"""
        h, w, _ = frame.shape
        
        # Draw hand skeleton
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Get key landmarks
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        wrist = hand_landmarks.landmark[0]
        
        # Convert to pixel coordinates
        thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_px = (int(index_tip.x * w), int(index_tip.y * h))
        wrist_px = (int(wrist.x * w), int(wrist.y * h))
        
        # Draw pinch line
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        color = (0, 255, 0) if pinch_distance < 0.05 else (255, 255, 255)
        cv2.line(frame, thumb_px, index_px, color, 2)
        
        # Detect gesture
        gesture, confidence = self.detect_gestures(hand_landmarks, handedness)
        
        # Draw info box
        info_y = 30 + (hand_index * 120)
        cv2.rectangle(frame, (10, info_y), (350, info_y + 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, info_y), (350, info_y + 110), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Hand {hand_index + 1}: {handedness}", 
                   (20, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Gesture: {gesture}", 
                   (20, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (20, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Pinch Distance: {pinch_distance:.3f}", 
                   (20, info_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Get finger status
        fingers = self.get_finger_status(hand_landmarks, handedness)
        finger_names = ['👍', '☝️', '🖕', '💍', '🤙']
        finger_status = ' '.join([name if up else '❌' for name, up in zip(finger_names, fingers)])
        
        return gesture, finger_status
    
    def run_webcam_demo(self):
        """
        Run interactive webcam demo.
        Shows real-time hand tracking with gesture recognition!
        """
        print("\n" + "="*70)
        print("🎥 STARTING WEBCAM HAND TRACKING")
        print("="*70)
        print("\n📋 Instructions:")
        print("   • Position your hand(s) in front of camera")
        print("   • Try different gestures:")
        print("       - Open palm 🖐️")
        print("       - Peace sign ✌️")
        print("       - Pinch 🤏 (thumb + index)")
        print("       - Thumbs up 👍")
        print("       - Rock on 🤘")
        print("       - Fist ✊")
        print("\n⌨️  Controls:")
        print("   • Press 'q' to quit")
        print("   • Press 's' to save screenshot")
        print("   • Press 'r' to reset statistics")
        print("="*70 + "\n")
        
        # Try different camera indices for cross-platform compatibility
        camera_indices = [0, 1, 2]  # Try multiple cameras
        cap = None
        
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"✅ Camera {idx} opened successfully!")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            print("❌ Error: Could not open any camera!")
            print("\n💡 Troubleshooting:")
            print("   Mac: Check System Preferences > Security & Privacy > Camera")
            print("   Windows: Check Settings > Privacy > Camera")
            print("   Linux: Ensure user is in 'video' group")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_time = time.time()
        fps = 0
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("⚠️  Failed to grab frame")
                    break
                
                self.frame_count += 1
                
                # Flip frame for selfie view
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = self.hands.process(rgb_frame)
                
                # Draw UI background
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
                
                # Calculate FPS
                current_time = time.time()
                if current_time - fps_time > 0:
                    fps = 1.0 / (current_time - fps_time)
                fps_time = current_time
                
                # Draw FPS and stats
                cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, f"Frame: {self.frame_count}", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Process hand landmarks
                if results.multi_hand_landmarks and results.multi_handedness:
                    self.hands_detected_count += 1
                    
                    for hand_idx, (hand_landmarks, handedness_info) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)
                    ):
                        handedness = handedness_info.classification[0].label
                        
                        # Draw hand info and get gesture
                        gesture, finger_status = self.draw_hand_info(
                            frame, hand_landmarks, handedness, hand_idx
                        )
                        
                        # Log interesting gestures
                        if gesture != "UNKNOWN" and time.time() - self.last_gesture_time > 1.0:
                            print(f"   🎯 {gesture} detected! Fingers: {finger_status}")
                            self.last_gesture_time = time.time()
                
                else:
                    cv2.putText(frame, "No hands detected - Show your hand!", 
                               (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 0, 255), 2)
                
                # Show instructions
                cv2.putText(frame, "Press 'q' to quit | 's' to screenshot", 
                           (w - 400, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Hand Tracking Lab - MediaPipe', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Exiting...")
                    break
                elif key == ord('s'):
                    filename = f'hand_tracking_screenshot_{int(time.time())}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.frame_count = 0
                    self.hands_detected_count = 0
                    print("\n🔄 Statistics reset!")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            print("\n" + "="*70)
            print("📊 SESSION STATISTICS")
            print("="*70)
            print(f"   Total frames processed: {self.frame_count}")
            print(f"   Frames with hands detected: {self.hands_detected_count}")
            if self.frame_count > 0:
                detection_rate = (self.hands_detected_count / self.frame_count) * 100
                print(f"   Detection rate: {detection_rate:.1f}%")
            print("="*70 + "\n")


def process_image_file(image_path):
    """
    Process a single image file (if you don't have a webcam).
    Perfect for testing with saved photos!
    """
    print(f"\n📸 Processing image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        
        # Process image
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)
        
        # Draw results
        if results.multi_hand_landmarks:
            print(f"✅ Detected {len(results.multi_hand_landmarks)} hand(s)!")
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Print landmark coordinates
                print("\n📍 Hand Landmarks (sample):")
                landmarks_to_show = [0, 4, 8, 12, 16, 20]  # Wrist, Thumb, Index, Middle, Ring, Pinky tips
                landmark_names = ['Wrist', 'Thumb tip', 'Index tip', 'Middle tip', 'Ring tip', 'Pinky tip']
                
                for idx, name in zip(landmarks_to_show, landmark_names):
                    lm = hand_landmarks.landmark[idx]
                    print(f"   {name}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
        
        else:
            print("❌ No hands detected in image")
        
        # Save and show result
        output_path = 'hand_detection_result.jpg'
        cv2.imwrite(output_path, img)
        print(f"\n✅ Result saved to: {output_path}")
        
        # Display
        cv2.imshow('Hand Detection Result - Press any key to close', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ====== MAIN USAGE ======
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🖐️  MEDIAPIPE HAND TRACKING LAB")
    print("="*70)
    print("\n📚 What you'll learn:")
    print("   • Real-time hand landmark detection (21 points per hand)")
    print("   • Gesture recognition algorithms")
    print("   • Distance calculations in 3D space")
    print("   • Computer vision performance optimization")
    
    print("\n" + "="*70)
    print("🎯 USAGE OPTIONS:")
    print("="*70)
    
    print("\n1️⃣  Webcam mode (RECOMMENDED):")
    print("    lab = HandTrackingLab()")
    print("    lab.run_webcam_demo()")
    
    print("\n2️⃣  Process image file (no webcam needed):")
    print("    process_image_file('hand_photo.jpg')")
    
    print("\n" + "="*70)
    print("🎮 TRY THESE GESTURES:")
    print("="*70)
    print("   🖐️  Open Palm - All fingers extended")
    print("   ✌️  Peace Sign - Index + Middle up")
    print("   👍 Thumbs Up - Only thumb extended")
    print("   🤏 Pinch - Thumb + Index together")
    print("   ✊ Fist - All fingers closed")
    print("   🤘 Rock On - Index + Pinky up")
    print("   ☝️  Pointing - Only index up")
    
    print("\n" + "="*70)
    print("💡 TECHNICAL DETAILS:")
    print("="*70)
    print("   • 21 landmarks per hand (wrist + 4 fingers × 5 points)")
    print("   • 3D coordinates (x, y, z) normalized to [0, 1]")
    print("   • Real-time processing: ~30 FPS on modern hardware")
    print("   • Works with: Mac, Windows, Linux")
    
    print("\n" + "="*70)
    print("\n▶️  Ready to start? Run:")
    print("    lab = HandTrackingLab()")
    print("    lab.run_webcam_demo()")
    print("="*70 + "\n")
    
    # Uncomment to auto-start webcam demo:
    lab = HandTrackingLab()
    lab.run_webcam_demo()