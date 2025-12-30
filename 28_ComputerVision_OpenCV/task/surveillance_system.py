#!/usr/bin/env python3
"""
🚨 SURVEILLANCE & FIRE SAFETY SYSTEM 🚨
Real-time motion detection and smoke detection using OpenCV

Features:
- Motion detection via frame differencing
- Smoke detection via HSV color analysis
- Visual alarms with red borders
- Automatic snapshot saving
- Console logging with timestamps
- Multi-window display

Press 'q' to quit
Press 's' to save manual snapshot
Press 'r' to reset alarms
"""

import cv2
import numpy as np
from datetime import datetime
import os
from collections import deque

# ==================== CONFIGURATION ====================

# Motion Detection Settings
MOTION_THRESHOLD = 2000  # Total pixels changed to trigger motion alarm
MIN_CONTOUR_AREA = 1000  # Minimum object size in pixels (filters noise)
MOTION_SENSITIVITY = 25  # Lower = more sensitive (threshold value)

# Smoke Detection Settings
SMOKE_THRESHOLD = 5000  # Minimum smoke pixels to trigger alarm
SMOKE_GROWTH_FACTOR = 1.3  # 30% growth indicates expanding smoke
SMOKE_HISTORY_SIZE = 10  # Number of frames to track smoke growth

# HSV Color Range for Smoke Detection
# Smoke characteristics: low saturation, high value (gray/white)
LOWER_SMOKE = np.array([0, 0, 180])    # [Hue, Saturation, Value]
UPPER_SMOKE = np.array([180, 60, 255])  # Wide hue range, low sat, high val

# Visual Settings
ALARM_BORDER_THICKNESS = 10
MOTION_BOX_COLOR = (0, 255, 0)  # Green for motion boxes
MOTION_ALARM_COLOR = (0, 0, 255)  # Red for motion alarm
SMOKE_ALARM_COLOR = (0, 100, 255)  # Orange for smoke alarm

# File Settings
SNAPSHOT_DIR = 'surveillance_snapshots'

# ==================== INITIALIZATION ====================

# Create snapshot directory if it doesn't exist
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
    print(f"📁 Created directory: {SNAPSHOT_DIR}/")

# Initialize webcam
print("🎥 Initializing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera!")
    print("💡 Try checking:")
    print("   - Camera permissions")
    print("   - If another app is using the camera")
    print("   - Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

# Wait for camera to warm up
print("⏳ Camera warming up...")
for _ in range(10):
    cap.read()

# Get first frame for motion detection
ret, previous_frame = cap.read()
if not ret:
    print("❌ ERROR: Could not read from camera!")
    exit()

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

# Initialize smoke history tracking
smoke_history = deque(maxlen=SMOKE_HISTORY_SIZE)

# Alarm states
motion_alarm_active = False
smoke_alarm_active = False
motion_alarm_count = 0
smoke_alarm_count = 0

# ==================== HELPER FUNCTIONS ====================

def save_snapshot(frame, event_type):
    """Save a snapshot with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{SNAPSHOT_DIR}/{event_type}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"   📸 Snapshot saved: {filename}")
    return filename

def get_timestamp():
    """Get formatted timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def draw_info_panel(frame, motion_area, smoke_pixels, fps=0):
    """Draw information panel on frame"""
    h, w = frame.shape[:2]

    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-80), (300, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw text
    cv2.putText(frame, f"Motion: {int(motion_area)} px", (10, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Smoke: {int(smoke_pixels)} px", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Status indicators
    status_x = w - 150
    cv2.putText(frame, "MOTION", (status_x, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(frame, "SMOKE", (status_x, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    if motion_alarm_active:
        cv2.circle(frame, (status_x + 85, h - 60), 8, MOTION_ALARM_COLOR, -1)
    if smoke_alarm_active:
        cv2.circle(frame, (status_x + 85, h - 35), 8, SMOKE_ALARM_COLOR, -1)

# ==================== STARTUP MESSAGE ====================

print("\n" + "="*60)
print("🚨 SURVEILLANCE & FIRE SAFETY SYSTEM ONLINE 🚨")
print("="*60)
print(f"\n📊 Configuration:")
print(f"   Motion threshold: {MOTION_THRESHOLD} pixels")
print(f"   Min object size: {MIN_CONTOUR_AREA} pixels")
print(f"   Smoke threshold: {SMOKE_THRESHOLD} pixels")
print(f"   Smoke growth factor: {SMOKE_GROWTH_FACTOR}x")
print(f"\n🎮 Controls:")
print(f"   Press 'q' to quit")
print(f"   Press 's' to save manual snapshot")
print(f"   Press 'r' to reset alarms")
print(f"\n📁 Snapshots will be saved to: {SNAPSHOT_DIR}/")
print("\n" + "="*60)
print("🟢 System active - monitoring started...\n")

# ==================== MAIN LOOP ====================

frame_count = 0
start_time = cv2.getTickCount()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Warning: Failed to read frame, retrying...")
            continue

        frame_count += 1
        original_frame = frame.copy()

        # Calculate FPS
        if frame_count % 30 == 0:
            end_time = cv2.getTickCount()
            fps = 30 / ((end_time - start_time) / cv2.getTickFrequency())
            start_time = end_time
        else:
            fps = 0

        # ==================== MOTION DETECTION ====================

        # Prepare current frame
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

        # Calculate frame difference
        frame_diff = cv2.absdiff(previous_gray, current_gray)

        # Threshold the difference
        _, motion_thresh = cv2.threshold(frame_diff, MOTION_SENSITIVITY, 255, cv2.THRESH_BINARY)

        # Dilate to fill gaps
        motion_thresh = cv2.dilate(motion_thresh, None, iterations=2)

        # Find contours
        motion_contours, _ = cv2.findContours(motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze motion
        motion_detected = False
        total_motion_area = 0

        for contour in motion_contours:
            area = cv2.contourArea(contour)

            if area > MIN_CONTOUR_AREA:
                motion_detected = True
                total_motion_area += area

                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), MOTION_BOX_COLOR, 2)
                cv2.putText(frame, f"{int(area)}px", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, MOTION_BOX_COLOR, 1)

        # Trigger motion alarm
        if total_motion_area > MOTION_THRESHOLD:
            if not motion_alarm_active:
                motion_alarm_count += 1
                timestamp = get_timestamp()
                print(f"🚨 MOTION ALERT #{motion_alarm_count} [{timestamp}]")
                print(f"   Area changed: {int(total_motion_area)} pixels")
                save_snapshot(original_frame, "motion")
                motion_alarm_active = True

            # Visual alarm
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                         MOTION_ALARM_COLOR, ALARM_BORDER_THICKNESS)
            cv2.putText(frame, "!!! MOTION DETECTED !!!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, MOTION_ALARM_COLOR, 3)
        else:
            if motion_alarm_active:
                print(f"   Motion cleared at {get_timestamp()}")
            motion_alarm_active = False

        # ==================== SMOKE DETECTION ====================

        # Convert frame to HSV
        hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)

        # Create smoke mask
        smoke_mask = cv2.inRange(hsv, LOWER_SMOKE, UPPER_SMOKE)

        # Clean up noise with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

        # Calculate smoke area
        smoke_pixels = np.sum(smoke_mask > 0)

        # Track smoke over time
        smoke_history.append(smoke_pixels)

        # Check for smoke alarm conditions
        smoke_detected = False

        # Condition 1: Immediate threshold
        if smoke_pixels > SMOKE_THRESHOLD:
            smoke_detected = True

        # Condition 2: Growing smoke pattern (more sophisticated)
        if len(smoke_history) >= SMOKE_HISTORY_SIZE:
            recent_avg = np.mean(list(smoke_history)[-5:])
            old_avg = np.mean(list(smoke_history)[:5])

            if old_avg > 0 and recent_avg > old_avg * SMOKE_GROWTH_FACTOR:
                smoke_detected = True

        # Trigger smoke alarm
        if smoke_detected:
            if not smoke_alarm_active:
                smoke_alarm_count += 1
                timestamp = get_timestamp()
                print(f"🔥 SMOKE/FIRE ALERT #{smoke_alarm_count} [{timestamp}]")
                print(f"   Smoke pixels detected: {int(smoke_pixels)}")

                if len(smoke_history) >= 5:
                    growth = (smoke_history[-1] / max(smoke_history[0], 1)) * 100
                    print(f"   Growth rate: {growth:.1f}%")

                save_snapshot(original_frame, "smoke")
                smoke_alarm_active = True

            # Visual alarm (orange/red border)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                         SMOKE_ALARM_COLOR, ALARM_BORDER_THICKNESS)
            cv2.putText(frame, "!!! SMOKE/FIRE ALERT !!!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, SMOKE_ALARM_COLOR, 3)

            # Draw smoke contours
            smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, smoke_contours, -1, SMOKE_ALARM_COLOR, 2)
        else:
            if smoke_alarm_active:
                print(f"   Smoke cleared at {get_timestamp()}")
            smoke_alarm_active = False

        # ==================== DISPLAY ====================

        # Draw info panel
        draw_info_panel(frame, total_motion_area, smoke_pixels, fps if fps > 0 else 30)

        # Show all windows
        cv2.imshow('🚨 Surveillance System', frame)
        cv2.imshow('Motion Detection', motion_thresh)
        cv2.imshow('Smoke Detection', smoke_mask)

        # Update previous frame
        previous_gray = current_gray

        # ==================== KEYBOARD CONTROLS ====================

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n🛑 Shutting down surveillance system...")
            break
        elif key == ord('s'):
            timestamp = get_timestamp()
            filename = save_snapshot(original_frame, "manual")
            print(f"📸 Manual snapshot saved [{timestamp}]: {filename}")
        elif key == ord('r'):
            motion_alarm_active = False
            smoke_alarm_active = False
            smoke_history.clear()
            print(f"🔄 Alarms reset at {get_timestamp()}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user (Ctrl+C)")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ==================== CLEANUP ====================

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("📊 SURVEILLANCE SESSION SUMMARY")
    print("="*60)
    print(f"Total motion alerts: {motion_alarm_count}")
    print(f"Total smoke alerts: {smoke_alarm_count}")
    print(f"Total frames processed: {frame_count}")
    print(f"Snapshots saved in: {SNAPSHOT_DIR}/")
    print("\n✅ Surveillance System Stopped")
    print("="*60)
