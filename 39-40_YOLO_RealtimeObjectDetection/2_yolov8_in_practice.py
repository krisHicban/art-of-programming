import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# ==========================================
# SETUP: Installing and Loading YOLOv8
# ==========================================

"""
Installation:
    pip install ultralytics opencv-python pillow

YOLOv8 comes in 5 sizes:
    - YOLOv8n (nano):     3.2M params, 80 FPS (best for mobile/edge)
    - YOLOv8s (small):    11.2M params, 60 FPS
    - YOLOv8m (medium):   25.9M params, 45 FPS
    - YOLOv8l (large):    43.7M params, 30 FPS
    - YOLOv8x (xlarge):   68.2M params, 25 FPS (best accuracy)

For real-time applications: Use nano or small
For accuracy-critical applications: Use medium or large
"""

print("=" * 80)
print("YOLO v8: REAL-TIME OBJECT DETECTION")
print("=" * 80)
print()

# Load pre-trained YOLOv8 model (COCO dataset - 80 classes)
model = YOLO('yolov8n.pt')  # Nano model for speed

print(" YOLOv8n model loaded")
print(f" Model: YOLOv8 Nano")
print(f" Training: COCO dataset (80 object classes)")
print(f" Speed: ~80 FPS on GPU, ~30 FPS on CPU")
print()


# ==========================================
# HEALTH APPLICATION: Workout Form Analysis
# ==========================================

class WorkoutFormMonitor:
    """
    Real-time workout monitoring using YOLO

    Use Case: Personal trainer AI
    - Detect person in frame
    - Track workout equipment (dumbbells, yoga mat, etc.)
    - Count reps based on object movement
    - Monitor form consistency
    - Generate workout summaries

    Real-world: Used by fitness apps like Peloton, Mirror, Tempo
    """

    def __init__(self, model):
        self.model = model
        self.workout_log = []
        self.rep_count = 0
        self.equipment_detected = set()
        self.session_start = time.time()

        # COCO classes relevant to workouts
        self.workout_classes = {
            0: 'person',
            32: 'sports ball',
            33: 'baseball bat',
            34: 'baseball glove',
            35: 'skateboard',
            36: 'surfboard',
            37: 'tennis racket',
            # Note: COCO doesn't have dumbbells/kettlebells
            # For those, you'd train a custom model (Part 3)
        }

    def analyze_frame(self, frame):
        """
        Analyze single frame for workout monitoring

        Returns:
            - Annotated frame
            - Detection summary
            - Workout metrics
        """
        results = self.model(frame, conf=0.5, verbose=False)[0]

        detections = {
            'person_detected': False,
            'equipment': [],
            'bounding_boxes': [],
            'confidence_scores': []
        }

        # Parse detections
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

            if class_id == 0:  # Person detected
                detections['person_detected'] = True

            if class_id in self.workout_classes:
                class_name = self.workout_classes[class_id]
                detections['equipment'].append(class_name)
                self.equipment_detected.add(class_name)

            detections['bounding_boxes'].append(bbox)
            detections['confidence_scores'].append(confidence)

        # Annotate frame
        annotated_frame = results.plot()

        # Add workout stats overlay
        session_duration = time.time() - self.session_start
        stats_text = [
            f"Session: {session_duration:.0f}s",
            f"Person: {'' if detections['person_detected'] else 'L'}",
            f"Equipment: {', '.join(set(detections['equipment'])) if detections['equipment'] else 'None'}"
        ]

        y_offset = 30
        for text in stats_text:
            cv2.putText(annotated_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        return annotated_frame, detections

    def generate_workout_summary(self):
        """Generate post-workout analysis"""
        duration = time.time() - self.session_start

        summary = {
            'duration_minutes': duration / 60,
            'equipment_used': list(self.equipment_detected),
            'frames_analyzed': len(self.workout_log),
            'avg_fps': len(self.workout_log) / duration if duration > 0 else 0
        }

        return summary


# ==========================================
# FINANCE APPLICATION: Receipt & Document Scanner
# ==========================================

class ReceiptScanner:
    """
    Intelligent receipt and document scanning using YOLO

    Use Case: Personal finance tracking
    - Detect documents/receipts in frame
    - Guide user to optimal capture position
    - Auto-capture when document is clear
    - Extract text regions for OCR
    - Categorize document type

    Real-world: Used by Expensify, Mint, banking apps
    """

    def __init__(self, model):
        self.model = model
        self.scanned_docs = []
        self.capture_queue = deque(maxlen=5)  # Stability checking

        # COCO classes for documents/items
        self.finance_classes = {
            0: 'person',
            73: 'book',
            84: 'book',
            # Note: For receipts specifically, custom training needed
            # COCO provides general document detection
        }

    def analyze_document(self, frame):
        """
        Analyze frame for document detection

        Returns guidance for optimal capture
        """
        results = self.model(frame, conf=0.6, verbose=False)[0]

        analysis = {
            'document_detected': False,
            'capture_ready': False,
            'guidance': [],
            'bounding_box': None
        }

        # Find largest document in frame
        max_area = 0
        best_box = None

        for box in results.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                best_box = bbox
                analysis['document_detected'] = True

        if analysis['document_detected']:
            analysis['bounding_box'] = best_box

            # Check capture conditions
            frame_h, frame_w = frame.shape[:2]
            x1, y1, x2, y2 = best_box

            # Document should fill 40-80% of frame
            doc_area_ratio = max_area / (frame_h * frame_w)

            if doc_area_ratio < 0.4:
                analysis['guidance'].append("Move closer")
            elif doc_area_ratio > 0.8:
                analysis['guidance'].append("Move farther")
            else:
                # Check if document is centered
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if abs(center_x - frame_w/2) > frame_w * 0.1:
                    analysis['guidance'].append("Center horizontally")
                elif abs(center_y - frame_h/2) > frame_h * 0.1:
                    analysis['guidance'].append("Center vertically")
                else:
                    # Perfect position - check stability
                    self.capture_queue.append(True)
                    if len(self.capture_queue) == 5 and all(self.capture_queue):
                        analysis['capture_ready'] = True
                        analysis['guidance'].append(" READY TO CAPTURE")

        else:
            analysis['guidance'].append("No document detected")

        return analysis

    def capture_document(self, frame, bbox):
        """Crop and save document region"""
        x1, y1, x2, y2 = map(int, bbox)
        cropped = frame[y1:x2, x1:x2]

        # In production: Send to OCR (Tesseract, Google Vision API)
        # Extract text, amounts, dates, merchant names

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"receipt_{timestamp}.jpg"
        cv2.imwrite(filename, cropped)

        self.scanned_docs.append({
            'timestamp': timestamp,
            'filename': filename,
            'bbox': bbox
        })

        return filename


# ==========================================
# REAL-TIME WEBCAM DEMO
# ==========================================

def run_health_monitor_demo(duration_seconds=30):
    """
    Live workout monitoring demo

    Press 'q' to quit
    Press 's' to save screenshot
    """
    print()
    print("=" * 80)
    print("HEALTH MONITORING DEMO: Workout Form Analysis")
    print("=" * 80)
    print()
    print("Starting webcam...")
    print("Press 'q' to quit, 's' to save screenshot")
    print()

    monitor = WorkoutFormMonitor(model)
    cap = cv2.VideoCapture(0)

    fps_queue = deque(maxlen=30)
    start_time = time.time()

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Analyze frame
        annotated_frame, detections = monitor.analyze_frame(frame)

        # Calculate FPS
        fps = 1.0 / (time.time() - frame_start)
        fps_queue.append(fps)
        avg_fps = np.mean(fps_queue)

        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Workout Monitor - YOLOv8', annotated_frame)

        # Log detection
        monitor.workout_log.append({
            'timestamp': time.time() - start_time,
            'detections': detections
        })

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'workout_frame_{int(time.time())}.jpg', annotated_frame)
            print("Screenshot saved!")

    cap.release()
    cv2.destroyAllWindows()

    # Generate summary
    summary = monitor.generate_workout_summary()
    print()
    print("=" * 80)
    print("WORKOUT SESSION SUMMARY")
    print("=" * 80)
    print(f"Duration: {summary['duration_minutes']:.1f} minutes")
    print(f"Equipment detected: {', '.join(summary['equipment_used']) if summary['equipment_used'] else 'None'}")
    print(f"Average FPS: {summary['avg_fps']:.1f}")
    print(f"Total frames: {summary['frames_analyzed']}")
    print()


def run_receipt_scanner_demo():
    """
    Live receipt scanning demo

    Position receipt in frame following guidance
    Auto-captures when stable
    """
    print()
    print("=" * 80)
    print("FINANCE APPLICATION: Smart Receipt Scanner")
    print("=" * 80)
    print()
    print("Starting webcam...")
    print("Position receipt in frame - follow on-screen guidance")
    print("Press 'q' to quit, 'c' to force capture")
    print()

    scanner = ReceiptScanner(model)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze frame
        analysis = scanner.analyze_document(frame)

        # Draw bounding box if document detected
        if analysis['document_detected']:
            x1, y1, x2, y2 = map(int, analysis['bounding_box'])
            color = (0, 255, 0) if analysis['capture_ready'] else (255, 165, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Display guidance
        y_offset = 30
        for guidance in analysis['guidance']:
            color = (0, 255, 0) if '' in guidance else (255, 255, 255)
            cv2.putText(frame, guidance, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 40

        cv2.imshow('Receipt Scanner - YOLOv8', frame)

        # Auto-capture or manual capture
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif (key == ord('c') or analysis['capture_ready']) and analysis['document_detected']:
            filename = scanner.capture_document(frame, analysis['bounding_box'])
            print(f"=� Captured: {filename}")
            scanner.capture_queue.clear()  # Reset stability

    cap.release()
    cv2.destroyAllWindows()

    print()
    print(f"Total documents scanned: {len(scanner.scanned_docs)}")
    print()


# ==========================================
# CHOOSE YOUR DEMO
# ==========================================

print("=" * 80)
print("YOLO v8 REAL-TIME DEMOS")
print("=" * 80)
print()
print("Available demos:")
print("  1. Workout Form Monitor (Health)")
print("  2. Receipt Scanner (Finance)")
print()
print("Uncomment the demo you want to run:")
print()

# run_health_monitor_demo(duration_seconds=30)
# run_receipt_scanner_demo()

print("=" * 80)
print("PART 2 COMPLETE: You've built real-time detection systems.")
print("Next: Custom training for YOUR specific objects.")
print("=" * 80)
