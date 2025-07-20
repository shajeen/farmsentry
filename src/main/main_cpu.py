"""
Main application for the FarmSentry animal detection system (CPU Version).
"""

import cv2
import time
from src.core.detection_module import AnimalDetector
from src.core.alert_module import AlertHandler
import src.config.config_cpu as config

def main():
    """
    Main function to run the animal detection system.
    """
    print("\n=== FarmSentry: Farmland Animal Alert System (CPU) ===")
    print(f"Configuration: Confidence={config.CONFIDENCE_THRESHOLD}, Frame Size={config.DETECTION_SIZE}")

    # Initialize video capture
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at {config.VIDEO_PATH}")
        return

    # Initialize components
    detector = AnimalDetector(config)
    alert_handler = AlertHandler(config)
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nVideo processing completed.")
                break

            frame_count += 1

            # Skip frames for performance if configured
            if frame_count % (config.FRAME_SKIP + 1) != 0:
                continue

            # Process the frame for animal detections
            processed_frame, detections = detector.process_frame(frame)

            # Update and display alerts
            alert_handler.update_alerts(detections)
            processed_frame = alert_handler.add_visual_alerts(processed_frame)

            # Display the processed frame
            cv2.imshow('FarmSentry - Animal Detection (CPU)', processed_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nUser terminated the process.")
                break

    finally:
        # Clean up and display summary
        end_time = time.time()
        duration = end_time - start_time
        fps = frame_count / duration if duration > 0 else 0

        print("\n=== Summary ===")
        print(f"Processed {frame_count} frames in {duration:.2f} seconds.")
        print(f"Average FPS: {fps:.2f}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()