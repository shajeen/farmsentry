import time
from typing import Dict, List
import numpy as np
import cv2

class AlertHandler:
    """Manages alert states and visual indicators with a cooldown period."""

    def __init__(self, config_module):
        self.config = config_module
        self.active_alerts: Dict[str, float] = {}
        self.last_alert_time = 0
        self.last_cooldown_time = 0

    def update_alerts(self, detections: Dict[str, List[tuple]]) -> None:
        current_time = time.time()

        if detections:
            self.last_alert_time = current_time
            # Update active alerts
            for class_name in detections:
                self.active_alerts[class_name] = current_time

        # Expire old alerts
        expired = [k for k, v in self.active_alerts.items()
                   if current_time - v > self.config.ALERT_DURATION]
        for k in expired:
            del self.active_alerts[k]

    def add_visual_alerts(self, frame: np.ndarray) -> np.ndarray:
        current_time = time.time()

        # Check if an alert should be displayed
        if current_time - self.last_alert_time < self.config.ALERT_DURATION:
            # Check for cooldown
            if current_time - self.last_cooldown_time > self.config.ALERT_COOLDOWN:
                self.last_cooldown_time = current_time

                # Display visual alerts
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1], frame.shape[0]),
                              (0, 0, 255), 15)  # Red border

                if int(current_time * 2) % 2 == 0:  # Flashing text
                    alert_text = "ANIMAL INTRUSION DETECTED!"
                    (text_width, text_height), _ = cv2.getTextSize(
                        alert_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
                    x = (frame.shape[1] - text_width) // 2
                    y = (frame.shape[0] + text_height) // 2
                    cv2.putText(frame, alert_text, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

            # Display active alert count
            alert_count = len(self.active_alerts)
            cv2.putText(frame, f"Active Alerts: {alert_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame
