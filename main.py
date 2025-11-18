"""Real-time hand gesture overlay with MediaPipe-style landmark dots."""

from typing import List, Tuple

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time


LANDMARK_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)    # Pinky
]

TIP_IDS = [4, 8, 12, 16, 20]


def draw_landmarks(frame: np.ndarray, lm_list: List[List[int]]) -> None:
    """Draw MediaPipe-style skeleton (white lines + red dots)."""
    for start, end in LANDMARK_CONNECTIONS:
        x1, y1 = lm_list[start][:2]
        x2, y2 = lm_list[end][:2]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    for idx, (x, y, _) in enumerate(lm_list):
        color = (0, 200, 255) if idx in TIP_IDS else (0, 0, 255)
        radius = 6 if idx in TIP_IDS else 5
        cv2.circle(frame, (x, y), radius, color, -1)
        if idx in TIP_IDS:
            cv2.putText(frame, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def classify_gesture(fingers_state: List[int]) -> str:
    """Return a friendly gesture label based on which fingers are raised."""
    count = sum(fingers_state)
    thumb, index, middle, ring, pinky = fingers_state

    if count == 0:
        return "Fist"
    if count == 5:
        return "Open Palm"
    if count == 2 and index and middle and not ring and not pinky:
        return "Victory"
    if count == 1 and thumb and not any([index, middle, ring, pinky]):
        return "Thumbs Up"
    if count == 1 and index:
        return "Pointing"
    return f"{count} Fingers"


def draw_hud(frame: np.ndarray, gesture: str, finger_count: int) -> None:
    """Draw translucent info panel with gesture text."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (360, 110), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    cv2.putText(frame, f"Gesture: {gesture}", (28, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Fingers: {finger_count}", (28, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def annotate_tip_coordinates(frame: np.ndarray, lm_list: List[List[int]]) -> None:
    """Display (x, y) next to each fingertip."""
    for idx in TIP_IDS:
        x, y, _ = lm_list[idx]
        cv2.putText(frame, f"({x}, {y})", (x - 20, y + 25),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.85, maxHands=1)
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        hands, _ = detector.findHands(frame, draw=False)
        gesture = "No Hand Detected"
        finger_count = 0

        if hands:
            hand = hands[0]
            lm_list = hand["lmList"]
            bbox = hand["bbox"]
            center = hand["center"]

            draw_landmarks(frame, lm_list)
            annotate_tip_coordinates(frame, lm_list)
            cv2.rectangle(frame, bbox, (128, 0, 255), 2)
            cv2.circle(frame, center, 6, (128, 0, 255), -1)

            fingers_state = detector.fingersUp(hand)
            finger_count = sum(fingers_state)
            gesture = classify_gesture(fingers_state)

        draw_hud(frame, gesture, finger_count)

        current_time = time.time()
        fps = 1 / (current_time - prev_time + 1e-9)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (1120, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




