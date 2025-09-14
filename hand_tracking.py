import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """
    MediaPipe Hands wrapper with helpers:
      - process(frame) -> results
      - index_tip_xy(frame, results)
      - fingers_up(results) -> [thumb, index, middle, ring, pinky]
      - is_palm_open(results) -> True if all fingers extended
      - palm_center_xy(frame, results) -> approx palm center
    """

    def __init__(
        self, max_num_hands=1, detection_conf=0.7, tracking_conf=0.7, draw=False
    ):
        self.draw_landmarks = draw
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if self.draw_landmarks and results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
        return results

    @staticmethod
    def index_tip_xy(frame, results, hand_idx=0):
        if not results.multi_hand_landmarks:
            return None
        h, w = frame.shape[:2]
        lm = results.multi_hand_landmarks[hand_idx].landmark
        x = int(lm[8].x * w)
        y = int(lm[8].y * h)
        return (x, y)

    @staticmethod
    def fingers_up(results, hand_idx=0):
        """
        Returns list of 5 ints [thumb, index, middle, ring, pinky], 1=up, 0=down
        Heuristics: fingers are 'up' if tip.y < pip.y (except thumb: use x relation)
        """
        if not results.multi_hand_landmarks:
            return [0, 0, 0, 0, 0]
        lm = results.multi_hand_landmarks[hand_idx].landmark
        # Thumb: compare x of tip(4) and ip(3) (assuming right hand mirrored webcam)
        thumb_up = 1 if lm[4].x < lm[3].x else 0
        index_up = 1 if lm[8].y < lm[6].y else 0
        middle_up = 1 if lm[12].y < lm[10].y else 0
        ring_up = 1 if lm[16].y < lm[14].y else 0
        pinky_up = 1 if lm[20].y < lm[18].y else 0
        return [thumb_up, index_up, middle_up, ring_up, pinky_up]

    def is_palm_open(self, results, hand_idx=0):
        # All fingers extended
        f = self.fingers_up(results, hand_idx)
        return all(v == 1 for v in f)

    @staticmethod
    def palm_center_xy(frame, results, hand_idx=0):
        """
        Approximate palm center as the mean of wrist & MCPs: [0, 5, 9, 13, 17]
        """
        if not results.multi_hand_landmarks:
            return None
        h, w = frame.shape[:2]
        ids = [0, 5, 9, 13, 17]
        lm = results.multi_hand_landmarks[hand_idx].landmark
        xs, ys = [], []
        for i in ids:
            xs.append(int(lm[i].x * w))
            ys.append(int(lm[i].y * h))
        if not xs:
            return None
        return (int(np.mean(xs)), int(np.mean(ys)))
