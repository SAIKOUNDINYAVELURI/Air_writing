import cv2
import numpy as np
import os
import time
from collections import deque


class SmoothPoint:
    """Simple moving-average smoother for (x,y)."""

    def __init__(self, window=5):
        self.buf = deque(maxlen=window)

    def filter(self, x, y):
        self.buf.append((x, y))
        xs = [p[0] for p in self.buf]
        ys = [p[1] for p in self.buf]
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


class AirCanvas:
    def __init__(
        self,
        width=640,
        height=480,
        pen_color=(255, 0, 0),
        pen_thickness=5,
        erase_radius=35,
    ):
        # Start with white canvas
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        self.prev = None
        self.pen_color = pen_color
        self.pen_thickness = pen_thickness
        self.erase_radius = erase_radius

        # Predefined color palette
        self.colors = {
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "black": (0, 0, 0),
            "purple": (255, 0, 255),
            "orange": (0, 165, 255),
        }

    def set_color(self, color_name):
        """Change pen color by name (from palette)."""
        if color_name in self.colors:
            self.pen_color = self.colors[color_name]

    def draw_from_to(self, pt_from, pt_to):
        if pt_from is None or pt_to is None:
            self.prev = pt_to
            return
        cv2.line(
            self.canvas, pt_from, pt_to, self.pen_color, self.pen_thickness, cv2.LINE_AA
        )
        self.prev = pt_to

    def erase_at(self, center):
        if center is None:
            return
        cv2.circle(
            self.canvas, center, self.erase_radius, (255, 255, 255), -1, cv2.LINE_AA
        )

    def reset_stroke(self):
        self.prev = None

    def overlay(self, frame, alpha=0.35):
        # Non-destructive blend
        return cv2.addWeighted(frame, 1.0, self.canvas, alpha, 0)

    def clear(self):
        self.canvas[:] = 255

    def save(self, dirname="output/saved_drawings"):
        os.makedirs(dirname, exist_ok=True)
        path = os.path.join(dirname, f"drawing_{int(time.time())}.png")
        cv2.imwrite(path, self.canvas)
        return path
