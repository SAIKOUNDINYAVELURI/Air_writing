import cv2

class Button:
    def __init__(self, label, x, y, w=120, h=40, color=(220, 220, 220)):
        self.label = label
        self.rect = (x, y, w, h)
        self.active = False
        self.base_color = color

    def draw(self, frame):
        x, y, w, h = self.rect
        fill = (0, 200, 0) if self.active else self.base_color
        cv2.rectangle(frame, (x, y), (x + w, y + h), fill, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), 2)
        cv2.putText(frame, self.label, (x + 10, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255) if self.base_color == (0,0,0) else (0, 0, 0), 2)

    def hit(self, px, py):
        x, y, w, h = self.rect
        return x <= px <= x + w and y <= py <= y + h


class ButtonBar:
    def __init__(self, frame_width=640, margin=10, btn_width=120, btn_height=40):
        self.buttons = []
        self.active_label = None

        # Define labels and colors
        labels_colors = [
            ("Draw", (220, 220, 220)),
            ("Erase", (220, 220, 220)),
            ("Save", (220, 220, 220)),
            ("Black", (0, 0, 0)),
            ("Red", (0, 0, 255)),
            ("Green", (0, 255, 0)),
            ("Blue", (255, 0, 0)),
            ("Exit", (180, 180, 180))
        ]

        # Auto arrange buttons into rows
        x, y = margin, margin
        for label, color in labels_colors:
            if x + btn_width + margin > frame_width:  # Wrap to next row
                x = margin
                y += btn_height + margin
            self.buttons.append(Button(label, x, y, btn_width, btn_height, color))
            x += btn_width + margin

    def draw(self, frame, active_label):
        for b in self.buttons:
            b.active = (b.label.lower() == active_label.lower())
            b.draw(frame)

    def which_clicked(self, x, y):
        for b in self.buttons:
            if b.hit(x, y):
                return b.label.lower()
        return None
