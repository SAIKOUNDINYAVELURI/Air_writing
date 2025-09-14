import cv2

from hand_tracking import HandTracker
from drawing_utils import AirCanvas, SmoothPoint
from ui_buttons import ButtonBar
from alphabet_recognition import recognize_alphabets

# ---- Camera ----
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
if not ok:
    raise RuntimeError("Could not read from webcam.")
frame = cv2.flip(frame, 1)
H, W = frame.shape[:2]

# ---- Make window resizable ----
cv2.namedWindow("AirWriting — Draw / Erase / Save + OCR", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AirWriting — Draw / Erase / Save + OCR", 1280, 720)

# ---- Modules ----
tracker = HandTracker(draw=True)  # set draw=False to hide skeleton
canvas = AirCanvas(W, H, pen_color=(255, 0, 0), pen_thickness=6, erase_radius=38)
smoother = SmoothPoint(window=5)
buttons = ButtonBar()

# ---- App State ----
mode = "draw"  # "draw" | "erase"
show_ocr = ""  # last OCR result
hover_cooldown = 0  # frames to avoid multiple button fires while hovering

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    # Draw UI buttons
    buttons.draw(frame, active_label="Draw" if mode == "draw" else "Erase")

    # Process hand
    results = tracker.process(frame)

    # Default: fingertip marker
    idx_xy = tracker.index_tip_xy(frame, results)
    if idx_xy:
        cv2.circle(frame, idx_xy, 7, (0, 0, 255), -1)

    # --- Button interaction ---
    if hover_cooldown > 0:
        hover_cooldown -= 1
    if idx_xy and hover_cooldown == 0:
        clicked = buttons.which_clicked(*idx_xy)
        if clicked == "draw":
            mode = "draw"
            hover_cooldown = 12
        elif clicked == "erase":
            mode = "erase"
            hover_cooldown = 12
        elif clicked == "save":
            path = canvas.save()
            # Run OCR on saved drawing
            img = cv2.imread(path)
            text, _ = recognize_alphabets(img)
            show_ocr = text if text else "(no text detected)"
            hover_cooldown = 15
        elif clicked == "red":
            canvas.pen_color = (0, 0, 255)
            hover_cooldown = 12
        elif clicked == "green":
            canvas.pen_color = (0, 255, 0)
            hover_cooldown = 12
        elif clicked == "blue":
            canvas.pen_color = (255, 0, 0)
            hover_cooldown = 12
        elif clicked == "black":
            canvas.pen_color = (0, 0, 0)
            hover_cooldown = 12
        elif clicked == "exit":  # ✅ Added exit button handler
            break

    # --- Gestures for writing/erasing ---
    if results.multi_hand_landmarks:
        fingers = tracker.fingers_up(results, hand_idx=0)
        palm_open = tracker.is_palm_open(results, hand_idx=0)

        if palm_open:
            center = tracker.palm_center_xy(frame, results, hand_idx=0)
            canvas.erase_at(center)
            canvas.reset_stroke()
        else:
            index_up = fingers[1] == 1
            middle_down = fingers[2] == 0

            if idx_xy and index_up and middle_down:
                x, y = smoother.filter(*idx_xy)
                if mode == "draw":
                    canvas.draw_from_to(canvas.prev, (x, y))
                elif mode == "erase":
                    canvas.erase_at((x, y))
                    canvas.reset_stroke()
            else:
                canvas.reset_stroke()
    else:
        canvas.reset_stroke()

    # Overlay canvas
    out = canvas.overlay(frame, alpha=0.35)

    # HUD
    cv2.putText(
        out,
        f"Mode: {mode.upper()}  |  Tip: Index finger to draw, open palm to erase",
        (12, H - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if show_ocr:
        cv2.putText(
            out,
            f"OCR: {show_ocr}",
            (W // 2 - 220, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    cv2.imshow("AirWriting — Draw / Erase / Save + OCR", out)

    # ---- Exit keys ----
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord("c"):
        canvas.clear()
        show_ocr = ""

cap.release()
cv2.destroyAllWindows()
