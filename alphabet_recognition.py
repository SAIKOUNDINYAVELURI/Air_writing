import os
import cv2
import numpy as np
import pytesseract

# Set Tesseract path (env variable preferred, fallback to default Windows path)
tess_cmd = os.getenv("TESSERACT_CMD")
if tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = tess_cmd
else:
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


def preprocess_for_ocr(img_bgr):
    """
    Preprocess the canvas for better OCR on handwritten characters:
      - Convert to gray, invert (white background, dark text)
      - Threshold, denoise, dilate (connect strokes), resize larger
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)  # invert → black text on white
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.medianBlur(thr, 3)  # remove noise
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.dilate(thr, kernel, iterations=1)  # strengthen strokes
    thr = cv2.resize(
        thr, (thr.shape[1] * 2, thr.shape[0] * 2), interpolation=cv2.INTER_LINEAR
    )
    return thr


def recognize_alphabets(img_bgr):
    """
    Runs Tesseract OCR constrained to alphabets (A–Z, a–z).
    """
    proc = preprocess_for_ocr(img_bgr)
    config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    text = pytesseract.image_to_string(proc, config=config)
    text = " ".join(text.split())  # cleanup
    return text, proc
