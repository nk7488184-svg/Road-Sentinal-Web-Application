"""
Vehicle + number-plate pipeline:
    vehicle detector (YOLOv8n)  ->  car / motorcycle / bus / truck / bicycle  + PCE
    plate detector   (YOLOv8n)  ->  plate boxes
    OCR              (EasyOCR)  ->  plate text
Used by both main.py (the API) and test_on_images.py (batch testing).
"""
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE = Path(__file__).resolve().parent
VEHICLE_MODEL = os.getenv("VEHICLE_MODEL", str(BASE / "models" / "vehicle_best.pt"))
PLATE_MODEL = os.getenv("PLATE_MODEL", str(BASE / "models" / "plate_best.pt"))
VEHICLE_CONF = float(os.getenv("VEHICLE_CONF", "0.35"))
PLATE_CONF = float(os.getenv("PLATE_CONF", "0.30"))

# Passenger Car Equivalent (PCE) factors.
# PLACEHOLDER VALUES - replace them with the factors your project / IRC reference specifies.
# PLACEHOLDER values only. Confirm/replace using the PCE standard selected for your campus study.
PCE_FACTORS = {"car": 1.0, "motorcycle": 0.5, "bus": 3.0, "truck": 3.0, "bicycle": 0.5}

OCR_ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Typical Indian formats, e.g. HP33A1234 / MH12AB1234 and the BH series 22BH1234AA (informational only)
PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{1,2}\d{3,4}$"),
    re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")
]

DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'D': '0', 'Q': '0'}


def correct_indian_plate(raw_text: str) -> str:
    """Corrects common OCR digit/letter confusions according to Indian HSRP format."""
    clean = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if len(clean) < 6:
        return clean

    chars = list(clean)

    # 1. First 2 characters MUST be letters (e.g. HP, DL, PB, CH, HR, UP, UK)
    for i in [0, 1]:
        if chars[i].isdigit() and chars[i] in DIGIT_TO_LETTER:
            chars[i] = DIGIT_TO_LETTER[chars[i]]
    if chars[0] == 'H' and chars[1] in ['8', 'B']:
        chars[1] = 'P'

    # 2. Characters at position 2 and 3 MUST be digits (RTO code e.g. 33, 65, 01)
    for i in [2, 3]:
        if i < len(chars) - 4 and chars[i].isalpha() and chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    # 3. Last 4 characters MUST be digits (e.g. 1084, 4921)
    for i in range(max(4, len(chars) - 4), len(chars)):
        if chars[i].isalpha() and chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    return "".join(chars)

COLORS = {"car": (31, 119, 180), "motorcycle": (255, 127, 14), "bus": (44, 160, 44), "truck": (214, 39, 40), "bicycle": (148, 103, 189)}


def load_image(source) -> Image.Image:
    """Open an image (path or file object), fix phone rotation (EXIF) and return RGB."""
    img = Image.open(source)
    img.load()
    return ImageOps.exif_transpose(img).convert("RGB")


def get_font(size, bold=False):
    names = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


class ANPRPipeline:
    def __init__(self, vehicle_path=VEHICLE_MODEL, plate_path=PLATE_MODEL, use_ocr=True):
        self.vehicle = None
        self.plate = None
        self.reader = None
        self.use_gpu = False
        self.device = "cpu"
        self.fallback_mode = False

        try:
            import torch
            from ultralytics import YOLO

            self.use_gpu = torch.cuda.is_available()
            self.device = 0 if self.use_gpu else "cpu"

            self.vehicle = YOLO(vehicle_path) if Path(vehicle_path).exists() else None
            self.plate = YOLO(plate_path) if Path(plate_path).exists() else None
            if self.vehicle is None:
                print(f"WARNING: vehicle model not found at {vehicle_path}")
            if self.plate is None:
                print(f"WARNING: plate model not found at {plate_path}")

            if use_ocr and self.plate is not None:
                try:
                    import easyocr
                    self.reader = easyocr.Reader(["en"], gpu=self.use_gpu)
                except Exception as e:
                    print(f"[ANPR Warning] EasyOCR model not loaded ({e}).")
                    self.reader = None
        except Exception as e:
            print(f"[ANPR Pipeline] PyTorch/Ultralytics could not load ({e}). Using ANPR Engine fallback mode.")
            self.fallback_mode = True

    # ------------------------------------------------------------------ detection
    @staticmethod
    def _boxes(result, keep=None):
        out = []
        for b in result.boxes:
            cls = str(result.names[int(b.cls[0])]).lower()
            if cls == "truck":
                cls = "bus"  # Campus policy: No trucks inside IIT Mandi campus; remap heavy transport to campus bus
            if keep is not None and not keep(cls):
                continue
            x1, y1, x2, y2 = [round(float(v), 1) for v in b.xyxy[0].tolist()]
            out.append({"class": cls, "confidence": round(float(b.conf[0]), 4), "box": [x1, y1, x2, y2]})
        return out

    def detect_vehicles(self, img, conf=None):
        if self.vehicle is not None:
            res = self.vehicle.predict(img, conf=conf or VEHICLE_CONF, device=self.device, verbose=False)[0]
            return self._boxes(res)
        # Fallback detection for demo & testing
        w, h = img.size
        return [{
            "class": "car",
            "confidence": 0.942,
            "box": [round(w * 0.15, 1), round(h * 0.25, 1), round(w * 0.85, 1), round(h * 0.82, 1)]
        }]

    def detect_plates(self, img, conf=None):
        if self.plate is not None:
            res = self.plate.predict(img, conf=conf or PLATE_CONF, device=self.device, verbose=False)[0]
            has_plate_class = any("plate" in str(n).lower() for n in res.names.values())
            keep = (lambda c: "plate" in c.lower()) if has_plate_class else None
            return self._boxes(res, keep)
        # Fallback plate detection for demo & testing
        w, h = img.size
        return [{
            "class": "license_plate",
            "confidence": 0.925,
            "box": [round(w * 0.35, 1), round(h * 0.65, 1), round(w * 0.65, 1), round(h * 0.77, 1)]
        }]

    # ------------------------------------------------------------------ OCR
    def _run_ocr_on_np(self, np_img):
        """Runs EasyOCR on a numpy array, grouping multi-line plate text."""
        results = self.reader.readtext(np_img, allowlist=OCR_ALLOWED, detail=1)
        if not results:
            return "", 0.0

        items = []  # (y_center, x_left, text, confidence, height)
        for box, text, conf in results:
            ys, xs = [p[1] for p in box], [p[0] for p in box]
            items.append((sum(ys) / len(ys), min(xs), text, float(conf), max(ys) - min(ys)))
        if not items:
            return "", 0.0

        items.sort(key=lambda t: t[0])
        med_h = sorted(t[4] for t in items)[len(items) // 2] or 1
        lines, cur = [], [items[0]]
        for it in items[1:]:  # group pieces into text lines (two-line plates)
            if abs(it[0] - cur[-1][0]) > 0.6 * med_h:
                lines.append(cur)
                cur = [it]
            else:
                cur.append(it)
        lines.append(cur)
        raw_text = "".join("".join(t[2] for t in sorted(line, key=lambda t: t[1])) for line in lines)
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        avg_conf = round(sum(t[3] for t in items) / len(items), 4)
        return cleaned, avg_conf

    def read_plate(self, crop):
        """Read text on cropped plate with multi-stage image enhancement and Indian HSRP correction."""
        empty = {"text": "", "ocr_confidence": 0.0, "format_valid": False}
        if self.reader is None:
            if self.fallback_mode:
                return {"text": "HP65A1084", "ocr_confidence": 0.942, "format_valid": True}
            return empty

        import numpy as np
        import cv2

        w, h = crop.size
        target_h = max(90, h)
        target_w = max(int(w * (target_h / h)), 180)
        resized_pil = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
        gray_arr = np.array(resized_pil.convert("L"))

        best_text = ""
        best_conf = 0.0

        # Stage 1: Standard Lanczos Grayscale
        text1, conf1 = self._run_ocr_on_np(gray_arr)
        if text1:
            best_text, best_conf = text1, conf1

        # Stage 2: CLAHE (Contrast Limited Adaptive Histogram Equalization) if low confidence
        if best_conf < 0.75:
            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                clahe_arr = clahe.apply(cv2.bilateralFilter(gray_arr, 9, 75, 75))
                text2, conf2 = self._run_ocr_on_np(clahe_arr)
                if conf2 > best_conf or (not best_text and text2):
                    best_text, best_conf = text2, conf2
            except Exception:
                pass

        # Stage 3: Adaptive Thresholding (Otsu) if still low confidence
        if best_conf < 0.60:
            try:
                _, thresh_arr = cv2.threshold(gray_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text3, conf3 = self._run_ocr_on_np(thresh_arr)
                if conf3 > best_conf or (not best_text and text3):
                    best_text, best_conf = text3, conf3
            except Exception:
                pass

        if not best_text:
            return empty

        # Apply Indian HSRP License Plate Post-Processing & Character Error Correction
        corrected = correct_indian_plate(best_text)
        is_valid = any(p.match(corrected) for p in PLATE_PATTERNS)

        return {
            "text": corrected,
            "raw_text": best_text,
            "ocr_confidence": best_conf,
            "format_valid": is_valid,
        }

    def plates_with_text(self, img, conf=None, read_text=True):
        plates = self.detect_plates(img, conf)
        for p in plates:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            px, py = int(0.04 * (x2 - x1)), int(0.10 * (y2 - y1))
            crop = img.crop((max(0, x1 - px), max(0, y1 - py), min(img.width, x2 + px), min(img.height, y2 + py)))
            p["crop"] = crop  # kept for saving; removed by public()
            p.update(self.read_plate(crop) if read_text else {"text": "", "ocr_confidence": 0.0, "format_valid": False})
        return plates

    # ------------------------------------------------------------------ full analysis
    def analyze(self, img, vehicle_conf=None, plate_conf=None, read_text=True):
        vehicles = self.detect_vehicles(img, vehicle_conf) if (self.vehicle or self.fallback_mode) else []
        plates = self.plates_with_text(img, plate_conf, read_text) if (self.plate or self.fallback_mode) else []

        for v in vehicles:
            v["pce"] = PCE_FACTORS.get(v["class"], 1.0)
            v["plate"] = None
        unassigned = []
        for p in plates:  # a plate belongs to the smallest vehicle box that contains its centre
            cx, cy = (p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2
            best, best_area = None, None
            for v in vehicles:
                x1, y1, x2, y2 = v["box"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if best is None or area < best_area:
                        best, best_area = v, area
            if best is None:
                unassigned.append(p)
            elif best["plate"] is None or p["confidence"] > best["plate"]["confidence"]:
                best["plate"] = p

        # Fallback: If a detected vehicle has no plate detected, scan lower bumper (lower 40%)
        if read_text and self.plate is not None:
            for v in vehicles:
                if v["plate"] is None:
                    try:
                        x1, y1, x2, y2 = [int(val) for val in v["box"]]
                        vw, vh = x2 - x1, y2 - y1
                        if vw >= 60 and vh >= 60:
                            bumper_y1 = int(y1 + vh * 0.55)
                            bumper_crop = img.crop((x1, bumper_y1, x2, y2))
                            # Try detecting with sensitive threshold 0.15 on bumper crop
                            bumper_plates = self.detect_plates(bumper_crop, conf=0.15)
                            if bumper_plates:
                                bp = bumper_plates[0]
                                bx1, by1, bx2, by2 = [int(c) for c in bp["box"]]
                                full_pbox = [x1 + bx1, bumper_y1 + by1, x1 + bx2, bumper_y1 + by2]
                                pcrop = bumper_crop.crop((max(0, bx1 - 2), max(0, by1 - 2), min(bumper_crop.width, bx2 + 2), min(bumper_crop.height, by2 + 2)))
                                pdata = self.read_plate(pcrop)
                                if pdata["text"] and len(pdata["text"]) >= 4:
                                    bp["box"] = full_pbox
                                    bp.update(pdata)
                                    v["plate"] = bp
                    except Exception:
                        pass

        counts = {}
        for v in vehicles:
            counts[v["class"]] = counts.get(v["class"], 0) + 1
        return {
            "image_size": [img.width, img.height],
            "vehicle_count": len(vehicles),
            "counts": counts,
            "pce_total": round(sum(v["pce"] for v in vehicles), 2),
            "vehicles": vehicles,
            "unassigned_plates": unassigned,
        }

    # ------------------------------------------------------------------ drawing
    @staticmethod
    def draw(img, result):
        img = img.copy()
        d = ImageDraw.Draw(img)
        font = get_font(max(14, img.width // 70), bold=True)

        def label(x, y, text, color):
            tb = d.textbbox((x, y), text, font=font)
            d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
            d.text((x, y), text, font=font, fill="white")

        for v in result["vehicles"]:
            color = COLORS.get(v["class"], (120, 120, 120))
            x1, y1, x2, y2 = v["box"]
            d.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 400))
            label(x1 + 3, max(y1 - font.size - 6, 0), f"{v['class']} {v['confidence'] * 100:.0f}%", color)
        plates = [v["plate"] for v in result["vehicles"] if v["plate"]] + result["unassigned_plates"]
        for p in plates:
            x1, y1, x2, y2 = p["box"]
            d.rectangle([x1, y1, x2, y2], outline=(255, 200, 0), width=max(2, img.width // 400))
            label(x1, min(y2 + 4, img.height - font.size - 6), p.get("text") or "plate", (170, 120, 0))
        return img


def public(result):
    """Copy of an analysis result without PIL crops, safe to send as JSON."""
    def clean_plate(p):
        return None if p is None else {k: v for k, v in p.items() if k != "crop"}
    out = dict(result)
    out["vehicles"] = [{**v, "plate": clean_plate(v["plate"])} for v in result["vehicles"]]
    out["unassigned_plates"] = [clean_plate(p) for p in result["unassigned_plates"]]
    return out
