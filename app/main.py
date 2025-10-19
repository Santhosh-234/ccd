import io
import cv2
import torch
from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from app.utils import bgr_image_to_png_bytes, read_imagefile

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Adjusted import since your YOLO model is in ../models
from models.price_estimator import get_estimated_price, get_original_price

# Load YOLO
yolo_model = YOLO("models/damage_detector.pt")
print("YOLO model loaded successfully")


# ---------- Utility ----------
def parse_yolo_result(result):
    """Convert YOLO prediction result into structured data."""
    # Handle YOLO returning list
    if isinstance(result, list):
        result = result[0]

    # If result is None or invalid
    if result is None:
        return type("DamageResult", (), {
            "annotated_bgr": None,
            "damage_types": {},
            "severity": 0.0,
            "num_regions": 0,
            "total_damage_area": 0
        })()

    annotated_bgr = None
    try:
        annotated_bgr = result.plot()
    except Exception:
        pass  # Fail gracefully

    damage_types = {"dents": 0, "scratches": 0, "cracks": 0}
    num_regions, total_damage_area, severity = 0, 0, 0.0

    if hasattr(result, "boxes") and result.boxes is not None:
        xyxy_list = getattr(result.boxes, "xyxy", [])
        num_regions = len(xyxy_list)

        cls_list = getattr(result.boxes, "cls", [])
        for cls_idx in cls_list:
            cls_name = result.names[int(cls_idx)]
            if cls_name in damage_types:
                damage_types[cls_name] += 1

        for xyxy in xyxy_list:
            x1, y1, x2, y2 = xyxy
            total_damage_area += (x2 - x1) * (y2 - y1)

        if hasattr(result, "orig_img") and result.orig_img is not None:
            h, w = result.orig_img.shape[:2]
            severity = total_damage_area / (h * w) if h * w > 0 else 0.0

    return type("DamageResult", (), {
        "annotated_bgr": annotated_bgr,
        "damage_types": damage_types,
        "severity": severity,
        "num_regions": num_regions,
        "total_damage_area": total_damage_area
    })()


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile,
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...)
):
    try:
        # 1. Read image bytes
        img = read_imagefile(await file.read())

        # 2. Run YOLO
        raw_result = yolo_model(img)
        damage_result = parse_yolo_result(raw_result)

        # 3. Encode annotated image safely
        annotated_png = None
        if damage_result.annotated_bgr is not None:
            try:
                annotated_png = bgr_image_to_png_bytes(damage_result.annotated_bgr)
            except Exception as e:
                print(f"⚠️ Failed to encode annotated image: {e}")

        # 4. Get prices (handle None)
        original_price = get_original_price(make, model, year) or 0
        estimated_price = get_estimated_price(
            make, model, year, damage_result.severity
        ) or 0

        # 5. Prepare context
        context = {
            "request": request,
            "damage_types": damage_result.damage_types,
            "severity": round(damage_result.severity * 100, 2),
            "num_regions": damage_result.num_regions,
            "original_price": original_price,
            "estimated_price": estimated_price,
            "annotated_png": annotated_png,
        }

        return templates.TemplateResponse("result.html", context)

    except Exception as e:
        print(f"ERROR in analyze endpoint: {e}")
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": str(e)},
            status_code=500,
        )
