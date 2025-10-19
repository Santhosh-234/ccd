from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from typing import Optional
import os
import traceback
import numpy as np

from .cv.damage import detect_with_yolo as detect_damage
from .pricing.fetch import fetch_market_prices
from .services.valuation import apply_damage_multiplier
from .utils import ensure_dirs, save_image_bytes, bgr_image_to_png_bytes, generate_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

ensure_dirs([STATIC_DIR, TEMPLATES_DIR])

app = FastAPI(title="Car Damage Detection & Valuation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ---------------------- Helper: Wrap DLResult ----------------------
def parse_yolo_result(result):
    """Convert Ultralytics DLResult (or list) into DamageResult object."""
    # If the result is a list, take the first element
    if isinstance(result, list):
        result = result[0]

    annotated_bgr = result.plot() if hasattr(result, "plot") else None

    damage_types = {"dents": 0, "scratches": 0, "cracks": 0}
    num_regions = 0
    total_damage_area = 0
    severity = 0

    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        num_regions = len(result.boxes.xyxy)
        for cls in result.boxes.cls:
            cls_name = result.names[int(cls)]
            if cls_name in damage_types:
                damage_types[cls_name] += 1

        for xyxy in result.boxes.xyxy:
            x1, y1, x2, y2 = xyxy
            total_damage_area += (x2 - x1) * (y2 - y1)

        if hasattr(result, "orig_img") and result.orig_img is not None:
            h, w = result.orig_img.shape[:2]
            severity = total_damage_area / (h * w)

    return type("DamageResult", (), {
        "annotated_bgr": annotated_bgr,
        "damage_types": damage_types,
        "severity": severity,
        "num_regions": num_regions,
        "total_damage_area": total_damage_area
    })()



# ---------------------- Routes ----------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    mileage: int = Form(...),
    city: Optional[str] = Form(None),
):
    img_filename = None

    try:
        image_bytes = await image.read()

        # Run YOLO detection
        raw_result = detect_damage(image_bytes)
        damage_result = parse_yolo_result(raw_result)

        # Prepare annotated image
        annotated_png = bgr_image_to_png_bytes(damage_result.annotated_bgr)
        img_filename = generate_filename(prefix="annotated_", ext=".png")
        img_path = os.path.join(STATIC_DIR, img_filename)
        save_image_bytes(img_path, annotated_png)

        # Fetch market prices
        listings = fetch_market_prices(make=make, model=model, year=year, mileage=mileage, city=city)
        prices = [l.price for l in listings if l.price is not None]
        avg_price = sum(prices) / len(prices) if prices else None
        adjusted_price = apply_damage_multiplier(avg_price, damage_result.severity) if avg_price is not None else None

        # Get original price
        from app.pricing.fetch import _get_original_price
        original_price = _get_original_price(make, model, year)

        # Filter and sort listings by price proximity
        filtered_listings = []
        for listing in listings:
            if listing.price is not None:
                price_diff = abs(listing.price - adjusted_price) if adjusted_price else float('inf')
                listing.price_diff = price_diff
                filtered_listings.append(listing)

        filtered_listings.sort(key=lambda x: x.price_diff)
        relevant_listings = filtered_listings[:6]

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "image_url": f"/static/{img_filename}",
                "severity": damage_result.severity,
                "num_regions": damage_result.num_regions,
                "damage_types": damage_result.damage_types,
                "total_damage_area": damage_result.total_damage_area,
                "avg_price": avg_price,
                "adjusted_price": adjusted_price,
                "original_price": original_price,
                "listings": relevant_listings,
                "make": make,
                "model": model,
                "year": year,
                "mileage": mileage,
                "city": city,
            },
        )
    except Exception as e:
        print(f"ERROR in analyze endpoint: {e}")
        traceback.print_exc()

        # Safe fallback for template
        fallback_damage_types = {"dents": 0, "scratches": 0, "cracks": 0}

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "image_url": f"/static/{img_filename}" if img_filename else None,
                "severity": "error",
                "num_regions": 0,
                "damage_types": fallback_damage_types,
                "total_damage_area": 0,
                "avg_price": None,
                "adjusted_price": None,
                "original_price": None,
                "listings": [],
                "make": make,
                "model": model,
                "year": year,
                "mileage": mileage,
                "city": city,
                "error": str(e),
            },
            status_code=200,
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
