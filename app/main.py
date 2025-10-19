from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from typing import Optional
import os

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
    # Define a default image filename and a safe damage_types dict
    # This ensures these variables are available even if the try block fails early
    img_filename = None
    safe_damage_types = {}
    
    try:
        image_bytes = await image.read()

        damage_result = detect_damage(image_bytes)
        
        # Prepare image assets
        annotated_png = bgr_image_to_png_bytes(damage_result.annotated_bgr)
        img_filename = generate_filename(prefix="annotated_", ext=".png")
        img_path = os.path.join(STATIC_DIR, img_filename)
        save_image_bytes(img_path, annotated_png)

        # Ensure damage_types is populated safely, in case the result from detect_damage 
        # is missing keys, though the template fix is the real solution.
        # This will be used if the try block succeeds.
        safe_damage_types = damage_result.damage_types
        # Fallback for known template expectations (e.g., if 'cracks' is missing)
        # However, relying on the template fix is cleaner. We will assume 
        # damage_result.damage_types is what we need if the try block runs successfully.

        listings = fetch_market_prices(make=make, model=model, year=year, mileage=mileage, city=city)
        prices = [l.price for l in listings if l.price is not None]
        avg_price = sum(prices) / len(prices) if prices else None
        adjusted_price = apply_damage_multiplier(avg_price, damage_result.severity) if avg_price is not None else None
        
        # Get original price when car was new
        from app.pricing.fetch import _get_original_price
        original_price = _get_original_price(make, model, year)
        
        # Filter and sort listings by relevance and price proximity
        filtered_listings = []
        for listing in listings:
            if listing.price is not None:
                # Calculate price difference from our calculated price
                if adjusted_price:
                    price_diff = abs(listing.price - adjusted_price)
                    listing.price_diff = price_diff
                else:
                    listing.price_diff = float('inf')
                filtered_listings.append(listing)
        
        # Sort by price proximity to our calculated price
        filtered_listings.sort(key=lambda x: x.price_diff)
        
        # Take top 6 most relevant listings
        relevant_listings = filtered_listings[:6]

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "image_url": f"/static/{img_filename}",
                "severity": damage_result.severity,
                "num_regions": damage_result.num_regions,
                "damage_types": safe_damage_types, # Use the result from damage detection
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
        import traceback
        traceback.print_exc()
        
        # --- FIX APPLIED HERE ---
        # The template requires 'cracks', so ensure the fallback dictionary
        # includes this key with a default value (0) to prevent the UndefinedError.
        fallback_damage_types = {
            "dents": 0,
            "scratches": 0,
            "cracks": 0, # <-- This is the key that was missing when 'damage_types' was just {}
        }
        
        # Fallback: show error on the results page rather than returning 500
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "image_url": f"/static/{img_filename}" if img_filename else None,
                "severity": "error",
                "num_regions": 0,
                "damage_types": fallback_damage_types, # Pass the safe, complete dictionary
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