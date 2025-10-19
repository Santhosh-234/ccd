from pydantic import BaseModel
from typing import List, Optional


class ListingModel(BaseModel):
    source: str
    title: str
    price: Optional[float]
    url: Optional[str]


class AnalyzeResponse(BaseModel):
    image_url: str
    severity: str
    num_regions: int
    avg_price: Optional[float]
    adjusted_price: Optional[float]
    listings: List[ListingModel]


