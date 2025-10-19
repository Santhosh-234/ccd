from typing import Optional


MULTIPLIERS = {
    "minor": 0.95,    # 5% reduction for minor damage
    "moderate": 0.85, # 15% reduction for moderate damage  
    "severe": 0.65,  # 35% reduction for severe damage
}


def apply_damage_multiplier(base_price: Optional[float], severity: str) -> Optional[float]:
    if base_price is None:
        return None
    factor = MULTIPLIERS.get(severity, 0.9)
    return round(base_price * factor, 2)


