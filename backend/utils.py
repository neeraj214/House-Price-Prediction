import re


def clean_bhk(bhk_str: str) -> int:
    """Extract integer BHK from strings like '2 BHK' or '3 Bedroom'."""
    if not isinstance(bhk_str, str):
        return int(bhk_str)
    match = re.search(r"(\d+)", bhk_str)
    return int(match.group(1)) if match else 1


def clean_area(area_str: str) -> float:
    """Handle ranges like '1100-1300' by taking the average."""
    if not isinstance(area_str, str):
        return float(area_str)
    if "-" in area_str:
        parts = area_str.split("-")
        return (float(parts[0].strip()) + float(parts[1].strip())) / 2
    return float(area_str)


def format_indian_currency(amount: float) -> str:
    """Format price in Lakhs or Crores."""
    if amount >= 10000000:
        return f"₹ {(amount / 10000000):.2f} Crore"
    elif amount >= 100000:
        return f"₹ {(amount / 100000):.2f} Lakhs"
    return f"₹ {int(amount):,}"


def get_feature_influence(input_data: dict) -> str:
    """Heuristic for feature importance since this is a clean inference project."""
    # Location and Area are typically the highest drivers in Indian real estate
    return "Location and Area had the highest impact on this estimate."
