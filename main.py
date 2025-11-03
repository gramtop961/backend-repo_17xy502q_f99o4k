import os
import math
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field("", description="Natural language query")
    lat: float
    lon: float
    radius_m: int = Field(2000, description="Search radius in meters")


class Place(BaseModel):
    id: str
    name: str
    address: Optional[str] = None
    category: Optional[str] = None
    rating: Optional[float] = None
    distance_m: float
    lat: float
    lon: float
    photo: Optional[str] = None
    score: float


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_overpass_query(query: str, lat: float, lon: float, radius_m: int) -> str:
    # Basic categories to broaden search beyond name-only matches
    amenities = [
        "cafe", "restaurant", "bar", "pub", "fast_food", "ice_cream", "biergarten",
        "park", "cinema", "theatre", "library", "nightclub", "gym", "marketplace",
        "pharmacy", "supermarket", "bakery"
    ]
    # Escape quotes in query
    q = query.replace('"', '\\"')
    name_filter = f'["name"~"{q}",i]' if q.strip() else ""
    amenity_union = "".join([f'node["amenity"="{a}"](around:{radius_m},{lat},{lon});' for a in amenities])
    shop_union = "node[\"shop\"](around:{radius_m},{lat},{lon});"
    parts = f"(node{name_filter}(around:{radius_m},{lat},{lon});{amenity_union}{shop_union});"
    return f"[out:json][timeout:25];{parts}out center tags 100;"


def category_from_tags(tags: dict) -> Optional[str]:
    for key in ("amenity", "shop", "leisure", "tourism"):  # simple priority
        if key in tags:
            return tags.get(key)
    return None


def address_from_tags(tags: dict) -> Optional[str]:
    bits = []
    for k in ["addr:housenumber", "addr:street", "addr:city"]:
        if tags.get(k):
            bits.append(tags[k])
    return ", ".join(bits) if bits else tags.get("addr:full")


@app.post("/search", response_model=List[Place])
def search_places(req: SearchRequest):
    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        ql = build_overpass_query(req.query, req.lat, req.lon, req.radius_m)
        resp = requests.post(overpass_url, data={"data": ql}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        elements = data.get("elements", [])

        # Deduplicate by name+coords
        seen = set()
        results: List[Place] = []
        q_words = [w for w in req.query.lower().split() if len(w) > 2]
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name:
                continue
            lat = el.get("lat") or (el.get("center") or {}).get("lat")
            lon = el.get("lon") or (el.get("center") or {}).get("lon")
            if lat is None or lon is None:
                continue
            key = (name, round(lat, 6), round(lon, 6))
            if key in seen:
                continue
            seen.add(key)

            distance = haversine(req.lat, req.lon, lat, lon)
            category = category_from_tags(tags)

            # Keyword relevance: name/tags contains words
            text = (name + " " + " ".join([f"{k}:{v}" for k, v in tags.items()])).lower()
            keywords = sum(1 for w in q_words if w in text)

            # Rating: OSM lacks ratings, default to None, but add tiny neutral contribution
            rating = None

            # Score: closer is better; keyword match boosts significantly
            # Normalize distance contribution (0..1) within radius
            dist_norm = max(0.0, 1.0 - (distance / max(1, req.radius_m)))
            score = dist_norm * 0.7 + min(1.0, keywords / 3.0) * 0.3

            # Photo suggestion via Unsplash source (keyless)
            photo_query = category or "place"
            photo = f"https://source.unsplash.com/800x600/?{photo_query},{name.split()[0]}"

            results.append(Place(
                id=str(el.get("id")),
                name=name,
                address=address_from_tags(tags),
                category=category,
                rating=rating,
                distance_m=round(distance, 2),
                lat=lat,
                lon=lon,
                photo=photo,
                score=round(score, 4),
            ))

        # Rank by score then distance
        results.sort(key=lambda r: (-r.score, r.distance_m))
        return results[:50]  # return more to allow client to pick top 5 visually
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Near Me backend running"}


@app.get("/test")
def test_database():
    """Simple OK signal and environment markers"""
    return {
        "backend": "OK",
        "database_url_set": bool(os.getenv("DATABASE_URL")),
        "database_name_set": bool(os.getenv("DATABASE_NAME")),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
