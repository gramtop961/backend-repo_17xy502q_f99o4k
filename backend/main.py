from __future__ import annotations

import math
import re
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Near Me API", version="1.0.0")

# CORS: allow all origins for sandbox; in production, restrict
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    lat: float
    lon: float
    radius_m: int = Field(2500, ge=200, le=20000)


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
    score: Optional[float] = None


@app.get("/test")
async def test() -> dict:
    return {"status": "ok"}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def keyword_tokens(q: str) -> list[str]:
    # simple tokenization: alphanum words longer than 2 chars
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", q) if len(t) > 2]


def build_overpass_query(lat: float, lon: float, radius: int, query: str) -> str:
    tokens = keyword_tokens(query)
    # map some common vibes to tag filters
    vibe_map = {
        "coffee": ["amenity=cafe"],
        "cafe": ["amenity=cafe"],
        "ramen": ["cuisine=ramen"],
        "tacos": ["cuisine=mexican", "cuisine=tacos"],
        "bar": ["amenity=bar", "amenity=pub"],
        "park": ["leisure=park"],
        "library": ["amenity=library"],
        "pizza": ["cuisine=pizza"],
        "restaurant": ["amenity=restaurant"],
    }
    tag_filters: list[str] = []
    for t in tokens:
        tag_filters.extend(vibe_map.get(t, []))

    # Overpass parts for nodes, ways, relations
    area = f"(around:{radius},{lat},{lon})"
    name_regex = None
    if tokens:
        name_regex = "|".join(re.escape(t) for t in tokens)

    def block(obj: str) -> str:
        parts = []
        if tag_filters:
            for tag in tag_filters:
                parts.append(f"{obj}[{tag}]{area};")
        # always also try name match
        if name_regex:
            parts.append(f"{obj}[name~\"{name_regex}\",i]{area};")
        # fallback to common POIs
        parts.append(f"{obj}[amenity~\"cafe|restaurant|bar|pub|fast_food\",i]{area};")
        return "".join(parts)

    q = f"""
    [out:json][timeout:25];
    (
      {block('node')}
      {block('way')}
      {block('relation')}
    );
    out center 60;
    """
    return q


async def fetch_overpass(query: str) -> dict:
    url = "https://overpass-api.de/api/interpreter"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, data={"data": query})
        resp.raise_for_status()
        return resp.json()


def extract_address(tags: dict) -> Optional[str]:
    parts = []
    for key in ["addr:housenumber", "addr:street", "addr:city"]:
        if tags.get(key):
            parts.append(tags[key])
    return ", ".join(parts) if parts else None


@app.post("/search", response_model=List[Place])
async def search(req: SearchRequest) -> List[Place]:
    try:
        ql = build_overpass_query(req.lat, req.lon, req.radius_m, req.query)
        data = await fetch_overpass(ql)
        elements = data.get("elements", [])
        results: list[Place] = []
        tokens = keyword_tokens(req.query)
        primary_token = tokens[0] if tokens else "places"

        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name:
                continue
            # resolve lat/lon for node or way/relation center
            if el.get("type") == "node":
                lat = el.get("lat")
                lon = el.get("lon")
            else:
                center = el.get("center") or {}
                lat = center.get("lat")
                lon = center.get("lon")
            if lat is None or lon is None:
                continue

            dist = haversine(req.lat, req.lon, float(lat), float(lon))

            # rudimentary category
            category = (
                tags.get("amenity")
                or tags.get("cuisine")
                or tags.get("shop")
                or "place"
            )

            # keyword score: +1 if token appears in name or cuisine
            text_blob = " ".join(
                [name, tags.get("cuisine", ""), tags.get("amenity", ""), tags.get("description", "")]
            ).lower()
            token_hits = sum(1 for t in tokens if t in text_blob)

            # distance score (closer better) and keyword bonus
            distance_score = max(0.0, 1.0 - (dist / max(1.0, req.radius_m)))  # 1 at center -> ~0 at edge
            score = distance_score * 0.7 + min(1.0, token_hits / 2) * 0.3

            photo = f"https://source.unsplash.com/480x320/?{primary_token},{category}"

            results.append(
                Place(
                    id=str(el.get("id")),
                    name=name,
                    address=extract_address(tags),
                    category=category,
                    rating=None,
                    distance_m=dist,
                    lat=float(lat),
                    lon=float(lon),
                    photo=photo,
                    score=score,
                )
            )

        # deduplicate by id and sort by score then distance
        uniq = {p.id: p for p in results}
        final = sorted(uniq.values(), key=lambda p: (-(p.score or 0), p.distance_m))
        return final[:40]

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error")


# Run: uvicorn main:app --host 0.0.0.0 --port 8000
