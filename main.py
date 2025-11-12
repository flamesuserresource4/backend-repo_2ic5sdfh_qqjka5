import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Travel Chatbot Intent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IntentDefinition(BaseModel):
    name: str
    description: str
    backend_command: str
    keywords: List[str]
    sample_utterances: List[str]
    required_slots: List[str] = []


# Simple intent catalog for a travel agency
INTENTS: List[IntentDefinition] = [
    IntentDefinition(
        name="search_flights",
        description="Find available flights between two cities on certain dates.",
        backend_command="SearchFlights",
        keywords=["flight", "fly", "ticket", "air", "plane", "one-way", "round trip"],
        sample_utterances=[
            "Find me flights from NYC to Paris next month",
            "I need a ticket from San Francisco to Tokyo on June 12",
            "Show round trip flights LA to Miami leaving Friday coming back Sunday",
        ],
        required_slots=["origin", "destination", "departure_date"],
    ),
    IntentDefinition(
        name="book_hotel",
        description="Search or book hotels in a given city with date range.",
        backend_command="BookHotel",
        keywords=["hotel", "stay", "room", "resort", "nights", "check-in", "check out"],
        sample_utterances=[
            "Book a hotel in Rome from May 2 to May 5",
            "Find a 4-star room in Chicago this weekend",
            "I need a family room near Times Square",
        ],
        required_slots=["city", "check_in", "check_out"],
    ),
    IntentDefinition(
        name="car_rental",
        description="Reserve a rental car in a city for specific dates.",
        backend_command="ReserveCar",
        keywords=["car", "rental", "drive", "pickup", "dropoff", "vehicle"],
        sample_utterances=[
            "I need a rental car in Denver from July 3 to July 8",
            "Rent a SUV in Orlando tomorrow",
            "Book a car pickup in Boston, dropoff in New York",
        ],
        required_slots=["city", "pickup_date", "dropoff_date"],
    ),
    IntentDefinition(
        name="travel_packages",
        description="Explore curated travel packages to popular destinations.",
        backend_command="GetPackages",
        keywords=["package", "deal", "vacation", "bundle", "all-inclusive"],
        sample_utterances=[
            "Show me vacation packages to Hawaii",
            "Any all-inclusive deals for Cancun?",
        ],
        required_slots=["destination"],
    ),
]


class ParseRequest(BaseModel):
    text: str


class ParseResponse(BaseModel):
    intent: Optional[str]
    confidence: float
    entities: Dict[str, Any]
    matched_keywords: List[str]
    candidates: List[Dict[str, Any]]


def simple_ner(text: str) -> Dict[str, Any]:
    # Extremely naive entity extraction for demo purposes
    t = text.lower()
    entities: Dict[str, Any] = {}
    # City guess by prepositions
    preps = ["to", "from", "in", "at"]
    words = t.replace(",", " ").replace("\n", " ").split()
    for i, w in enumerate(words):
        if w in ("nyc", "paris", "rome", "tokyo", "miami", "la", "denver", "orlando", "boston", "chicago", "hawaii", "cancun", "san", "francisco", "new", "york"):
            # Join common multiword city names
            if w == "san" and i + 1 < len(words) and words[i + 1] == "francisco":
                entities.setdefault("origin", "san francisco")
            elif w == "new" and i + 1 < len(words) and words[i + 1] == "york":
                entities.setdefault("destination", "new york")
            else:
                # Fill first missing of origin/destination/city heuristically
                if "from" in words[max(0, i - 3) : i] and "origin" not in entities:
                    entities["origin"] = w
                elif "to" in words[max(0, i - 3) : i] and "destination" not in entities:
                    entities["destination"] = w
                else:
                    entities.setdefault("city", w)

    # Dates (very naive)
    if any(x in t for x in ["today", "tomorrow"]):
        if "tomorrow" in t:
            entities.setdefault("departure_date", "tomorrow")
        else:
            entities.setdefault("departure_date", "today")

    # Trip type
    if "round" in t and "trip" in t:
        entities["trip_type"] = "round_trip"

    return entities


def score_intent(text: str, intent: IntentDefinition) -> Dict[str, Any]:
    t = text.lower()
    matched = [k for k in intent.keywords if k in t]
    score = min(1.0, len(matched) / max(3, len(intent.keywords)))
    # Boost if any sample utterance shares bigrams
    boost = 0.0
    for sample in intent.sample_utterances:
        if any(word in t for word in sample.lower().split()):
            boost = max(boost, 0.2)
    score = min(1.0, score + boost)
    return {"name": intent.name, "score": round(score, 3), "matched": matched}


@app.get("/")
def read_root():
    return {"message": "Travel Chatbot Intent API is running"}


@app.get("/intents", response_model=List[IntentDefinition])
def get_intents():
    return INTENTS


@app.post("/nlu/parse", response_model=ParseResponse)
def parse(req: ParseRequest):
    candidates = sorted(
        [score_intent(req.text, it) for it in INTENTS],
        key=lambda x: x["score"],
        reverse=True,
    )
    best = candidates[0] if candidates else {"name": None, "score": 0.0, "matched": []}
    entities = simple_ner(req.text)
    return {
        "intent": best["name"],
        "confidence": best["score"],
        "entities": entities,
        "matched_keywords": best["matched"],
        "candidates": candidates,
    }


class ExecuteRequest(BaseModel):
    intent: str
    parameters: Dict[str, Any] = {}


class ExecuteResponse(BaseModel):
    intent: str
    command: str
    status: str
    result: Dict[str, Any]


@app.post("/execute", response_model=ExecuteResponse)
def execute(req: ExecuteRequest):
    # Find mapping
    intent = next((i for i in INTENTS if i.name == req.intent), None)
    if not intent:
        return {
            "intent": req.intent,
            "command": "Unknown",
            "status": "error",
            "result": {"message": "Unknown intent"},
        }

    # Validate required slots
    missing = [s for s in intent.required_slots if s not in req.parameters]
    if missing:
        return {
            "intent": intent.name,
            "command": intent.backend_command,
            "status": "incomplete",
            "result": {"missing_parameters": missing},
        }

    # Simulate backend command execution
    if intent.name == "search_flights":
        res = {
            "itineraries": [
                {
                    "airline": "ACME Air",
                    "price": 542.35,
                    "currency": "USD",
                    "stops": 1,
                    "duration": "11h 20m",
                },
                {
                    "airline": "SkyJet",
                    "price": 618.10,
                    "currency": "USD",
                    "stops": 0,
                    "duration": "8h 50m",
                },
            ]
        }
    elif intent.name == "book_hotel":
        res = {
            "hotels": [
                {"name": "Grand Plaza", "stars": 4, "price": 189},
                {"name": "City Inn", "stars": 3, "price": 129},
            ]
        }
    elif intent.name == "car_rental":
        res = {
            "cars": [
                {"make": "EcoCar", "model": "Compact", "price_per_day": 38},
                {"make": "SUVPlus", "model": "SUV", "price_per_day": 64},
            ]
        }
    elif intent.name == "travel_packages":
        res = {
            "packages": [
                {"destination": req.parameters.get("destination"), "nights": 5, "price": 1299},
                {"destination": req.parameters.get("destination"), "nights": 7, "price": 1599},
            ]
        }
    else:
        res = {"message": "No executor implemented"}

    return {
        "intent": intent.name,
        "command": intent.backend_command,
        "status": "ok",
        "result": res,
    }


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
