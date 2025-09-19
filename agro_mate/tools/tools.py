import requests
from datetime import datetime

def get_current_time() -> dict:
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def get_location() -> dict:
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return {
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country"),
            "loc": data.get("loc")  # Latitude,Longitude
        }
    except Exception as e:
        return {"error": str(e)}
