import requests
import pandas as pd
from datetime import datetime
import time
import os

CSV_FILE = "weather_data.csv"

RIVER_ZONES = {
    "Krishna": {
        "K1_Mahabaleshwar": (17.9237, 73.6586),
        "K2_Sangli_Almatti": (16.8544, 74.5642),
        "K3_Raichur_Kurnool": (16.2076, 77.3463),
        "K4_Nagarjuna_Sagar": (16.5750, 79.3167),
        "K5_Vijayawada_Delta": (16.5062, 80.6480),
    },

    "Narmada": {
        "N1_Amarkantak": (22.6747, 81.7590),
        "N2_Jabalpur": (23.1815, 79.9864),
        "N3_Omkareshwar": (22.2452, 76.1510),
        "N4_Bharuch": (21.7051, 72.9959),
    },

    "Kaveri": {
        "C1_Talakaveri": (12.3855, 75.4894),
        "C2_Kodagu": (12.3375, 75.8069),
        "C3_Mysuru_KRS": (12.2958, 76.6394),
        "C4_Mettur_Dam": (11.7870, 77.8008),
        "C5_Thanjavur_Delta": (10.7867, 79.1378),
    }
}


def get_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,"
        f"relative_humidity_2m,"
        f"precipitation,"
        f"surface_pressure,"
        f"wind_speed_10m,"
        f"cloud_cover"
    )

    try:
        response = requests.get(url, timeout=20)
        data = response.json()

        current = data.get("current", {})

        return {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "rainfall": current.get("precipitation"),
            "pressure": current.get("surface_pressure"),
            "wind_speed": current.get("wind_speed_10m"),
            "cloud_cover": current.get("cloud_cover"),
        }

    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None


def collect_weather():
    rows = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for river, zones in RIVER_ZONES.items():

        for zone_name, coords in zones.items():

            lat, lon = coords

            weather = get_weather(lat, lon)

            if weather:

                row = {
                    "timestamp": timestamp,
                    "river": river,
                    "zone": zone_name,
                    "latitude": lat,
                    "longitude": lon,
                    **weather
                }

                rows.append(row)

                print(
                    f"[OK] {river} | {zone_name} | "
                    f"Temp={weather['temperature']}°C | "
                    f"Rain={weather['rainfall']} mm"
                )

    return rows


def save_to_csv(rows):

    df = pd.DataFrame(rows)

    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, index=False)

    print(f"\nSaved {len(rows)} records to {CSV_FILE}")


def main():

    rows = collect_weather()

    if rows:
        save_to_csv(rows)


if __name__ == "__main__":
    main()
