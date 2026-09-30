import os
import requests
from dotenv import load_dotenv

load_dotenv(".env")

API_URL = "https://api.pulsescore.net/api/winamax/ice-hockey/events"


def get_nhl_goal_scorer_odds():
    api_key = os.getenv("PULSESCORE_API_KEY")

    if not api_key:
        raise RuntimeError("Clé PULSESCORE_API_KEY introuvable.")

    response = requests.get(
        API_URL,
        headers={"X-Secret": api_key},
        params={"page": 1, "limit": 200},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    results = []

    for event in data.get("events", []):
        if event.get("league", "").upper() != "NHL":
            continue

        scorer_market = None

        for market in event.get("markets", []):
            if (
                market.get("rawName", "").strip() == "Buteur"
                and market.get("isActive", True)
            ):
                scorer_market = market
                break

        if scorer_market is None:
            continue

        for selection in scorer_market.get("selections", []):
            if not selection.get("isActive", True):
                continue

            results.append({
                "event_id": event.get("eventId"),
                "away": event.get("away"),
                "home": event.get("home"),
                "player": selection.get("rawName", "").strip(),
                "odds": selection.get("odds"),
            })

    return results


if __name__ == "__main__":
    odds = get_nhl_goal_scorer_odds()

    print(f"{len(odds)} cotes buteur NHL trouvées.\n")

    for item in odds:
        print(
            f'{item["away"]} @ {item["home"]} | '
            f'{item["player"]} -> {item["odds"]}'
        )
