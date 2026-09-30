import requests
from datetime import date
from nhl_data import get_seasons

STATS_URL = "https://api.nhle.com/stats/rest/en"

TEAM_IDS = {
    "MTL": 1, "TOR": 5, "BOS": 6, "NYR": 10,
    "CHI": 11, "DET": 12, "LAK": 14, "DAL": 15,
    "PHI": 16, "PIT": 17, "STL": 18, "BUF": 19,
    "VAN": 20, "CGY": 21, "NYI": 22, "NJD": 23,
    "WSH": 24, "EDM": 25, "CAR": 26, "COL": 27,
    "SJS": 29, "OTT": 30, "TBL": 31, "ANA": 32,
    "FLA": 33, "NSH": 34, "WPG": 35, "CBJ": 36,
    "MIN": 37, "VGK": 38, "SEA": 39, "UTA": 40
}


def get_team_games(team, season):
    team_id = TEAM_IDS[team]

    url = f"{STATS_URL}/team/summary"

    params = {
        "isAggregate": "false",
        "isGame": "true",
        "start": 0,
        "limit": 100,
        "cayenneExp": (
            f"franchiseId={team_id} "
            f"and seasonId={season} "
            f"and gameTypeId=2"
        )
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    games = response.json().get("data", [])

    return sorted(
        games,
        key=lambda x: x.get("gameDate", ""),
        reverse=True
    )


def aggregate_games(games):
    if not games:
        return None

    gp = len(games)

    ga = sum(g.get("goalsAgainst", 0) for g in games)
    sa = sum(g.get("shotsAgainstPerGame", 0) for g in games)

    # PK% récent : moyenne des valeurs disponibles.
    # On améliorera ensuite ce calcul avec les opportunités PK.
    pk_values = [
        g.get("penaltyKillPct")
        for g in games
        if g.get("penaltyKillPct") is not None
    ]

    pk = (
        sum(pk_values) / len(pk_values)
        if pk_values else 0
    )

    return {
        "GP": gp,
        "GA/GP": ga / gp,
        "SA/GP": sa / gp,
        "PK%": pk
    }


def get_defense_stats(team, reference_date=None):
    current_season, previous_season = get_seasons(reference_date)

    current_games = get_team_games(team, current_season)

    # Pour un backtest, aucun match joué le jour du pronostic
    # ou après cette date ne doit être utilisé.
    if reference_date:
        current_games = [
            g for g in current_games
            if g.get("gameDate", "") < reference_date.isoformat()
        ]

    # Début de saison :
    # saison précédente = niveau de fond + L10 précédent.
    if not current_games:
        season_used = previous_season
        season_games = get_team_games(team, previous_season)
        recent_games = season_games[:10]

    else:
        season_used = current_season
        season_games = current_games
        recent_games = current_games[:10]

    season = aggregate_games(season_games)
    recent = aggregate_games(recent_games)

    if not season or not recent:
        return None

    season_ga = season["GA/GP"]
    recent_ga = recent["GA/GP"]

    season_sa = season["SA/GP"]
    recent_sa = recent["SA/GP"]

    season_pk = season["PK%"]
    recent_pk = recent["PK%"]

    # Pondérations issues du modèle Excel
    ga_mix = 0.50 * season_ga + 0.50 * recent_ga
    sa_mix = 0.70 * season_sa + 0.30 * recent_sa
    pk_mix = 0.75 * season_pk + 0.25 * recent_pk

    return {
        "Team": team,
        "Season": season_used,
        "Season GP": season["GP"],

        "Season GA/GP": round(season_ga, 2),
        "Recent GA/GP": round(recent_ga, 2),
        "GA/GP Mix": round(ga_mix, 2),

        "Season SA/GP": round(season_sa, 2),
        "Recent SA/GP": round(recent_sa, 2),
        "SA/GP Mix": round(sa_mix, 2),

        "Season PK%": round(season_pk * 100, 1),
        "Recent PK%": round(recent_pk * 100, 1),
        "PK% Mix": round(pk_mix * 100, 1),

        "Recent Games": len(recent_games)
    }
