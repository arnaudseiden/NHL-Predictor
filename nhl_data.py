import requests
import json
from pathlib import Path
from datetime import date


BASE_URL = "https://api-web.nhle.com/v1"
STATS_URL = "https://api.nhle.com/stats/rest/en"


def get_seasons(reference_date=None):
    """Détermine automatiquement la saison NHL actuelle et précédente."""
    if reference_date is None:
        reference_date = date.today()

    year = reference_date.year

    # La nouvelle saison est considérée à partir de septembre
    if reference_date.month >= 9:
        start_year = year
    else:
        start_year = year - 1

    current = f"{start_year}{start_year + 1}"
    previous = f"{start_year - 1}{start_year}"

    return current, previous


def get_roster(team):
    url = f"{BASE_URL}/roster/{team}/current"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    players = []

    for player in response.json().get("forwards", []):
        players.append({
            "player_id": player["id"],
            "name": f'{player["firstName"]["default"]} {player["lastName"]["default"]}',
            "team": team,
            "position": player.get("positionCode", "")
        })

    return players


def toi_to_seconds(toi):
    if not toi:
        return 0

    minutes, seconds = toi.split(":")
    return int(minutes) * 60 + int(seconds)


def seconds_to_toi(seconds):
    seconds = round(seconds)
    minutes = seconds // 60
    remaining = seconds % 60
    return f"{minutes}:{remaining:02d}"


def get_game_log(player_id, season):
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"gamelog_{player_id}_{season}.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    url = (
        "https://"
        + f"api-web.nhle.com/v1/player/{player_id}/game-log/{season}/2"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    games = response.json().get("gameLog", [])

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(games, f)

    return games

def get_powerplay_games(player_id, seasons):
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    all_games = {}

    for season in seasons:
        cache_file = cache_dir / f"powerplay_{player_id}_{season}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                games = json.load(f)
        else:
            url = "https://" + "api.nhle.com/stats/rest/en/skater/powerplay"

            params = {
                "isAggregate": "false",
                "isGame": "true",
                "start": 0,
                "limit": 200,
                "cayenneExp": (
                    f"playerId={player_id} "
                    f"and seasonId={season} "
                    f"and gameTypeId=2"
                )
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            games = response.json().get("data", [])

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(games, f)

        for game in games:
            game_id = game.get("gameId")
            if game_id:
                all_games[game_id] = game

    return all_games

def get_last10(player_id, reference_date=None):
    """
    Construit un L10 dynamique.

    En début de saison, complète les matchs de la saison actuelle
    avec les derniers matchs de saison régulière de la saison précédente.
    """

    current_season, previous_season = get_seasons(reference_date)

    current_games = get_game_log(player_id, current_season)

    # Sécurité si une date historique est utilisée
    if reference_date:
        current_games = [
            g for g in current_games
            if g.get("gameDate", "") < reference_date.isoformat()
        ]

    games = current_games[:10]

    # S'il n'y a pas encore 10 matchs cette saison,
    # on complète avec la saison précédente.
    if len(games) < 10:
        previous_games = get_game_log(player_id, previous_season)

        needed = 10 - len(games)
        games.extend(previous_games[:needed])

    if not games:
        return {
            "GP": 0,
            "G": 0,
            "A": 0,
            "P": 0,
            "SOG": 0,
            "TOI": "0:00",
            "PP TOI": "0:00",
            "PPP": 0,
            "G/GP": 0,
            "P/GP": 0,
            "SOG/GP": 0
        }

    gp = len(games)

    goals = sum(g.get("goals", 0) for g in games)
    assists = sum(g.get("assists", 0) for g in games)
    points = sum(g.get("points", 0) for g in games)
    shots = sum(g.get("shots", 0) for g in games)

    total_toi = sum(
        toi_to_seconds(g.get("toi", "0:00"))
        for g in games
    )

    pp_games = get_powerplay_games(
        player_id,
        [current_season, previous_season]
    )

    total_pp_toi = 0
    pp_points = 0

    for game in games:
        pp = pp_games.get(game["gameId"])

        if pp:
            total_pp_toi += pp.get("ppTimeOnIce", 0) or 0
            pp_points += pp.get("ppPoints", 0) or 0

    return {
        "GP": gp,
        "G": goals,
        "A": assists,
        "P": points,
        "SOG": shots,
        "TOI": seconds_to_toi(total_toi / gp),
        "PP TOI": seconds_to_toi(total_pp_toi / gp),
        "PPP": pp_points,
        "G/GP": round(goals / gp, 2),
        "P/GP": round(points / gp, 2),
        "SOG/GP": round(shots / gp, 2)
    }


def get_team_last10(team, reference_date=None):
    results = []

    for player in get_roster(team):
        stats = get_last10(
            player["player_id"],
            reference_date
        )

        results.append({
            **player,
            **stats
        })

    return results


def get_season_player_stats(player_id, reference_date=None):
    current_season, previous_season = get_seasons(reference_date)

    # Matchs de la saison correspondant à la date
    games = get_game_log(player_id, current_season)

    if reference_date:
        games = [
            g for g in games
            if g.get("gameDate", "") < reference_date.isoformat()
        ]

    # Si aucun match de la saison actuelle n'avait encore été joué,
    # on utilise la saison précédente complète comme niveau de référence.
    if not games:
        season_used = previous_season
        games = get_game_log(player_id, previous_season)
    else:
        season_used = current_season

    if not games:
        return None

    gp = len(games)
    goals = sum(g.get("goals", 0) for g in games)
    assists = sum(g.get("assists", 0) for g in games)
    points = sum(g.get("points", 0) for g in games)
    shots = sum(g.get("shots", 0) for g in games)

    total_toi = sum(
        toi_to_seconds(g.get("toi", "0:00"))
        for g in games
    )

    # Données power play match par match
    pp_games = get_powerplay_games(player_id, [season_used])

    total_pp_toi = 0
    pp_goals = 0
    pp_points = 0

    for game in games:
        pp = pp_games.get(game["gameId"])

        if pp:
            total_pp_toi += pp.get("ppTimeOnIce", 0) or 0
            pp_goals += pp.get("ppGoals", 0) or 0
            pp_points += pp.get("ppPoints", 0) or 0

    return {
        "Season": season_used,
        "Season GP": gp,
        "Season G": goals,
        "Season P": points,
        "Season SOG": shots,

        "Season G/GP": round(goals / gp, 2),
        "Season P/GP": round(points / gp, 2),
        "Season SOG/GP": round(shots / gp, 2),

        "Season TOI": seconds_to_toi(total_toi / gp),
        "Season PP TOI": seconds_to_toi(total_pp_toi / gp),

        "Season PP Goals": pp_goals,
        "Season PP Points": pp_points
    }

