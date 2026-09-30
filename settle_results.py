import pandas as pd
import requests
from pathlib import Path

HISTORY_FILE = Path("data/betting_history.csv")


def get_game_result(game_id):
    """Retourne les buteurs et les joueurs ayant participé si le match est terminé."""

    pbp_url = (
        f"https://api-web.nhle.com/v1/gamecenter/"
        f"{game_id}/play-by-play"
    )

    box_url = (
        f"https://api-web.nhle.com/v1/gamecenter/"
        f"{game_id}/boxscore"
    )

    # Résultat / buteurs
    response = requests.get(pbp_url, timeout=20)
    response.raise_for_status()
    data = response.json()

    game_state = data.get("gameState")

    if game_state not in ["FINAL", "OFF"]:
        return None

    scorers = set()

    for play in data.get("plays", []):
        if play.get("typeDescKey") == "goal":
            player_id = play.get("details", {}).get(
                "scoringPlayerId"
            )

            if player_id is not None:
                scorers.add(int(player_id))

    # Joueurs ayant réellement participé
    response = requests.get(box_url, timeout=20)
    response.raise_for_status()
    boxscore = response.json()

    participants = set()

    player_stats = boxscore.get(
        "playerByGameStats", {}
    )

    for team in ["awayTeam", "homeTeam"]:
        team_stats = player_stats.get(team, {})

        for category in ["forwards", "defense"]:
            for player in team_stats.get(category, []):
                player_id = player.get("playerId")

                if player_id is not None:
                    participants.add(int(player_id))

    return scorers, participants


def settle_history():
    if not HISTORY_FILE.exists():
        print("Aucun historique trouvé.")
        return

    df = pd.read_csv(HISTORY_FILE)

    pending = df[df["result"].isna()].copy()

    if pending.empty:
        print("Aucun résultat en attente.")
        return

    print(f"{len(pending)} pari(s) en attente.")

    for game_id in pending["game_id"].unique():

        try:
            game_result = get_game_result(int(game_id))

            if game_result is None:
                print(
                    f"Match {game_id} : "
                    "pas encore terminé"
                )
                continue

            scorers, participants = game_result

            mask = (
                (df["game_id"] == game_id)
                & (df["result"].isna())
            )

            for index in df[mask].index:
                player_id = int(
                    df.at[index, "player_id"]
                )

                if player_id not in participants:
                    # Joueur absent / scratch :
                    # le pari reste non réglé.
                    continue

                df.at[index, "result"] = (
                    1 if player_id in scorers else 0
                )

            print(
                f"Match {game_id} : "
                f"{len(scorers)} buteur(s), "
                f"{len(participants)} joueurs de champ"
            )

        except Exception as e:
            print(
                f"Match {game_id} non réglé : {e}"
            )

    df.to_csv(HISTORY_FILE, index=False)

    print("Historique mis à jour.")


if __name__ == "__main__":
    settle_history()
