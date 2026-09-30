from pathlib import Path
import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import date

from nhl_data import get_team_last10, get_season_player_stats
from team_stats import get_defense_stats

from difflib import SequenceMatcher
import unicodedata


def normalize_player_name(name):
    """Normalise un nom pour comparer NHL et Winamax."""
    name = str(name).lower().strip()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = " ".join(name.split())
    return name


def find_winamax_odds(player_name, odds_dict):
    """Recherche exacte puis rapprochement prudent des noms."""
    target = normalize_player_name(player_name)

    # Correspondance exacte après normalisation
    for winamax_name, odds in odds_dict.items():
        if normalize_player_name(winamax_name) == target:
            return odds

    # Petite variante orthographique seulement
    best_odds = None
    best_score = 0

    for winamax_name, odds in odds_dict.items():
        candidate = normalize_player_name(winamax_name)

        # Le nom de famille doit être identique
        if target.split()[-1] != candidate.split()[-1]:
            continue

        score = SequenceMatcher(None, target, candidate).ratio()

        if score > best_score:
            best_score = score
            best_odds = odds

    if best_score >= 0.90:
        return best_odds

    return None

from settle_results import settle_history
from winamax_odds import get_nhl_goal_scorer_odds


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="NHL Predictor",
    page_icon="🏒",
    layout="wide"
)

st.title("🏒 NHL Predictor")

if st.button("🔄 Mettre à jour les résultats"):
    settle_history()
    st.success("Vérification des résultats terminée.")
    st.rerun()

st.caption("Analyse automatique des buteurs NHL")


# ---------------------------------------------------------
# CHARGEMENT DU MODELE V3
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    model = joblib.load("models/goal_v3.joblib")
    calibrator = joblib.load("models/goal_calibrator.joblib")

    model_v4 = joblib.load("models/goal_v4.joblib")
    calibrator_v4 = joblib.load("models/goal_v4_calibrator.joblib")

    return model, calibrator, model_v4, calibrator_v4


model, calibrator, model_v4, calibrator_v4 = load_models()


# ---------------------------------------------------------
# CALENDRIER DU JOUR
# ---------------------------------------------------------

today = date.today().isoformat()

st.subheader(f"📅 Matchs NHL — {today}")

url = f"https://api-web.nhle.com/v1/schedule/{today}"

games = []

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    for week in data.get("gameWeek", []):
        if week.get("date") == today:

            for game in week.get("games", []):
                away = game["awayTeam"]["abbrev"]
                home = game["homeTeam"]["abbrev"]

                games.append({
                    "Extérieur": away,
                    "Domicile": home,
                    "Match": f"{away} @ {home}",
                    "Game ID": game["id"]
                })

    if games:
        df_games = pd.DataFrame(games)

        st.success(
            f"✅ {len(df_games)} matchs récupérés automatiquement"
        )

        st.dataframe(
            df_games,
            width="stretch",
            hide_index=True
        )

    else:
        st.warning(
            "Aucun match NHL trouvé pour aujourd'hui."
        )

except Exception as e:
    st.error("Impossible de récupérer les données NHL.")
    st.code(str(e))


# ---------------------------------------------------------
# FONCTION ANALYSE V3
# ---------------------------------------------------------

def analyse_team(team, opponent=None):

    players = get_team_last10(team)
    results = []

    # Défense de l'adversaire pour V4
    defense = get_defense_stats(opponent) if opponent else None

    if defense:
        opp_ga_mix = defense.get("GA/GP Mix", 0)
        opp_sa_mix = defense.get("SA/GP Mix", 0)
        opp_pk_mix = defense.get("PK% Mix", 0)
    else:
        opp_ga_mix = 0
        opp_sa_mix = 0
        opp_pk_mix = 0

    def toi_to_seconds(value):
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return float(value)

        parts = str(value).split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])

        return 0

    for player in players:

        try:
            season = get_season_player_stats(
                player["player_id"]
            )

            if season is None:
                season = {}

            season_gp = season.get("Season GP", 0)
            season_g = season.get("Season G", 0)
            season_gpg = season.get("Season G/GP", 0)

            # ---------------- V3 ----------------

            features_v3 = pd.DataFrame([{
                "l10_sogpg": player["SOG/GP"],
                "l10_pg": player["P/GP"],
                "season_gpg": season_gpg
            }])

            v3 = model.predict_proba(
                features_v3
            )[0, 1]

            calibrated_v3 = calibrator.predict_proba(
                pd.DataFrame({
                    "V3": [v3]
                })
            )[0, 1]

            # ---------------- V4 ----------------

            features_v4 = pd.DataFrame([{
                "l10_gpg": player["G/GP"],
                "l10_pg": player["P/GP"],
                "l10_sogpg": player["SOG/GP"],
                "l10_pp_toi": toi_to_seconds(
                    player.get("PP TOI", "0:00")
                ),
                "l10_ppp": player.get("PPP", 0),
                "season_gpg": season_gpg,
                "season_sogpg": season.get("Season SOG/GP", 0),
                "season_pp_toi": toi_to_seconds(
                    season.get("Season PP TOI", "0:00")
                ),
                "season_pp_goals": season.get("Season PP Goals", 0),
                "season_pp_points": season.get("Season PP Points", 0),
                "opp_ga_mix": opp_ga_mix,
                "opp_sa_mix": opp_sa_mix,
                "opp_pk_mix": opp_pk_mix
            }])

            v4 = model_v4.predict_proba(
                features_v4
            )[0, 1]

            calibrated_v4 = calibrator_v4.predict_proba(
                pd.DataFrame({
                    "V4": [v4]
                })
            )[0, 1]

            fair_odds_v4 = (
                1 / calibrated_v4
                if calibrated_v4 > 0
                else None
            )

            results.append({
                "player_id": player["player_id"],
                "Joueur": player["name"],
                "Pos": player["position"],
                "GP saison": season_gp,
                "G saison": season_g,
                "G/GP saison": season_gpg,
                "L10 G/GP": player["G/GP"],
                "L10 P/GP": player["P/GP"],
                "L10 SOG/GP": player["SOG/GP"],

                "V3 brut": v3,
                "Proba V3": calibrated_v3,

                "V4 brut": v4,
                "Proba V4": calibrated_v4,

                # V4 devient le modèle utilisé par l'application
                "Proba but": calibrated_v4,
                "Cote juste": fair_odds_v4,

                # V3 reste disponible pour comparaison
                "Cote juste V4": fair_odds_v4
            })

        except Exception as e:
            print(
                "Erreur joueur",
                player["name"],
                e
            )

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(
            "Proba but",
            ascending=False
        )

    return df


# ---------------------------------------------------------
# PARIS DU JOUR - TOUS LES MATCHS
# ---------------------------------------------------------

st.divider()
st.header("🌎 Paris du jour")

if games:

    if st.button("🚀 Analyser tous les matchs du jour"):

        daily_results = []

        progress = st.progress(0)
        status = st.empty()

        total_teams = len(games) * 2
        done = 0

        for game in games:

            match_name = game["Match"]
            game_id = game["Game ID"]

            for team in [
                game["Extérieur"],
                game["Domicile"]
            ]:

                status.write(
                    f"Analyse de {team} — {match_name}"
                )

                opponent = (
                    game["Domicile"]
                    if team == game["Extérieur"]
                    else game["Extérieur"]
                )

                team_df = analyse_team(team, opponent)

                # Vue quotidienne : Top 3 V4 par équipe
                team_df = team_df.sort_values(
                    "Proba V4",
                    ascending=False
                ).head(3)

                if not team_df.empty:
                    team_df = team_df.copy()
                    team_df["Match"] = match_name
                    team_df["Game ID"] = game_id
                    team_df["Equipe"] = team

                    daily_results.append(team_df)

                done += 1
                progress.progress(done / total_teams)

        progress.empty()
        status.empty()

        if daily_results:

            daily_df = pd.concat(
                daily_results,
                ignore_index=True
            )

            daily_df = daily_df.sort_values(
                "Proba but",
                ascending=False
            )

            st.session_state["daily_df"] = daily_df

if "daily_df" in st.session_state:

    daily_df = st.session_state["daily_df"].copy()

    # Conversion pour affichage en pourcentage
    daily_df["Proba but"] = daily_df["Proba but"] * 100
    daily_df["Proba V3"] = daily_df["Proba V3"] * 100
    daily_df["Proba V4"] = daily_df["Proba V4"] * 100

    st.success(
        f"✅ {len(daily_df)} joueurs analysés"
    )

    history_file_daily = Path("data/betting_history.csv")

    # Récupération automatique des cotes buteur Winamax
    try:
        @st.cache_data(ttl=600)
        def load_winamax_odds():
            return get_nhl_goal_scorer_odds()

        winamax_data = load_winamax_odds()

        winamax_odds = {
            item["player"].replace("\xa0", " ").strip(): float(item["odds"])
            for item in winamax_data
            if item.get("odds") is not None
        }

        daily_df["Cote Winamax"] = daily_df["Joueur"].apply(
            lambda player: find_winamax_odds(
                player,
                winamax_odds
            )
        )

        found = daily_df["Cote Winamax"].notna().sum()

        st.caption(
            f"🎯 Cotes Winamax récupérées automatiquement : "
            f"{found}/{len(daily_df)} joueurs"
        )

    except Exception as e:
        st.warning(
            "Impossible de récupérer automatiquement les cotes Winamax. "
            "La saisie manuelle reste disponible."
        )

        daily_df["Cote Winamax"] = None

        print("Erreur récupération Winamax :", e)

    daily_edited = st.data_editor(
        daily_df[[
            "Match",
            "Equipe",
            "Joueur",
            "Pos",
            "Proba but",
            "Cote juste",
            "Cote Winamax"
        ]],
        column_config={
            "Proba but": st.column_config.NumberColumn(
                format="%.1f%%"
            ),
            "Cote juste": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Cote Winamax": st.column_config.NumberColumn(
                "Cote Winamax",
                min_value=1.01,
                step=0.05,
                format="%.2f",
                help="Saisir ici la cote buteur proposée par Winamax"
            )
        },
        disabled=[
            "Match",
            "Equipe",
            "Joueur",
            "Pos",
            "Proba but",
            "Cote juste"
        ],
        width="stretch",
        hide_index=True,
        key="daily_odds_editor"
    )


# Calcul Value / EV pour les cotes renseignées
    daily_value = daily_edited.copy()

    daily_value["Cote Winamax"] = pd.to_numeric(
        daily_value["Cote Winamax"],
        errors="coerce"
    )

    daily_value = daily_value[
        daily_value["Cote Winamax"].notna()
    ].copy()

    if not daily_value.empty:

        daily_value["Proba Winamax"] = (
            100 / daily_value["Cote Winamax"]
        )

        daily_value["Value"] = (
            daily_value["Proba but"]
            - daily_value["Proba Winamax"]
        )

        daily_value["EV"] = (
            (
                daily_value["Proba but"] / 100
            )
            * daily_value["Cote Winamax"]
            - 1
        ) * 100

        daily_value = daily_value.sort_values(
            "Value",
            ascending=False
        )

        st.markdown("### 💰 Value du jour")

        # Statut du pari : non joué par défaut
        daily_value["🎯 Joué"] = False

        # Recharge le statut déjà enregistré si disponible
        if history_file_daily.exists():
            try:
                history_selected = pd.read_csv(history_file_daily)

                for idx, row in daily_value.iterrows():

                    source = daily_df[
                        (daily_df["Match"] == row["Match"])
                        & (daily_df["Joueur"] == row["Joueur"])
                        & (daily_df["Equipe"] == row["Equipe"])
                    ]

                    if source.empty:
                        continue

                    source_row = source.iloc[0]

                    existing = history_selected[
                        (history_selected["game_id"].astype(str)
                         == str(source_row["Game ID"]))
                        &
                        (history_selected["player_id"].astype(str)
                         == str(source_row["player_id"]))
                    ]

                    if (
                        not existing.empty
                        and "selected" in existing.columns
                        and pd.notna(existing.iloc[-1]["selected"])
                    ):
                        daily_value.at[idx, "🎯 Joué"] = bool(
                            existing.iloc[-1]["selected"]
                        )

            except Exception as e:
                print("Erreur lecture statut pari :", e)

        daily_value = st.data_editor(
            daily_value[[
                "Match",
                "Joueur",
                "Equipe",
                "Proba but",
                "Cote juste",
                "Cote Winamax",
                "Value",
                "EV",
                "🎯 Joué"
            ]],
            column_config={
                "Proba V3": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
                "Proba V4": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
                "Proba but": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
                "Cote juste": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Cote Winamax": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Value": st.column_config.NumberColumn(
                    format="%+.1f pts"
                ),
                "EV": st.column_config.NumberColumn(
                    "EV théorique",
                    format="%+.1f%%"
                ),
                "🎯 Joué": st.column_config.CheckboxColumn(
                    "🎯 Joué",
                    help="Cocher si ce pari a réellement été joué"
                )
            },
            disabled=[
                "Match",
                "Joueur",
                "Equipe",
                "Proba but",
                "Cote juste",
                "Cote Winamax",
                "Value",
                "EV"
            ],
            width="stretch",
            hide_index=True
        )


        # Enregistrement des cotes du jour
        if st.button(
            "💾 Enregistrer les cotes du jour",
            type="primary"
        ):
            history_file = Path("data/betting_history.csv")
            rows = []

            # Retrouve les identifiants cachés depuis daily_df
            for _, row in daily_value.iterrows():

                source = daily_df[
                    (daily_df["Match"] == row["Match"])
                    & (daily_df["Joueur"] == row["Joueur"])
                    & (daily_df["Equipe"] == row["Equipe"])
                ]

                if source.empty:
                    continue

                source_row = source.iloc[0]

                rows.append({
                    "date": today,
                    "game_id": source_row["Game ID"],
                    "match": row["Match"],
                    "team": row["Equipe"],
                    "player": row["Joueur"],
                    "player_id": source_row["player_id"],
                    "v3_raw": source_row["V3 brut"],
                    "model_version": "V4",
                    "model_raw": source_row["V4 brut"],
                    "goal_probability": row["Proba but"] / 100,
                    "fair_odds": row["Cote juste"],
                    "winamax_odds": row["Cote Winamax"],
                    "winamax_probability": 1 / row["Cote Winamax"],
                    "value": (
                        row["Proba but"] / 100
                        - 1 / row["Cote Winamax"]
                    ),
                    "ev": (
                        (row["Proba but"] / 100)
                        * row["Cote Winamax"]
                        - 1
                    ),
                    "result": None,
                    "selected": int(row["🎯 Joué"]),
                    "bet_odds": (
                        row["Cote Winamax"]
                        if int(row["🎯 Joué"]) == 1
                        else None
                    )
                })

            new_history = pd.DataFrame(rows)

            if not new_history.empty:

                if history_file.exists():
                    old_history = pd.read_csv(history_file)

                    # Conserver le résultat et la cote réellement jouée
                    if "bet_odds" not in old_history.columns:
                        old_history["bet_odds"] = None

                    old_state = old_history[
                        ["game_id", "player_id", "result", "bet_odds"]
                    ].drop_duplicates(
                        ["game_id", "player_id"],
                        keep="last"
                    ).copy()

                    new_history = new_history.merge(
                        old_state,
                        on=["game_id", "player_id"],
                        how="left",
                        suffixes=("", "_old")
                    )

                    new_history["result"] = new_history[
                        "result_old"
                    ].combine_first(
                        new_history["result"]
                    )

                    new_history["bet_odds"] = new_history[
                        "bet_odds_old"
                    ].combine_first(
                        new_history["bet_odds"]
                    )

                    new_history = new_history.drop(
                        columns=["result_old", "bet_odds_old"]
                    )

                    history = pd.concat(
                        [old_history, new_history],
                        ignore_index=True
                    )

                else:
                    history = new_history

                history = history.drop_duplicates(
                    subset=["game_id", "player_id"],
                    keep="last"
                )

                history.to_csv(
                    history_file,
                    index=False
                )

                st.success(
                    f"✅ {len(new_history)} cote(s) enregistrée(s)"
                )


# ---------------------------------------------------------
# ANALYSE D'UN MATCH
# ---------------------------------------------------------

st.divider()
st.header("🎯 Prédiction buteurs — V3")

if games:

    match_options = [
        g["Match"] for g in games
    ]

    selected_match = st.selectbox(
        "Choisir un match à analyser",
        match_options
    )

    selected_game = next(
        g for g in games
        if g["Match"] == selected_match
    )

    away_team = selected_game["Extérieur"]
    home_team = selected_game["Domicile"]

    st.subheader(
        f"🏒 {away_team} @ {home_team}"
    )

    with st.spinner(
        "Calcul des probabilités V3..."
    ):
        away_df = analyse_team(away_team)
        home_df = analyse_team(home_team)

    col1, col2 = st.columns(2)

    column_config = {
        "G/GP saison": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "L10 G/GP": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "L10 P/GP": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "L10 SOG/GP": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "V3 brut": st.column_config.NumberColumn(
            format="%.1f%%"
        ),
        "Proba but": st.column_config.NumberColumn(
            format="%.1f%%"
        ),
        "Cote juste": st.column_config.NumberColumn(
            format="%.2f"
        )
    }

    # Streamlit attend 35.5 pour afficher 35.5%
    # plutôt que 0.355.
    for df in [away_df, home_df]:
        if not df.empty:
            df["V3 brut"] *= 100
            df["Proba but"] *= 100

    # ---------------------------------------------------------
    # RECHARGEMENT DES COTES WINAMAX DEJA ENREGISTREES
    # ---------------------------------------------------------

    saved_odds = {}

    history_file = Path("data/betting_history.csv")

    if history_file.exists():
        try:
            history = pd.read_csv(history_file)

            current_history = history[
                history["game_id"].astype(str)
                == str(selected_game["Game ID"])
            ].copy()

            for _, saved in current_history.iterrows():
                if pd.notna(saved["player_id"]) and pd.notna(saved["winamax_odds"]):
                    saved_odds[int(saved["player_id"])] = float(
                        saved["winamax_odds"]
                    )

        except Exception as e:
            print("Erreur lecture historique :", e)

    for df in [away_df, home_df]:
        if not df.empty:
            df["Cote Winamax"] = df["player_id"].apply(
                lambda player_id: saved_odds.get(
                    int(player_id),
                    None
                )
            )

    column_config["Cote Winamax"] = st.column_config.NumberColumn(
        "Cote Winamax",
        min_value=1.01,
        step=0.05,
        format="%.2f",
        help="Saisir ici la cote buteur proposée par Winamax"
    )

    with col1:
        st.markdown(f"### {away_team}")

        away_df = st.data_editor(
            away_df,
            width="stretch",
            hide_index=True,
            column_config=column_config,
            disabled=[
                c for c in away_df.columns
                if c != "Cote Winamax"
            ],
            column_order=[
                c for c in away_df.columns
                if c != "player_id"
            ],
            key=f"odds_{away_team}_{selected_game['Game ID']}"
        )

    with col2:
        st.markdown(f"### {home_team}")

        home_df = st.data_editor(
            home_df,
            width="stretch",
            hide_index=True,
            column_config=column_config,
            disabled=[
                c for c in home_df.columns
                if c != "Cote Winamax"
            ],
            column_order=[
                c for c in home_df.columns
                if c != "player_id"
            ],
            key=f"odds_{home_team}_{selected_game['Game ID']}"
        )


    # ---------------------------------------------------------
    # ANALYSE VALUE WINAMAX
    # ---------------------------------------------------------

    combined = pd.concat(
        [
            away_df.assign(Equipe=away_team),
            home_df.assign(Equipe=home_team)
        ],
        ignore_index=True
    )

    combined["Cote Winamax"] = pd.to_numeric(
        combined["Cote Winamax"],
        errors="coerce"
    )

    value_df = combined[
        combined["Cote Winamax"].notna()
    ].copy()

    if not value_df.empty:

        # Probabilité implicite de la cote bookmaker
        value_df["Proba Winamax"] = (
            100 / value_df["Cote Winamax"]
        )

        # Proba but est déjà affichée en %
        value_df["Value"] = (
            value_df["Proba but"]
            - value_df["Proba Winamax"]
        )

        # Expected Value théorique du pari
        # Proba but est stockée en pourcentage dans ce tableau
        value_df["EV"] = (
            (value_df["Proba but"] / 100)
            * value_df["Cote Winamax"]
            - 1
        ) * 100

        value_df = value_df.sort_values(
            "Value",
            ascending=False
        )

        st.divider()
        st.header("💰 Analyse Value")

        st.dataframe(
            value_df[
                [
                    "Joueur",
                    "Equipe",
                    "Proba but",
                    "Cote juste",
                    "Cote Winamax",
                    "Proba Winamax",
                    "Value",
                    "EV"
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "Proba but": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
                "Cote juste": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Cote Winamax": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Proba Winamax": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
                "Value": st.column_config.NumberColumn(
                    format="%+.1f pts"
                ),
                "EV": st.column_config.NumberColumn(
                    "EV théorique",
                    format="%+.1f%%"
                )
            }
        )


# ============================================================
# HISTORIQUE DES ANALYSES
# ============================================================

st.divider()
st.subheader("📊 Historique des analyses")

history_file = Path("data/betting_history.csv")

if history_file.exists():
    history = pd.read_csv(history_file)

    if history.empty:
        st.info("Aucune analyse enregistrée.")
    else:
        history_display = history.copy()

        # Affichage en pourcentage
        for col in [
            "goal_probability",
            "winamax_probability",
            "value",
            "ev"
        ]:
            history_display[col] = history_display[col] * 100

        # Profit théorique pour une mise de 1 unité
        history_display["Profit (u)"] = history_display.apply(
            lambda row:
                row["winamax_odds"] - 1
                if row["result"] == 1
                else (
                    -1.0
                    if row["result"] == 0
                    else None
                ),
            axis=1
        )

        # Résultat lisible
        history_display["Résultat"] = history_display["result"].map(
            {
                1.0: "✅ Gagné",
                0.0: "❌ Perdu"
            }
        ).fillna("⏳ En attente")

        history_display = history_display.rename(columns={
            "date": "Date",
            "match": "Match",
            "team": "Équipe",
            "player": "Joueur",
            "goal_probability": "Proba but",
            "fair_odds": "Cote juste",
            "winamax_odds": "Cote Winamax",
            "winamax_probability": "Proba Winamax",
            "value": "Value",
            "ev": "EV"
        })

        history_display["🎯 Joué"] = (
            history_display["selected"]
            .fillna(0)
            .astype(int)
            .astype(bool)
        )

        history_edited = st.data_editor(
            history_display[
                [
                    "Date",
                    "Match",
                    "Joueur",
                    "Équipe",
                    "Proba but",
                    "Cote juste",
                    "Cote Winamax",
                    "Value",
                    "EV",
                    "Résultat",
                    "Profit (u)",
                    "🎯 Joué"
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "Proba but": st.column_config.NumberColumn(format="%.1f%%"),
                "Cote juste": st.column_config.NumberColumn(format="%.2f"),
                "Cote Winamax": st.column_config.NumberColumn(format="%.2f"),
                "Value": st.column_config.NumberColumn(format="%+.1f pts"),
                "EV": st.column_config.NumberColumn(format="%+.1f%%"),
                "Profit (u)": st.column_config.NumberColumn(format="%+.2f"),
                "🎯 Joué": st.column_config.CheckboxColumn(
                    "🎯 Joué",
                    help="Cocher si ce pari a réellement été joué"
                ),
            }
        )

else:
    st.info("Aucune analyse enregistrée.")


# ============================================================
# ENREGISTREMENT DES PARIS JOUES
# ============================================================

if "history_edited" in locals():

    if st.button("💾 Enregistrer les paris joués"):

        history_original = pd.read_csv(
            "data/betting_history.csv"
        )

        # On associe les lignes affichées aux lignes du CSV
        for _, edited_row in history_edited.iterrows():

            mask = (
                (history_original["game_id"].astype(str) ==
                 str(
                     history[
                         (history["player"] == edited_row["Joueur"]) &
                         (history["match"] == edited_row["Match"])
                     ]["game_id"].iloc[0]
                 ))
                &
                (history_original["player"] == edited_row["Joueur"])
            )

            played = int(edited_row["🎯 Joué"])

            history_original.loc[
                mask,
                "selected"
            ] = played

            # Figer la cote au moment où le pari est joué.
            # Une cote déjà figée ne sera jamais remplacée.
            if played == 1:
                needs_bet_odds = (
                    mask
                    & history_original["bet_odds"].isna()
                )

                history_original.loc[
                    needs_bet_odds,
                    "bet_odds"
                ] = history_original.loc[
                    needs_bet_odds,
                    "winamax_odds"
                ]

        history_original.to_csv(
            "data/betting_history.csv",
            index=False
        )

        st.success("Paris joués enregistrés.")
        st.rerun()


# ============================================================
# PERFORMANCE DES PARIS REELLEMENT JOUES
# ============================================================

st.divider()
st.subheader("💰 Performance des paris joués")

history_file = Path("data/betting_history.csv")

if history_file.exists():
    performance = pd.read_csv(history_file)

    settled_bets = performance[
        (performance["selected"] == 1)
        & (performance["result"].notna())
    ].copy()

    if settled_bets.empty:
        st.info(
            "Aucun pari joué et réglé pour le moment. "
            "Les statistiques apparaîtront après les premiers résultats."
        )

    else:
        settled_bets["profit"] = settled_bets.apply(
            lambda row:
                row["bet_odds"] - 1
                if row["result"] == 1
                else -1.0,
            axis=1
        )

        nb_bets = len(settled_bets)
        wins = int((settled_bets["result"] == 1).sum())
        losses = int((settled_bets["result"] == 0).sum())

        hit_rate = wins / nb_bets * 100
        total_profit = settled_bets["profit"].sum()

        # Mise fixe : 1 unité par pari
        roi = total_profit / nb_bets * 100

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Paris réglés", nb_bets)
        c2.metric("Gagnés / Perdus", f"{wins} / {losses}")
        c3.metric("Réussite", f"{hit_rate:.1f}%")
        c4.metric("Profit", f"{total_profit:+.2f} u")
        c5.metric("ROI", f"{roi:+.1f}%")

else:
    st.info("Aucun historique disponible.")
