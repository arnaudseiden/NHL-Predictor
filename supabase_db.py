import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")


def _get_config():
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
        key = st.secrets.get("SUPABASE_SECRET_KEY", os.getenv("SUPABASE_SECRET_KEY"))
    except Exception:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SECRET_KEY")

    return url, key


def test_connection():
    url, key = _get_config()

    response = requests.get(
        f"{url}/rest/v1/betting_history?select=game_id&limit=1",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
        },
        timeout=15,
    )

    return response.status_code == 200


def load_history():
    url, key = _get_config()

    response = requests.get(
        f"{url}/rest/v1/betting_history?select=*&order=date.desc",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
        },
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def upsert_rows(rows):
    url, key = _get_config()

    # Colonnes gérées directement par Supabase
    technical_columns = {"id", "created_at", "updated_at"}

    rows = [
        {
            key_name: (
                None
                if pd.isna(value)
                else value
            )
            for key_name, value in row.items()
            if key_name not in technical_columns
        }
        for row in rows
    ]

    response = requests.post(
        f"{url}/rest/v1/betting_history?on_conflict=game_id,player_id",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        json=rows,
        timeout=30,
    )

    response.raise_for_status()
    return True
