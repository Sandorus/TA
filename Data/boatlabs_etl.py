"""
BoatLabs ETL Pipeline
---------------------
Fetches all race events from the BoatLabs API, normalizes data into relational tables,
and stores them in a local SQLite database (`boatlabs.db`).

Tables:
  - events
  - rounds
  - results
  - drivers
  - laps
"""

import requests
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from time import sleep

# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = None
EVENTS_URL = None
EVENT_DATA_URL = None

BOATLABS_DB = "boatlabs.db"
EVENT_API_DB = "event_boatlabs.db"

BOATLABS_EVENTS_URL = "https://api.boatlabs.net/v1/timingsystems/getEvents"
BOATLABS_EVENT_DATA_URL = "https://api.boatlabs.net/v1/timingsystems/getEvent/{}"

EVENT_API_EVENTS_URL = "https://api.event.boatlabs.net/v1/timingsystems/getEvents"
EVENT_API_EVENT_DATA_URL = "https://api.event.boatlabs.net/v1/timingsystems/getEvent/{}"

# -----------------------------
# DATABASE SETUP
# -----------------------------
def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_name TEXT PRIMARY KEY,
            date INTEGER,
            state TEXT,
            track_name TEXT,
            signed_up INTEGER,
            last_updated TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            round_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT,
            round_name TEXT,
            type TEXT,
            track_name TEXT,
            FOREIGN KEY (event_name) REFERENCES events(event_name)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id INTEGER,
            driver_name TEXT,
            driver_uuid TEXT,
            laps INTEGER,
            time_ms INTEGER,
            position INTEGER,
            FOREIGN KEY (round_id) REFERENCES rounds(round_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drivers (
            driver_uuid TEXT PRIMARY KEY,
            driver_name TEXT,
            races_entered INTEGER DEFAULT 0,
            last_seen TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS laps (
            lap_id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id INTEGER,
            driver_uuid TEXT,
            driver_name TEXT,
            lap_number INTEGER,
            lap_time_ms INTEGER,
            is_fastest BOOLEAN,
            pit_stop BOOLEAN,
            FOREIGN KEY (round_id) REFERENCES rounds(round_id),
            FOREIGN KEY (driver_uuid) REFERENCES drivers(driver_uuid)
        )
    """)
    conn.commit()

# -----------------------------
# API FUNCTIONS
# -----------------------------
def fetch_all_events():
    r = requests.get(EVENTS_URL)
    r.raise_for_status()
    return r.json().get("events", [])

def fetch_event_details(event_name):
    url = EVENT_DATA_URL.format(event_name)
    r = requests.get(url)
    if r.status_code != 200:
        print(f"⚠️ Skipping {event_name}, status={r.status_code}")
        return None
    return r.json()

# -----------------------------
# NORMALIZATION
# -----------------------------
def process_event(event_data):
    """Flatten event JSON into tables: event, rounds, results, laps."""
    if not event_data:
        return None, [], [], []

    now = datetime.now(timezone.utc).isoformat()
    event_name = event_data.get("name")

    event_row = {
        "event_name": event_name,
        "date": event_data.get("date"),
        "state": event_data.get("state"),
        "track_name": event_data.get("track_name"),
        "signed_up": event_data.get("signed_up", 0),
        "last_updated": now
    }

    rounds, results, laps = [], [], []

    for rnd in event_data.get("rounds", []):
        rnd_name = rnd.get("name")
        rounds.append({
            "event_name": event_name,
            "round_name": rnd_name,
            "type": rnd.get("type"),
            "track_name": rnd.get("track_name", event_data.get("track_name"))
        })

        # Unified driver processing function
        def process_driver(driver, position=None):
            driver_uuid = driver.get("uuid")
            driver_name = driver.get("name")
            # Total laps/time
            total_laps = driver.get("laps") if isinstance(driver.get("laps"), int) else len(driver.get("laps", []))
            total_time = driver.get("total_time") or driver.get("result", {}).get("time")
            results.append({
                "event_name": event_name,
                "round_name": rnd_name,
                "driver_name": driver_name,
                "driver_uuid": driver_uuid,
                "laps": total_laps,
                "time_ms": total_time,
                "position": position or driver.get("position")
            })
            # Extract lap-level data
            lap_list = driver.get("laps") or driver.get("result", {}).get("laps") or []
            if isinstance(lap_list, list):
                for idx, lap in enumerate(lap_list, start=1):
                    laps.append({
                        "event_name": event_name,
                        "round_name": rnd_name,
                        "driver_uuid": driver_uuid,
                        "driver_name": driver_name,
                        "lap_number": idx,
                        "lap_time_ms": lap.get("time"),
                        "is_fastest": lap.get("isFastest", False),
                        "pit_stop": lap.get("pitStop", False)
                    })

        # Process top-level results
        if rnd.get("results"):
            for pos, r in enumerate(rnd["results"], start=1):
                process_driver(r, position=pos)

        # Process heats with driver_details
        if rnd.get("heats"):
            for heat in rnd["heats"]:
                for driver in heat.get("driver_details", []):
                    process_driver(driver)

    return event_row, rounds, results, laps

# -----------------------------
# ETL MAIN
# -----------------------------
def run_etl():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    print("📡 Fetching events list...")
    events = fetch_all_events()
    print(f"Found {len(events)} events.")

    for e in events:
        event_name = e.get("name")

        # Skip existing events
        if conn.execute("SELECT 1 FROM events WHERE event_name=?", (event_name,)).fetchone():
            continue

        print(f"⏳ Fetching {event_name}...")
        event_data = fetch_event_details(event_name)
        if not event_data:
            continue

        event_row, rounds, results, laps = process_event(event_data)

        # Insert event
        pd.DataFrame([event_row]).to_sql("events", conn, if_exists="append", index=False)

        # Insert rounds
        if rounds:
            df_rounds = pd.DataFrame(rounds)
            df_rounds.to_sql("rounds", conn, if_exists="append", index=False)

        # Build round_id mapping
        round_map = dict(conn.execute(
            "SELECT round_id, round_name FROM rounds WHERE event_name=?",
            (event_name,)
        ).fetchall())
        round_name_to_id = {v: k for k, v in round_map.items()}

        # Insert results
        if results:
            df_results = pd.DataFrame(results)
            df_results["round_id"] = df_results["round_name"].map(round_name_to_id)
            df_results = df_results.drop(columns=["event_name", "round_name"])
            df_results.to_sql("results", conn, if_exists="append", index=False)

            # Update driver stats
            df_drivers = df_results[["driver_uuid", "driver_name"]].drop_duplicates()
            for _, row in df_drivers.iterrows():
                conn.execute("""
                    INSERT INTO drivers (driver_uuid, driver_name, races_entered, last_seen)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(driver_uuid) DO UPDATE SET
                        driver_name=excluded.driver_name,
                        races_entered=drivers.races_entered+1,
                        last_seen=excluded.last_seen
                """, (row.driver_uuid, row.driver_name, datetime.now(timezone.utc).isoformat()))

        # Insert laps
        if laps:
            df_laps = pd.DataFrame(laps)
            df_laps["round_id"] = df_laps["round_name"].map(round_name_to_id)
            df_laps["lap_number"] = df_laps["lap_number"].astype(int)
            df_laps["lap_time_ms"] = df_laps["lap_time_ms"].astype(int)
            df_laps["is_fastest"] = df_laps["is_fastest"].astype(bool)
            df_laps["pit_stop"] = df_laps["pit_stop"].astype(bool)
            df_laps = df_laps.drop(columns=["event_name", "round_name"])
            df_laps.to_sql("laps", conn, if_exists="append", index=False)

        conn.commit()
        sleep(0.1)

    conn.close()
    print("✅ ETL complete. Data stored in", DB_PATH)

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    # Ask user which API to use
    print("Select API to use:")
    print("1. BoatLabs API")
    print("2. Event API")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        DB_PATH = BOATLABS_DB
        EVENTS_URL = BOATLABS_EVENTS_URL
        EVENT_DATA_URL = BOATLABS_EVENT_DATA_URL
    elif choice == "2":
        DB_PATH = EVENT_API_DB
        EVENTS_URL = EVENT_API_EVENTS_URL
        EVENT_DATA_URL = EVENT_API_EVENT_DATA_URL
    else:
        print("Invalid choice, exiting.")
        exit(1)
    
    print(f"Using database: {DB_PATH}")
    run_etl()
