import logging
from typing import Dict, List, Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveDataLoader")

class LiveNFLDataLoader:
    """Ingests live in-game telemetry, clock, situations, and in-game lines."""

    SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

    @classmethod
    def fetch_live_scoreboard(cls) -> List[Dict]:
        """Pulls real-time game states for all active and upcoming games."""
        try:
            resp = requests.get(cls.SCOREBOARD_URL, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Failed to fetch live scoreboard: %s", str(e))
            return []

        live_games = []
        events = data.get("events", [])

        for event in events:
            comp = event.get("competitions", [{}])[0]
            status = event.get("status", {})
            status_type = status.get("type", {})
            state = status_type.get("state", "pre")  # 'pre', 'in', 'post'

            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})

            situation = comp.get("situation", {})
            odds_arr = comp.get("odds", [{}])
            live_odds = odds_arr[0] if odds_arr else {}

            # Extract period and clock seconds remaining in regulation
            period = status.get("period", 1)
            display_clock = status.get("displayClock", "15:00")
            
            try:
                mins, secs = map(int, display_clock.split(":"))
                quarter_seconds = mins * 60 + secs
            except Exception:
                quarter_seconds = 900

            # Calculate total regulation seconds remaining (3600 total)
            if period <= 4:
                seconds_remaining = max(0, (4 - period) * 900 + quarter_seconds)
            else:
                seconds_remaining = max(0, quarter_seconds) # Overtime

            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))

            game_data = {
                "game_id": str(event.get("id")),
                "state": state,  # 'in' = live game currently active
                "period": period,
                "display_clock": display_clock,
                "seconds_remaining": seconds_remaining,
                "home_team": home.get("team", {}).get("abbreviation", "HOME"),
                "away_team": away.get("team", {}).get("abbreviation", "AWAY"),
                "home_score": home_score,
                "away_score": away_score,
                "margin": home_score - away_score,
                "current_total": home_score + away_score,
                "possession": situation.get("possession"),
                "down": situation.get("down", 1),
                "distance": situation.get("distance", 10),
                "yardline": situation.get("yardLine", 50),
                "is_redzone": situation.get("isRedZone", False),
                "down_distance_text": situation.get("downDistanceText", "1st & 10"),
                "possession_text": situation.get("possessionText", ""),
                "live_spread_line": live_odds.get("spread", 0.0),
                "live_over_under": live_odds.get("overUnder", 0.0)
            }
            live_games.append(game_data)

        return live_games
