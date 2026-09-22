import logging
import time
from typing import Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveDataLoader")

class LiveNFLDataLoader:
    """Robust, cached live telemetry ingestor for active NFL game states."""

    SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    _cache_timestamp: float = 0.0
    _cached_data: List[Dict] = []
    _CACHE_TTL_SECONDS: float = 5.0

    # Reusable session with automated retries and connection pooling
    _session: Optional[requests.Session] = None

    @classmethod
    def _get_session(cls) -> requests.Session:
        if cls._session is None:
            cls._session = requests.Session()
            retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
            cls._session.mount("https://", adapter)
        return cls._session

    @staticmethod
    def _parse_clock_seconds(display_clock: str, period: int) -> int:
        """Safely parses clock strings into remaining regulation seconds."""
        quarter_seconds = 900
        if display_clock and ":" in display_clock:
            try:
                parts = display_clock.split(":")
                quarter_seconds = int(parts[0]) * 60 + int(parts[1])
            except Exception:
                quarter_seconds = 900
        elif display_clock and display_clock.isdigit():
            quarter_seconds = int(display_clock)

        if period <= 4:
            return max(0, (4 - period) * 900 + quarter_seconds)
        return max(0, quarter_seconds) # Overtime

    @classmethod
    def fetch_live_scoreboard(cls) -> List[Dict]:
        """Returns live game telemetry, serving from 5-second cache when valid."""
        current_time = time.time()
        if cls._cached_data and (current_time - cls._cache_timestamp) < cls._CACHE_TTL_SECONDS:
            return cls._cached_data

        session = cls._get_session()
        try:
            resp = session.get(cls.SCOREBOARD_URL, timeout=4)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Scoreboard API request failed, serving stale cache: %s", str(e))
            return cls._cached_data

        live_games = []
        events = data.get("events", [])

        for event in events:
            try:
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

                period = int(status.get("period", 1))
                display_clock = status.get("displayClock", "15:00")
                seconds_remaining = 0 if state == "post" else cls._parse_clock_seconds(display_clock, period)

                home_score = int(home.get("score", 0))
                away_score = int(away.get("score", 0))

                live_games.append({
                    "game_id": str(event.get("id", "")),
                    "state": state,
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
                    "down": int(situation.get("down", 1) or 1),
                    "distance": int(situation.get("distance", 10) or 10),
                    "yardline": int(situation.get("yardLine", 50) or 50),
                    "is_redzone": bool(situation.get("isRedZone", False)),
                    "down_distance_text": situation.get("downDistanceText", "1st & 10"),
                    "possession_text": situation.get("possessionText", ""),
                    "live_spread_line": float(live_odds.get("spread", 0.0) or 0.0),
                    "live_over_under": float(live_odds.get("overUnder", 0.0) or 0.0)
                })
            except Exception as item_err:
                logger.error("Failed to parse event %s: %s", event.get("id"), str(item_err))
                continue

        cls._cached_data = live_games
        cls._cache_timestamp = current_time
        return live_games
