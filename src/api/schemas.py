from typing import List, Optional
from pydantic import BaseModel, Field

class GamePredictionResponse(BaseModel):
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    spread_line: float
    total_line: float
    pred_margin: float
    pred_total: float
    home_win_prob: float
    away_win_prob: float
    spread_edge: float
    total_edge: float

class PropPredictionRequest(BaseModel):
    player_id: str
    prop_name: str
    market_line: float

class PropPredictionResponse(BaseModel):
    player_id: str
    player_name: Optional[str] = None
    position: Optional[str] = None
    team: Optional[str] = None
    prop_name: str
    market_line: float
    expected_value: float
    over_prob: float
    under_prob: float

class AuditRecord(BaseModel):
    season: int
    week: int
    audited_at_utc: str
    total_games_audited: int
    spread_mae: float
    total_mae: float
    moneyline_brier: float
    moneyline_accuracy: float
    spread_ats_win_rate: float
    total_props_audited: int
    props_over_under_accuracy: float
