import os

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
GAMES_PATH = os.path.join(DATA_DIR, 'games.csv')
GAMES_DETAILS_PATH = os.path.join(DATA_DIR, 'games_details.csv')
TEAMS_PATH = os.path.join(DATA_DIR, 'teams.csv')
RANKING_PATH = os.path.join(DATA_DIR, 'ranking.csv')

# --- Model Parameters ---
MODEL_PARAMS = {
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2
}

# --- Features ---
FEATURES = [
    'PTS_avg_home', 'FG_PCT_avg_home', 'FT_PCT_avg_home', 'FG3_PCT_avg_home', 'AST_avg_home', 'REB_avg_home',
    'PTS_avg_away', 'FG_PCT_avg_away', 'FT_PCT_avg_away', 'FG3_PCT_avg_away', 'AST_avg_away', 'REB_avg_away'
]

TARGET = 'HOME_TEAM_WINS'