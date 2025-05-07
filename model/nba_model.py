import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load CSV files
games = pd.read_csv("../data/games.csv")
games_details = pd.read_csv("../data/games_details.csv", low_memory=False)
players = pd.read_csv("../data/players.csv")
ranking = pd.read_csv("../data/ranking.csv")
teams = pd.read_csv("../data/teams.csv")

# Drop rows with missing target
games = games.dropna(subset=["PTS_home", "PTS_away"])

# Encode teams by name instead of ID
team_names = teams[["TEAM_ID", "NICKNAME"]].drop_duplicates()
games = games.merge(team_names, left_on="HOME_TEAM_ID", right_on="TEAM_ID", how="left").rename(columns={"NICKNAME": "HOME_TEAM"})
games = games.merge(team_names, left_on="VISITOR_TEAM_ID", right_on="TEAM_ID", how="left").rename(columns={"NICKNAME": "AWAY_TEAM"})
games = games.drop(columns=["TEAM_ID_x", "TEAM_ID_y"])

# Create a DataFrame with one row per team per game (for modeling)
home_df = games[[
    "GAME_ID", "SEASON", "HOME_TEAM", "AWAY_TEAM",
    "PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home"
]].copy()
home_df["TEAM"] = home_df["HOME_TEAM"]
home_df["OPPONENT"] = home_df["AWAY_TEAM"]
home_df["TARGET"] = home_df["PTS_home"]

away_df = games[[
    "GAME_ID", "SEASON", "AWAY_TEAM", "HOME_TEAM",
    "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away"
]].copy()
away_df["TEAM"] = away_df["AWAY_TEAM"]
away_df["OPPONENT"] = away_df["HOME_TEAM"]
away_df["TARGET"] = away_df["PTS_away"]

away_df.columns = home_df.columns  # Rename to match
df = pd.concat([home_df, away_df], ignore_index=True)

# Drop original team columns
df = df.drop(columns=["HOME_TEAM", "AWAY_TEAM"])

# Encode categorical columns
team_encoder = LabelEncoder()
df["TEAM_ENC"] = team_encoder.fit_transform(df["TEAM"])
opponent_encoder = LabelEncoder()
df["OPPONENT_ENC"] = opponent_encoder.fit_transform(df["OPPONENT"])

# Save encoders
joblib.dump(team_encoder, "team_encoder.pkl")
joblib.dump(opponent_encoder, "opponent_encoder.pkl")

# Select features and scale
features = ["TEAM_ENC", "OPPONENT_ENC", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home"]
X = df[[
    "TEAM_ENC", "OPPONENT_ENC", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home"
]].fillna(0)

y = df["TARGET"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "nba_score_model.pkl")

# Optional: print score
print("Model R² Score:", model.score(X_test, y_test))