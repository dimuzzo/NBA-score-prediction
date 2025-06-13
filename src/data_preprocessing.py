import pandas as pd
from . import config


def load_and_engineer_features():
    """
    Loads data and engineers features representing team form (rolling averages).
    """
    print("Loading datasets...")
    games = pd.read_csv(config.GAMES_PATH, parse_dates=['GAME_DATE_EST'])
    details = pd.read_csv(config.GAMES_DETAILS_PATH, low_memory=False)

    # Aggregate player stats to get team stats for each game
    team_game_stats = details.groupby(['GAME_ID', 'TEAM_ID']).agg({
        'PTS': 'sum',
        'FG_PCT': 'mean',
        'FT_PCT': 'mean',
        'FG3_PCT': 'mean',
        'AST': 'sum',
        'REB': 'sum'
    }).reset_index()

    # Merge with games to get the date for each team's game
    games_and_stats = pd.merge(games[['GAME_ID', 'GAME_DATE_EST']], team_game_stats, on='GAME_ID')

    # Sort by team and date to prepare for rolling averages
    games_and_stats.sort_values(by=['TEAM_ID', 'GAME_DATE_EST'], inplace=True)

    # Define stats to calculate rolling averages for
    stats_to_roll = ['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']

    # Calculate 10-game rolling average for each team
    print("Engineering rolling average features...")
    rolling_stats = games_and_stats.groupby('TEAM_ID')[stats_to_roll].rolling(window=10, min_periods=1).mean()
    rolling_stats = rolling_stats.reset_index(level=0, drop=True)  # Drop TEAM_ID index
    rolling_stats.rename(columns=lambda x: x + '_rolling_avg', inplace=True)

    # Combine original stats with rolling stats
    games_with_rolling = pd.concat([games_and_stats, rolling_stats], axis=1)

    # We need to use the stats from *before* the current game to avoid data leakage.
    # We do this by shifting the rolling average data by one game for each team.
    games_with_rolling[['PTS_avg', 'FG_PCT_avg', 'FT_PCT_avg', 'FG3_PCT_avg', 'AST_avg', 'REB_avg']] = \
        games_with_rolling.groupby('TEAM_ID')[rolling_stats.columns].shift(1)

    # Now, let's merge these features back into our main games dataframe
    games_df = games[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']]

    # --- FIX: Added .copy() to prevent SettingWithCopyWarning ---
    home_team_features = games_with_rolling[
        ['GAME_ID', 'TEAM_ID', 'PTS_avg', 'FG_PCT_avg', 'FT_PCT_avg', 'FG3_PCT_avg', 'AST_avg', 'REB_avg']].copy()
    home_team_features.rename(columns=lambda x: x + '_home' if x not in ['GAME_ID', 'TEAM_ID'] else x, inplace=True)
    final_df = pd.merge(games_df, home_team_features, left_on=['GAME_ID', 'HOME_TEAM_ID'],
                        right_on=['GAME_ID', 'TEAM_ID'], how='inner')

    # --- FIX: Added .copy() to prevent SettingWithCopyWarning ---
    away_team_features = games_with_rolling[
        ['GAME_ID', 'TEAM_ID', 'PTS_avg', 'FG_PCT_avg', 'FT_PCT_avg', 'FG3_PCT_avg', 'AST_avg', 'REB_avg']].copy()
    away_team_features.rename(columns=lambda x: x + '_away' if x not in ['GAME_ID', 'TEAM_ID'] else x, inplace=True)
    final_df = pd.merge(final_df, away_team_features, left_on=['GAME_ID', 'VISITOR_TEAM_ID'],
                        right_on=['GAME_ID', 'TEAM_ID'], how='inner')

    # Clean up columns and drop rows with missing data (from the initial rolling window)
    final_df.drop(columns=['TEAM_ID_x', 'TEAM_ID_y'], inplace=True)
    final_df.dropna(inplace=True)

    print(f"Processed data with new features has {final_df.shape[0]} samples.")
    return final_df


def get_features_and_target(df):
    """
    Separates features from the target variable and applies normalization.
    """
    print("Separating features and target...")
    features = df[config.FEATURES]
    target = df[config.TARGET]

    # Normalize features
    features_normalized = (features - features.mean()) / features.std()

    return features_normalized, target