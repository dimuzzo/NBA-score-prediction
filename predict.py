import pandas as pd
import numpy as np
import json
from tensorflow.keras.models import load_model
from src.config import FEATURES


def predict_single_game(game_data, model_path='nba_prediction_model.keras'):
    """
    Predicts the outcome of a single game using the trained model and proper normalization.
    'game_data' should be a dictionary with the necessary features.
    """
    try:
        model = load_model(model_path)
    except IOError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please train the model first by running 'python -m src.train'")
        return None

    try:
        with open('normalization_stats.json', 'r') as f:
            norm_stats = json.load(f)
        mean = pd.Series(norm_stats['mean'])
        std = pd.Series(norm_stats['std'])
    except FileNotFoundError:
        print("Error: 'normalization_stats.json' not found. Please train the model first.")
        return None

    input_df = pd.DataFrame([game_data])

    try:
        # This part will now succeed because the names in game_data match FEATURES
        input_df = input_df[FEATURES]
    except KeyError as e:
        print(f"Error: Missing feature in game_data: {e}")
        return None

    input_df_normalized = (input_df - mean) / std

    prediction_prob = model.predict(input_df_normalized)[0][0]

    home_win_prob = prediction_prob * 100
    away_win_prob = (1 - prediction_prob) * 100

    print(f"\nPrediction Analysis:")
    print(f"  > Home Team Win Probability: {home_win_prob:.2f}%")
    print(f"  > Away Team Win Probability: {away_win_prob:.2f}%")

    if home_win_prob > 50:
        print("  > Predicted Winner: Home Team")
    else:
        print("  > Predicted Winner: Away Team")

    return prediction_prob


if __name__ == '__main__':
    # The keys now match the feature names in config.py (e.g., 'PTS_avg_home')
    sample_game_data = {
        'PTS_avg_home': 115.2, 'FG_PCT_avg_home': 0.495, 'FT_PCT_avg_home': 0.78,
        'FG3_PCT_avg_home': 0.37, 'AST_avg_home': 26.5, 'REB_avg_home': 44.8,
        'PTS_avg_away': 110.5, 'FG_PCT_avg_away': 0.470, 'FT_PCT_avg_away': 0.81,
        'FG3_PCT_avg_away': 0.35, 'AST_avg_away': 24.1, 'REB_avg_away': 43.2
    }

    predict_single_game(sample_game_data)