import pandas as pd
import tensorflow as tf
from .data_preprocessing import load_and_engineer_features, get_features_and_target
from .model import create_nba_model
from . import config


def train_model():
    """
    Main function to load data, create the model, and start the training process.
    """
    print("--- Starting Model Training Pipeline ---")

    print("\nStep 1: Loading and preprocessing data...")
    df = load_and_engineer_features()

    # Save normalization stats BEFORE normalizing the data
    training_features = df[config.FEATURES]
    norm_stats = {
        'mean': training_features.mean().to_dict(),
        'std': training_features.std().to_dict()
    }
    pd.DataFrame(norm_stats).to_json('normalization_stats.json')
    print("Normalization stats saved to 'normalization_stats.json'")

    X, y = get_features_and_target(df)  # Normalization happens inside this function

    print("\nStep 2: Creating the model...")
    input_shape = X.shape[1]
    model = create_nba_model(input_shape)

    print("\nStep 3: Training the model...")
    history = model.fit(
        X, y,
        epochs=config.MODEL_PARAMS['epochs'],
        batch_size=config.MODEL_PARAMS['batch_size'],
        validation_split=config.MODEL_PARAMS['validation_split'],
        verbose=1
    )

    print("\n--- Training complete! ---")

    # Save the trained model using the modern .keras format
    model.save('nba_prediction_model.keras')
    print("Model has been saved as 'nba_prediction_model.keras'")

    return history


if __name__ == '__main__':
    train_model()