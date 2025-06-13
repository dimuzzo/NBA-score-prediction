from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from . import config


def create_nba_model(input_shape):
    """
    Creates and compiles the deep learning model using the modern functional API style.
    """
    model = Sequential([
        # Explicit Input layer, as recommended by Keras
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        Dropout(0.3),  # Dropout helps prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        # Single output neuron for binary classification (win/loss)
        # Sigmoid activation outputs a probability between 0 and 1
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=config.MODEL_PARAMS['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Appropriate loss function for binary classification
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    # Example of how to create the model
    input_dim = len(config.FEATURES)
    model = create_nba_model(input_dim)
    print("Model created successfully. Summary:")
    model.summary()