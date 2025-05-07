# NBA Score Prediction

This project aims to predict NBA game scores using machine learning techniques. By leveraging a dataset of NBA games and player statistics, the goal is to estimate the score of a game based on the performance metrics of both teams.

## Overview

This project allows users to predict the score of an NBA game between two selected teams by inputting various performance statistics. The model utilizes machine learning techniques and pre-trained models to provide an estimated score for both teams involved in the game.

## Features

- **Team Selection**: Choose two teams (your team and the opponent) from a dropdown list.
- **Performance Metrics**: Input various game statistics, including:
  - Field Goal Percentage (FG%)
  - Free Throw Percentage (FT%)
  - Three-Point Percentage (3PT%)
  - Assists (AST)
  - Rebounds (REB)
  
- **Score Prediction**: Based on the selected teams and the input statistics, the model predicts the final score for each team.

## Installation

To run this project, make sure you have the following Python packages installed:

- `joblib`
- `numpy`
- `ipywidgets`
- `sklearn`
- `IPython`

You can install the required packages using `pip`:

```bash
pip install joblib numpy ipywidgets scikit-learn ipython
```

## How to Use
- **Step 1**: Select 2 Teams

    From the dropdown menus, select the your team and opponent team for the game.

- **Step 2**: Enter Statistical Data

    For each team, adjust the sliders to input the following statistics:

    - Field Goal Percentage (FG%): A percentage of successful field goals.

    - Free Throw Percentage (FT%): A percentage of successful free throws.

    - Three-Point Percentage (3PT%): A percentage of successful three-point shots.

    - Assists (AST): Total number of assists.

    - Rebounds (REB): Total number of rebounds.

- **Step 3**: Predict the Score
    
    Click the Predict Score button to generate a prediction for both teams based on the input statistics. The model will process the data and provide the predicted score for both teams.

## Example Usage
Select Team A as your team and Team B as the opponent.

Input the following statistics for **Team A**:

- .FG%: 0.45

- .FT%: 0.75

- .3PT%: 0.35

- AST: 25

- REB: 40

Input the following statistics for **Team B**:

- .FG%: 0.46

- .FT%: 0.74

- .3PT%: 0.34

- AST: 23

- REB: 42

Click the Predict Score button to generate the predicted score.

The result will show the predicted scores for Team A and Team B based on the input statistics.

## Model Details
The model was trained using historical NBA game data, including team statistics and performance metrics. The input features used for the prediction are:

- Team Encoding: Teams are encoded into numerical values using a label encoder.

- Opponent Encoding: The opponent team is also encoded similarly.

- Statistical Features: FG%, FT%, 3PT%, AST, and REB values.

The model uses a regression technique to predict the total score for each team.

## Notes
**Model Accuracy**: The model was trained on historical data, but its predictions will depend on the quality and recency of the input data (data coverage until season 2021/2022).

**Warnings**: If you encounter a warning like UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names, you can safely ignore it. This warning does not affect the accuracy of the model.