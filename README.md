# NBA Game Winner Prediction 🏀

![GitHub last commit](https://img.shields.io/github/last-commit/dimuzzo/NBA-score-prediction?style=flat-square&logo=github&label=Last%20Commit)
![GitHub repo size](https://img.shields.io/github/repo-size/dimuzzo/NBA-score-prediction?style=flat-square&logo=github&label=Repo%20Size)
![GitHub stars](https://img.shields.io/github/stars/dimuzzo/NBA-score-prediction?style=flat-square&logo=github&label=Stars)

This project utilizes a deep learning model to predict the winner of NBA games based on engineered features derived from historical game statistics.

---
## 🗺️ Overview

The core of this project is a Keras-based neural network trained on team performance metrics. Instead of using a game's own final stats to predict its outcome (which would be data leakage), we engineer features that represent each team's form and strength *before* the game begins. This is achieved by calculating 10-game rolling averages for key statistical categories.

The project is structured into a modular pipeline that handles data preprocessing, model training, and prediction.

---
## 📂 Project Structure

The project is organized to separate concerns, making it clean and maintainable.

-   `data/`: Contains all the original CSV datasets.
-   `notebooks/`:
    -   `data_analysis.ipynb`: For initial exploratory data analysis (EDA).
    -   `nba_widget.ipynb`: An interactive Jupyter widget to test the trained model.
-   `src/`: Contains the modular source code.
    -   `config.py`: Configuration file for paths, features, and model hyperparameters.
    -   `data_preprocessing.py`: Script for loading data and performing feature engineering (e.g., rolling averages).
    -   `model.py`: Defines the Keras model architecture.
    -   `train.py`: The main script to execute the model training pipeline.
-   `predict.py`: A command-line script to make a single prediction using the saved model.
-   `requirements.txt`: A list of the project's Python dependencies.
-   `nba_prediction_model.keras`: The saved, trained model (generated after running the training script).
-   `normalization_stats.json`: Saved mean/std values from the training set for consistent data normalization.

---
## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9-3.12
- Git

It is highly recommended to use a virtual environment.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/dimuzzo/NBA-score-prediction.git
cd nba-score-prediction
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Training the Model
To train the model, run the `train.py` script from the project's root directory:
```bash
python -m src.train
```

This command will load the data, engineer the features, train the model, and save `nba_prediction_model.keras` and `normalization_stats.json` in the root folder.

### 4. Making a Prediction
You can make predictions in two ways:

A) Via Command Line:
    
- Modify the `sample_game_data` dictionary in `predict.py` and run the script:
    ```bash
    python predict.py
    ```

B) Via Interactive Widget:

- Launch Jupyter Lab and open the widget notebook:
    ```bash
    jupyter lab notebooks/nba_widget.ipynb
    ```
- Use directly software like PyCharm.

Use the interactive sliders to input team stats and see the prediction in real-time.

---

## 🙏 Contributions

If you want to contribute to the project, feel free to fork the repository and submit a pull request!

---

> Created with passion by [dimuzzo](https://github.com/dimuzzo)
