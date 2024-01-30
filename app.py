import numpy as np
import pandas as pd
import random 
import streamlit as st
import altair as alt
from PIL import Image
from collections import Counter
import requests
from bs4 import BeautifulSoup
from basketball_reference_web_scraper import client
from unidecode import unidecode
import os
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import pickle
import warnings
warnings.filterwarnings('ignore')
import mvp_functions as mvp
import sys
print(sys.version)
# link with odds
url_odds = "https://sportsbook.draftkings.com/event/nba-awards-2023-24/6fe78ab7-324d-4d1a-7f10-08db724c2a58"

# use custom functions to load current odds and statistics for candidates
odds, cands = mvp.scrape_odds(url_odds)
url_basic = "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html"
df_basic = mvp.scrape_basic(url_basic, cands)
url_adv = "https://www.basketball-reference.com/leagues/NBA_2024_advanced.html"
adv = mvp.scrape_adv(url_adv, cands)
url_standings = "https://www.basketball-reference.com/leagues/NBA_2024_standings.html"
standings = mvp.scrape_standings(url_standings)
df = mvp.merge_dfs(df_basic, adv, standings)

#choose appropriate variables
X_cols = ['PRA',
 'WS/48',
 'PER',
 'OBPM',
 'VORP',
 'W/L%',
 'seed']

X = df[X_cols]

# predict with neural network model
load_folder = 'NN_models'
num_models = 5
loaded_models = []

for i in range(num_models):
    # Load the model from the 'NN_models' folder
    model = load_model(os.path.join(load_folder, f'model_{i + 1}.h5'))
    loaded_models.append(model)
nn_preds = np.mean([model.predict(X) for model in loaded_models], axis=0)


# predict with RandomForest model
rf = joblib.load("./rf_best.joblib")
rf_preds = rf.predict(X)

# Load results into dataframe
res = pd.DataFrame()
res['Player'] = df['Player']
res['NN_pred'] = nn_preds
res['RF_pred'] = rf_preds
res1 = res.sort_values(by='NN_pred', ascending=False)
results= pd.merge(res1, odds, on='Player', how='inner')

st.title("NBA MVP Prediction")
st.write("by Tommy Sullivan")

# Intro
st.subheader("NBA MVP Award")
st.caption("The NBA MVP Award is decided by a panel of media members that cover the league. Every voter ranks their top 5 candidates and players are awarded a 'Share' percentage based on the votes. The player with the highest percentage wins MVP. A Detailed explanation of the scoring system can be found at [here](https://en.wikipedia.org/wiki/NBA_Most_Valuable_Player_Award#:~:text=Each%20first%2Dplace%20vote%20is,point%20total%20wins%20the%20award). Scoring for the 2023 Award can be seen [here](https://www.basketball-reference.com/awards/awards_2023.html)")

# Data
st.subheader("Data Collection")
st.caption("Historical data on mvp voting and player statistics was scraped from basketball-reference.com. Current candidates and live odds were taken from [DraftKings](https://sportsbook.draftkings.com/event/nba-awards-2023-24/6fe78ab7-324d-4d1a-7f10-08db724c2a58)")

# Models
st.subheader("Predictive Models")
st.caption("Multiple regression models were experimented with including Linear Regression, Random Forests, XGBoost and a custom Neural Network")

# Training and evaluation
st.subheader("Model training and evaluation")
st.caption("A total of 40 years of MVP voting was used. Each year was used as a test set once for each type of regression model. Models were evaluated based on how well they predicted the 'Share' for each player in the test set. The main evaluation was whether the player with the highest predicted 'Share' matched the MVP winner for that year. R-squared and MSE were also calculated ")

st.header("2024 Prediction")
st.caption("Neural Network and Random Forests were the best regression models with Random Forest correctly picking the winner on 31/40 (78%) of years. Training results and model architectures can be seen below the predictions")
st.dataframe(results)

# Train/test results
st.subheader("Train/test results")
st.caption("The results of each year as a test set can be seen below. When you select a given year, you will see the predictions of a model that had that year removed from training")

#add dicts with NN and RF results
with open('nn_results_year.pkl', 'rb') as fp:
    nn_results_year = pickle.load(fp)

with open('rf_results_year.pkl', 'rb') as fp1:
    rf_results_year = pickle.load(fp1)

years = nn_results_year.keys()

year = st.selectbox('Select a Year', years)
col1, col2 = st.columns(2)
with col1:
    st.write("Neural Network model")    
    st.dataframe(nn_results_year[year])

with col2:
    st.write("Random Forest model")    
    st.dataframe(rf_results_year[year])

# Model parameters
nn_dict = {"number_hidden_layers":4,
            "neurons_per_layer":74,
            "hidden_layers_activation":"relu",
            "hidden_layers_initializer": "he_normal",
            "learning_rate": 0.008,
            "optimizer": "adam",
            "loss_function": "mse",
            "early_stopping": True}

st.subheader("Model hyperparameters")
st.caption("Different hyperparameters were tested for each model. Check out my notebooks on [Github](https://github.com/tsully54/NBA-MVP-Predictor) for complete training details. The best hyperparameters are shown for each model")
col1, col2 = st.columns(2)
with col1:
    st.write("Neural Network model")    
    st.write(nn_dict)

with col2:
    st.write("Random Forest model")    
    st.write(rf.get_params())
