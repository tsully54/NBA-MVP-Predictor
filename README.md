# NBA-MVP-Predictor
HTMLs of JupyterNotebooks. Project involved web scraping award voting and statistics of MVP candidates for every year starting from 1980. Different regression models were then tested and evaluated on different sets of training data. The best model was chosen to predict the MVP for the current season


## 2023 MVP Predictions.html
This notebook scrapes statistics for MVP candidates of current season and predicts MVP Share using previous years data.

## Random Forest Model.html, Lin Reg Model.html, XGBoost Regressor.html
These files contain the training and testing of different regression models. For each model, a total of 43 train/test predictions were run. For each of the 43 iterations, one season was left out to be the test set. So far, Random Forest Regression has offered the most accurate predictions and highest R2.

## Scraping of Advanced Stats.html, Merge Stats and EDA.html
These two files contain data preparation and EDA
