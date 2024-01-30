# NBA-MVP-Predictor

##### This project involved creating and analyzing a dataset of MVP voting in the NBA over the last 40+ years. The NBA MVP Award is decided by a panel of media members that cover the league. Every voter ranks their top 5 candidates and players are awarded a 'Share' percentage based on the votes. The player with the highest percentage wins MVP. A Detailed explanation of the scoring system can be found at [here](https://en.wikipedia.org/wiki/NBA_Most_Valuable_Player_Award#:~:text=Each%20first%2Dplace%20vote%20is,point%20total%20wins%20the%20award). Scoring for the 2023 Award can be seen [here](https://www.basketball-reference.com/awards/awards_2023.html)
**The goal of this project is to build a regression model that can predict the voting 'Share' for each MVP candidate today**

## Project Steps
1. Scrape historical MVP voting and player statistics from basketball-reference.com. This also included pre-processing and feature engineering of variables.
   - can be found in scrape_mvp_data.ipynb notebook
2. Analyze data to find which statistics were most relevant predictors and create a baseline model
   - can be found in EDA.ipybnb and Lin_reg.ipynb
3. Experiment training different Regression models to find best predictor. In addition to Linear Regression, Random Forest, XGBoost and a custom Neural Network were all considered:
   - 40 years of data was used in the study and each year was held out as test set once during training. Models were evaluated based on whether they predicted the correct winner (prediction with the highest 'Share' matched the actual winner) in the test set for each of the 40 years.
   - can be found in appropriate notebooks
4. Scrape live betting odds and statistics for MVP candidates of current NBA season
   - can be found in scrape_current_cands.ipynb
5. Use best performing models to predict 'Share' for each candidate, publish results on streamlit
   - app.py and mvp_functions folder were used for this

Live odds and predictions were published on this [site](https://nba-mvp-predictor-5bblsks5tvgsu9x4ymzjtp.streamlit.app/) using Streamlit.
