############### FUNCTIONS FOR SCRAPING MVP DATA #################
import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
from basketball_reference_web_scraper import client
from unidecode import unidecode

### Scrape MVP Odds
def scrape_odds(url):
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the section containing "Regular Season MVP"
    mvp_section = soup.find('ul', {'class': 'game-props-card17'}) #adjust class accordingly as site changes

    data = []
    player_list = mvp_section.find_all('li', {'class': 'game-props-card17__cell'})
    for player in player_list:
        player_name = player.find('span', {'class': 'sportsbook-outcome-cell__label'})
        odds = player.find('span', {'class': 'sportsbook-odds'})
                
        if player_name and odds:
            data.append({
                'name': player_name.text.strip(),
                'odds': odds.text.strip()
            })
    
    odds = pd.DataFrame(data, columns=['name', 'odds']).head(10)
    cands = odds['name']
    return odds, cands

### Scrape basic statistics
def scrape_basic(year, cands):
    stats1 = pd.DataFrame(client.players_season_totals(season_end_year=year))
    stats1['name'] = stats1['name'].apply(lambda x: unidecode(x))
    stats2 = stats1[stats1['name'].isin(cands)]
    stats2['PTS'] = stats2['points'] / stats2['games_played']
    stats2['REB'] = (stats2['offensive_rebounds'] + stats2['defensive_rebounds']) / stats2['games_played']
    stats2['AST'] = stats2['assists'] / stats2['games_played']
    stats2['PRA'] = round(stats2['PTS'] + stats2['REB'] + stats2['AST'], 3)
    basic_cols = ['name', 'PRA']
    df1 = stats2[basic_cols]
    return df1

### Scrape Advanced statistics
def scrape_adv(year, cands):
    adv1 = pd.DataFrame(client.players_advanced_season_totals(season_end_year=year))
    adv1['name'] = adv1['name'].apply(lambda x: unidecode(x))
    adv2 = adv1[adv1['name'].isin(cands)]
    adv2['WS/48'] = adv2['win_shares_per_48_minutes']
    teams = []

    for i in adv2['team']:
        tm = i.value
        teams.append(tm)
    adv2['team'] = teams
    return adv2

### Scrape live standings
def scrape_standings(year):
    standings = pd.DataFrame(client.standings(season_end_year=year))
    standings['wl_pct'] = standings['wins']/(standings['wins']+standings['losses'])
    standings['conference'] = pd.Categorical(standings['conference'])
    standings['seed'] = float('nan')
    standings['seed'] = standings.groupby('conference')['wl_pct'].rank(ascending=False, method='min')
    standings['seed'] = standings['seed'].astype(int)
    standings['team'] = standings['team'].apply(lambda x: x.value)
    return standings

### Merge final dataframes
def merge_dfs(basic, adv, standings):
    # combine seed and wl_pct with adv stats of cands
    adv_cols = ['name', 'team', 'WS/48', 'player_efficiency_rating', 'offensive_box_plus_minus',
        'value_over_replacement_player']
    stand_cols = ['team', 'wl_pct', 'seed']
    df2 = pd.merge(adv[adv_cols], standings[stand_cols], on='team', how='inner')
    df = pd.merge(basic, df2, on='name', how='inner')
    return df

