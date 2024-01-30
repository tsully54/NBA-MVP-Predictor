############### FUNCTIONS FOR SCRAPING MVP DATA #################
import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
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
                'Player': player_name.text.strip(),
                'odds': odds.text.strip()
            })
    
    odds = pd.DataFrame(data, columns=['Player', 'odds']).head(10)
    cands = odds['Player']
    return odds, cands

### Scrape basic statistics
def scrape_basic(url, cands):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pg_stats = soup.find('table', {'id': 'per_game_stats'}) 
    header_row = pg_stats.find_all('tr')[0]
    column_headers = [th.text.strip() for th in header_row.find_all('th')]

    data_rows = pg_stats.find_all('tr')[2:]  # Exclude header rows

    data = []
    # Iterate through data rows and append data to the list
    for row in data_rows:
        row_data = [td.text.strip() for td in row.find_all('td')]
        data.append(row_data)

    df1 = pd.DataFrame(data, columns=column_headers[1:])
    df2 = df1.dropna()
    df2['Player'] = df2['Player'].apply(lambda x: unidecode(x))
    df3 = df2[df2['Player'].isin(cands)]

    num_cols = df3.columns.difference(['Player', 'Pos', 'Tm'])
    df3[num_cols] = df1[num_cols].apply(pd.to_numeric, errors='coerce')
    df3['PRA'] = df3['PTS'] + df3['TRB'] + df3['AST']
    basic_cols = ['Player', 'Tm', 'PRA']
    df4 = df3[basic_cols]
    return df4

### Scrape Advanced statistics
def scrape_adv(url, cands):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    adv_stats = soup.find('table', {'id': 'advanced_stats'}) 

    header_row = adv_stats.find_all('tr')[0]
    column_headers = [th.text.strip() for th in header_row.find_all('th')]

    data_rows = adv_stats.find_all('tr')[1:]  # Exclude header rows

    data = []
    # Iterate through data rows and append data to the list
    for row in data_rows:
        row_data = [td.text.strip() for td in row.find_all('td')]
        data.append(row_data)

    df1 = pd.DataFrame(data, columns=column_headers[1:])
    df2 = df1.dropna()
    adv_cols = ['Player', 'Tm', 'WS/48', 'PER', 'OBPM', 'VORP']
    df3 = df2[adv_cols]
    df3['Player'] = df3['Player'].apply(lambda x: unidecode(x))
    df4 = df3[df3['Player'].isin(cands)]
    num_cols = df4.columns.difference(['Player', 'Tm'])
    df4[num_cols] = df4[num_cols].apply(pd.to_numeric, errors='coerce')
    
    return df4

### Scrape live standings
def scrape_standings(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    stnd_E = soup.find('table', {'id': 'confs_standings_E'})  
    stnd_W = soup.find('table', {'id': 'confs_standings_W'}) 

    header_row_E = stnd_E.find_all('tr')[0]
    column_headers_E = [th.text.strip() for th in header_row_E.find_all('th')]

    data_rows_E = stnd_E.find_all('tr')[1:]  # Exclude header rows

    data_E = []

    # Iterate through data rows and append data to the list
    for row in data_rows_E:
        a_tag = row.find('th', {'class': 'left', 'data-stat': 'team_name'}).find('a')
        href_value = a_tag['href']
        team = href_value.split('/')[2]
        row_data = [td.text.strip() for td in row.find_all('td')]
        row_data.insert(0, team)
        data_E.append(row_data)

    # Create a Pandas DataFrame
    east = pd.DataFrame(data_E, columns=column_headers_E)
    east['seed'] = east['W/L%'].rank(ascending=False).astype(int)
    east = east.rename(columns={'Eastern Conference':'Tm'})

    header_row_W = stnd_W.find_all('tr')[0]
    column_headers_W = [th.text.strip() for th in header_row_W.find_all('th')]

    data_rows_W = stnd_W.find_all('tr')[1:]  # Exclude header rows

    data_W = []

    # Iterate through data rows and append data to the list
    for row in data_rows_W:
        a_tag = row.find('th', {'class': 'left', 'data-stat': 'team_name'}).find('a')
        href_value = a_tag['href']
        team = href_value.split('/')[2]
        row_data = [td.text.strip() for td in row.find_all('td')]
        row_data.insert(0, team)
        data_W.append(row_data)

    west = pd.DataFrame(data_W, columns=column_headers_W)
    west['seed'] = west['W/L%'].rank(ascending=False).astype(int)
    west = west.rename(columns={'Western Conference':'Tm'})
    standings = pd.concat([east, west], ignore_index=True)
    standings['W/L%'] = standings['W/L%'].apply(pd.to_numeric, errors='coerce')
    return standings

### Merge final dataframes
def merge_dfs(df_basic, adv, standings):
    stand_cols = ['Tm', 'W/L%', 'seed']
    df1 = pd.merge(adv, standings[stand_cols], on='Tm', how='inner')
    df2 = pd.merge(df_basic, df1, on='Player', how='inner')
    return df2

