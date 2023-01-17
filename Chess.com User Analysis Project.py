#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import pandas as pd
import bs4
from bs4 import BeautifulSoup
import requests
import csv
import datetime
import time
import hashlib
import os  
from selenium import webdriver  
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.chrome.options import Options 

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
now = datetime.datetime.now()

USERNAME = "Plushdementals"
PASSWORD = "B0bthEchesser"
GAMES_URL = "https://www.chess.com/games/archive/plushdementals?gameOwner=my_game&gameTypes%5B0%" +         "5D=chess960&gameTypes%5B1%5D=daily&gameType=live&page="
LOGIN_URL = "https://www.chess.com/login"

driver = webdriver.Chrome("chromedriver.exe", options=options)
driver.get(LOGIN_URL)
uname = driver.find_element("id", "username")
uname.send_keys(USERNAME)
pword = driver.find_element("id", "password")
pword.send_keys(PASSWORD)
driver.find_element("id", "login").click()
time.sleep(5)

tables = []
game_links = []

for page_number in range(68):
    driver.get(GAMES_URL + str(page_number + 1))
    time.sleep(5)
    tables.append(
        pd.read_html(
            driver.page_source, 
            attrs={'class':'table-component table-hover archive-games-table'}
        )[0]
    )
    
    table_user_cells = driver.find_elements(By.CLASS_NAME, 'archive-games-user-cell')
    for cell in table_user_cells:
        link = cell.find_element(By.TAG_NAME,'a')
        game_links.append(link.get_attribute('href'))
        
driver.close()

games = pd.concat(tables)

identifier = pd.Series(
    games['Players'] + str(games['Result']) + str(games['Moves']) + games['Date']
).apply(lambda x: x.replace(" ", ""))

games.insert(
    0, 
    'GameId', 
    identifier.apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())
)

print(games.head(3))


# In[63]:


#Cleaning up the data that we scarped from chess.com

#creating 4 columns from the "Players" column.
#The 4 new columns are white player, white player rating, black player, black player rating
#expand = True is used so that a dataframe is returned and not a series/index
players_list = games.Players.str.split(expand = True)
print(players_list)


# In[64]:


#stripping the parenthesis off of the player ratings (ELO's)
white_elo = players_list[1].str.replace('(','').str.replace(')','')
black_elo = players_list[3].str.replace('(','').str.replace(')','')
print(players_list)


# In[66]:


#creating a relative database so I do not mistakenly ruin my data, dropping columns that aren't needed or cleaned yet)
game_archive = games[["GameId", "Unnamed: 0", "Accuracy", "Moves"]]


# In[83]:


#splitting the "Result" column to create the following columns: white result, black result
result_list = games.Result.str.split(expand = True)


# In[84]:


#Adding the columns to the table
game_archive["White"] = players_list[0]
game_archive["Black"] = players_list[2]
game_archive["White ELO"] = white_elo
game_archive["Black ELO"] = black_elo
game_archive["White Result"] = result_list[0]
game_archive["Black Result"] = result_list[1]


# In[94]:


#creating a column that shows if I won or lost a particular game
print(game_archive)
print(result_list.at[1,0])
my_result = pd.DataFrame()
for i in range(len(result_list)):
    if result_list[0] == "1" & game_archive["White"] == USERNAME:
        my_result.append(1)
    elif result_list[0] == "0" & game_archive["White"] == USERNAME:
        my_result.append(0)
    else:
        my_result.append(0.5)
print(my_result)


# In[69]:


#Cleaning the "Date" Column to a more presentable MM/DD/YYYY Format using two pandas functions
game_archive["Date"] = pd.to_datetime(games["Date"], errors="coerce").dt.strftime("%m/%d/%y")
print(game_archive)

