#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
#For this section of the project, a majority of this code is borrowed from the shedloadofcode.com article that inspired this project. I did need to spend time updating certain aspects of the code for a more current selenium version and tailoring it to my account specifically, but the logic used remained the same. 


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

# this part is edited to retain privacy
USERNAME = "***"
PASSWORD = "***"
GAMES_URL = "https://www.chess.com/games/archive/***?gameOwner=my_game&gameTypes%5B0%" + \
        "5D=chess960&gameTypes%5B1%5D=daily&gameType=live&page="
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

for page_number in range(64):
    #the if and else statements needed added because chess.com would not let me access certain pages for some reason.
    if page_number <= 25:
        driver.get(GAMES_URL + str(page_number + 1))
    else:
        driver.get(GAMES_URL + str(page_number + 5))
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

'''


# In[28]:


'''
From this point forwards, the code written is my own. With that said, I did use online resources to find the appropriate
syntax and logic for functions I was unfamiliar with. Additionally, some of the code ended up being very similar to the
code from the example project that inspired me to make this, but that was bound to happen with the similarity between the
two projects and the alike problems that were bound to arise. 

The websites I referenced are in the README section of the github repository related to this project.
'''


# In[3]:


import numpy as np
import pandas as pd
import requests
import csv
import datetime
import time
import hashlib
import os  
import matplotlib.pyplot as plt
import pandasql
from pandasql import sqldf
import scipy
from scipy import stats 

USERNAME = '***'

#opening up saved data
new_archive = pd.read_csv('new_archive_data.csv', sep=';')
game_archive = pd.read_csv('chess-data.csv', sep=';')
games = pd.read_csv('game-data.csv', sep=';')

#getting rid of annoying warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[4]:


#Cleaning up the data that we scarped from chess.com

#creating 4 columns from the "Players" column.
#The 4 new columns are white player, white player rating, black player, black player rating
#expand = True is used so that a dataframe is returned and not a series/index
players_list = games.Players.str.split(expand = True)

#stripping the parenthesis off of the player ratings (ELO's)
white_elo = players_list[1].str.replace('(','').str.replace(')','')
black_elo = players_list[3].str.replace('(','').str.replace(')','')


# In[5]:


#creating a relative database so I do not mistakenly ruin my data and dropping columns that aren't needed or cleaned yet)
game_archive = games[['GameId', 'Unnamed: 0', 'Accuracy', 'Moves']]


# In[6]:


#splitting the "Result" column to create the following columns: White_Result, Black_Result
result_list = games.Result.str.split(expand = True)


# In[7]:


#Adding columns to the table for the white/black player username, white/black player elo, and each colors' result
game_archive['White'] = players_list[0]
game_archive['Black'] = players_list[2]
game_archive['White_ELO'] = white_elo
game_archive['Black_ELO'] = black_elo
game_archive['White_Result'] = result_list[0]
game_archive['Black_Result'] = result_list[1]

#Renaming poorly named column
game_archive.rename(columns = {'Unnamed: 0':'Time_Control'}, inplace = True)

#removing rows from the data that were not properly scraped from chess.com
game_archive['White_ELO'] = game_archive['White_ELO'].str.replace('(\D+)', '0')
game_archive['Black_ELO'] = game_archive['Black_ELO'].str.replace('(\D+)', '0')
game_archive['White_ELO'] = pd.to_numeric(game_archive['White_ELO'])
game_archive['Black_ELO'] = pd.to_numeric(game_archive['Black_ELO'])


# In[8]:


#creating a column that shows if I won, lost, or drew a particular game

conditions = [
    (game_archive['White'] == USERNAME) & (game_archive['White_Result'] == '1'),
    (game_archive['White'] == USERNAME) & (game_archive['White_Result'] == '0'),
    (game_archive['White'] == USERNAME) & (game_archive['White_Result'] == '½'),
    (game_archive['White'] != USERNAME) & (game_archive['White_Result'] == '1'),
    (game_archive['White'] != USERNAME) & (game_archive['White_Result'] == '0'),
    (game_archive['White'] != USERNAME) & (game_archive['White_Result'] == '½')
    ]

values = [1,0,0.5,0,1,0.5]

my_result = np.select(conditions, values)

#truncating the 'My Result' column because having multiple decimal places doesn't make sense in this context
game_archive['My_Result'] = my_result.astype('float')


# In[9]:


#creating a column that shows what my color was in the game

conditions = [
    (game_archive['White'] == USERNAME),
    (game_archive['White'] != USERNAME)
]

values = ['White', 'Black']

my_color = np.select(conditions, values)

game_archive['My_Color'] = my_color


# In[10]:


#Cleaning the 'Date' Column to a more presentable MM/DD/YYYY Format using two pandas functions
game_archive['Date'] = pd.to_datetime(games['Date'], errors='coerce')
game_archive['Date'] = game_archive['Date'].dt.strftime('%m/%d/%y')
game_archive['Date'] = pd.to_datetime(game_archive['Date'], errors='coerce')

'''
I had no July values in my data set, after some thinking I believe that I likely played little to no games in the summer
and also may have lost games from july from the pages chess.com wouldn't let me access
'''

conditions = [
    (game_archive.Date.dt.month >= 0) & (game_archive.Date.dt.month <= 3),
    (game_archive.Date.dt.month >= 4) & (game_archive.Date.dt.month <= 6),
    (game_archive.Date.dt.month >= 7) & (game_archive.Date.dt.month <= 9),
    (game_archive.Date.dt.month >= 10) & (game_archive.Date.dt.month <= 12)
]

values = ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']

quarter = np.select(conditions, values)

game_archive['Quarter'] = quarter


# In[11]:


#Adding a column to game_archive that shows the difference in ratings between me and my opponent

conditions = [
    game_archive['White'] == USERNAME,
    game_archive['White'] != USERNAME
]

values = [game_archive['White_ELO'], game_archive['Black_ELO']]

my_elo = np.select(conditions, values)

#the conditions stay the same for opponent elo so only new values are needed
values = [game_archive['Black_ELO'], game_archive['White_ELO']]

opponent_elo = np.select(conditions, values)

game_archive['Rating_Diff'] = my_elo - opponent_elo 


# In[12]:


#creating a column that shows if my rating is better, worse, or equal to my opponent

conditions = [
    game_archive['Rating_Diff'] > 0,
    game_archive['Rating_Diff'] == 0,
    game_archive['Rating_Diff'] < 0
]

values = ['Better', 'Equal', 'Worse']

better_worse = np.select(conditions, values)

game_archive['Better_Worse'] = better_worse


# In[13]:


#adding a column to the game_archive data frame that descripves what type of game I played
conditions = [
    game_archive['Time_Control'] == '1 min',
    game_archive['Time_Control'] == '3 min',
    game_archive['Time_Control'] == '5 min',
    game_archive['Time_Control'] == '10 min',
    game_archive['Time_Control'] == '30 min',
    game_archive['Time_Control'] == '1|1',
    game_archive['Time_Control'] == '2|1',
    game_archive['Time_Control'] == '3|2',
    game_archive['Time_Control'] == '5|2',
    game_archive['Time_Control'] == '5|5',
    game_archive['Time_Control'] == '15|5',
    game_archive['Time_Control'] == '10|10',
    game_archive['Time_Control'] == '15|10',
    game_archive['Time_Control'] == '25|10'
]

values = ['Lightning','Blitz','Blitz','Rapid','Rapid','Lightning','Lightning','Blitz','Blitz','Blitz','Rapid','Rapid', 'Rapid', 'Rapid']

game_archive['Game_Type'] = np.select(conditions, values)


# In[14]:


#removing data values that were improperly scraped and creating a new polished data frame
new_archive = game_archive[
                           (game_archive['White_ELO'] > 100) & 
                           (game_archive['White_ELO'] < 3000) &
                           (game_archive['Black_ELO'] > 100) &
                           (game_archive['Black_ELO'] < 3000)
                          ]

new_archive['Date'] = pd.to_datetime(game_archive['Date'], errors='coerce')


# In[15]:


#making a data frame consisting of my analyzed games for later analysis, since only a portion of my games were analyzed.
accuracy_df = games.Accuracy.str.split(expand = True)
accuracy_df['GameId'] = games.GameId
accuracy_df = accuracy_df.dropna()


analysis = pd.DataFrame()
analysis['White_Accuracy'] = pd.to_numeric(accuracy_df[0])
analysis['Black_Accuracy'] = pd.to_numeric(accuracy_df[1])
analysis['GameId'] = accuracy_df['GameId']


# In[17]:


#using pandasql to create a dataframe containing the accuracy for all games that I have analyzed

query = """
SELECT
    a.'White_Accuracy',
    a.'Black_Accuracy',
    a.GameId,
    n.'Moves',    
    n.'White_ELO',
    n.'Black_ELO',
    n.'White_Result',
    n.'Black_Result',
    n.'White',
    n.'Black',
    n.'Game_Type',
    n.'Rating_Diff',
    n.'My_Result',
    (a.'White_Accuracy' - a.'Black_Accuracy') as 'Accuracy_Diff'
FROM analysis as a
JOIN new_archive as n
ON a.GameId = n.GameId;
"""

sqldf(query).head()

final_accuracy = sqldf(query)


# In[18]:


#Am I more successful as white or black?

#getting the counts of the wins, draws, and losses by color
color_counts = new_archive.groupby(['My_Color', 'My_Result']).size()

index = np.arange(2)

losses = [706, 678]
losses_bar = plt.bar(index-0.25, 
                    losses, 
                    width=0.25, 
                    color = 'black', 
                    label = 'Losses',
                    edgecolor='black')

wins = [797, 835]
wins_bar = plt.bar(index,
                   wins,
                   width=0.25,
                   color = 'green',
                   label = 'Wins',
                   edgecolor='black')

draws = [86, 95]
draws_bar = plt.bar(index+0.25, 
                    draws, 
                    width=0.25, 
                    color = 'gray', 
                    label = 'Draws', 
                    edgecolor='black')

plt.xticks(index, ['Black', 'White'])
plt.legend(bbox_to_anchor=(0.35,1))
plt.xlabel('Color')
plt.ylabel('Total L/W/D')

caption = 'Viz by Robbie Goss | Data Collected from chess.com'
plt.text(0.5,-160, caption, ha='right')

plt.show()


# In[19]:


#Does the more accurate player actually win?

#removing extranneous values
final_accuracy_filtered1 = final_accuracy[final_accuracy['Rating_Diff'] < 200]
final_accuracy_filtered2 = final_accuracy_filtered1[final_accuracy_filtered1['Rating_Diff'] > -200]

fig, accuracy = plt.subplots()
accuracy.scatter(final_accuracy_filtered2['Rating_Diff'],
                 final_accuracy_filtered2['Accuracy_Diff'],
                 color = 'gray')

accuracy.set(title = 'Scatter Plot of Accuracy Difference vs. Player Rating Difference',
             xlabel = 'Rating Difference',
             ylabel = 'Accuracy Difference')

#creating a line of best fit
slope, intercept, r_val, p_val, std_err = scipy.stats.linregress(final_accuracy_filtered2['Rating_Diff'],
                                                                 final_accuracy_filtered2['Accuracy_Diff'])


accuracy.plot(final_accuracy_filtered2['Rating_Diff'],
              slope * final_accuracy_filtered2['Accuracy_Diff'] + intercept,
              lw=0.5,
              color = 'green')

caption = 'Viz by Robbie Goss | Data Collected from chess.com'
accuracy.text(0.5,-150, caption, ha='right')

plt.show()


# In[20]:


#Do I win longer or shorter games?


#Creating a data frame that will have my win percentage at each number of moves played
win_pct_df = pd.DataFrame([0]*137, range(137))
win_pct_df.rename(columns={0: 'Win_Pct'}, inplace=True)
win_pct_df.insert(0, 'Num_Moves', range(0, len(win_pct_df)))
win_pct_df['Numerator'] = 0
win_pct_df['Denominator'] = 0
                          
#Creating variable for the for loop that will help me calculate my win percentages at each number of moves played
num_moves = 0 

for index in new_archive.index:
    num_moves = new_archive['Moves'][index]
    if new_archive['My_Result'][index] == 1:
        win_pct_df['Numerator'][num_moves] += 1
        win_pct_df['Denominator'][num_moves] += 1
    elif new_archive['My_Result'][index] == 0.5:
        win_pct_df['Numerator'][num_moves] += 0.5
        win_pct_df['Denominator'][num_moves] += 1
    elif new_archive['My_Result'][index] == 0:
        win_pct_df['Numerator'][num_moves] += 0
        win_pct_df['Denominator'][num_moves] += 1

#Calculating win percentages by diving my wins (plus 0.5 for each draw) divided by games played
win_pct_df['Win_Pct'] = win_pct_df['Numerator'] / win_pct_df['Denominator']

#removing rows with no win percentage (no games played)
win_pct_df = win_pct_df.dropna()
win_pct_filtered1 = win_pct_df[win_pct_df['Win_Pct'] > 0]
win_pct_filtered2 = win_pct_filtered1[win_pct_filtered1['Win_Pct'] < 1]


#Creating a scatter plot to visiualize my win pct by moves in a game
fig, moves = plt.subplots()

moves.scatter(win_pct_filtered2['Num_Moves'], 
              win_pct_filtered2['Win_Pct'],
              color = 'gray')
moves.set(title = 'Scatter Plot of Win Percentage by Number of Moves in a Game',
       xlabel = 'Moves',
       ylabel = 'Win Percentage')

caption = 'Viz by Robbie Goss | Data Collected from chess.com'
moves.text(0.5,-.05, caption, ha='center')

#creating a polynomial line of best fit for the data, as the data is non-linear
model = np.poly1d(np.polyfit(win_pct_filtered2['Num_Moves'], win_pct_filtered2['Win_Pct'], 3))
length = np.linspace(0,87,87)
moves.plot(length, model(length), color = 'green')

plt.show()


# In[21]:


#Am I better off playing one or multiple games in a day? 

#using pd.crosstab to make a data frame with my total number of games in a day and win percentage for that day
daily_wl = pd.crosstab(new_archive['Date'], new_archive['My_Result'])
daily_wl['Total_Games'] = daily_wl[0.0] + daily_wl[0.5] + daily_wl[1.0]
daily_wl['My_Score'] = (daily_wl[0.5]*0.5) + daily_wl[1.0]
daily_wl['Win_Pct'] = daily_wl['My_Score'] / daily_wl['Total_Games']

#creating a scatter plot of win percentage by number of games played in a day
fig, dates = plt.subplots()
dates.scatter(daily_wl['Win_Pct'],
              daily_wl['Total_Games'],
              color = 'black',
              alpha = 0.8)

dates.set(title = 'Scatter Plot of Win Percentage by Number of Games Played in a Day',
          ylabel = 'Games in a Day',
          xlabel = 'Win Percentage')

'''
# I noticed that this scatter plot appears to follow a bell curve, which makes sense because as the number of games in
a days increases we would expect the sample variance to decrease and see more values concentrated to the population
mean (0.5). 
'''

mean = 0.5
sd = 0.2
x_vals = np.linspace(0, 1, 100)
plt.plot(x_vals, 
         25*stats.norm.pdf(x_vals, mean, sd),
         color = 'gray')

caption = 'Viz by Robbie Goss | Data Collected from chess.com'
dates.text(0, -12, caption, ha='center')
plt.show()


# In[22]:


#In what type of game am I most successful?

type_wl = pd.crosstab(new_archive['Game_Type'], new_archive['My_Result'])
type_wl['Total_Games'] = type_wl[0.0] + type_wl[0.5] + type_wl[1.0]
type_wl['My_Score'] = (type_wl[0.5]*0.5) + type_wl[1.0]
type_wl['Win_Pct'] = type_wl['My_Score'] / type_wl['Total_Games']

#creating a bar plot of win percentage by game type
wl_bar = type_wl.plot.bar(y = ['Win_Pct'],
                          color = 'green',
                          edgecolor = 'black',
                          title = 'Bar Plot of Win Percentage by Game Type',
                          xlabel = 'Game Type',
                          ylabel = 'Win Percentage',
                          rot = 0,
                          legend = '')

caption = 'Viz by Robbie Goss | Data Collected from chess.com'
wl_bar.text(0,-0.12, caption, ha='center')

plt.show()


# In[23]:



final_accuracy.loc[final_accuracy['White'] == USERNAME, 'My_Accuracy'] = final_accuracy['White_Accuracy']    
final_accuracy.loc[final_accuracy['White'] != USERNAME, 'My_Accuracy'] = final_accuracy['Black_Accuracy'] 

#creating variables for my average accuracy in each type of chess game
total_blitz_accuracy = 0
total_lightning_accuracy = 0
total_rapid_accuracy = 0
total_blitz_games = 0
total_lightning_games = 0
total_rapid_games = 0

#creating a sum of the total accuracy for each game type
for index in final_accuracy.index:
    if final_accuracy['Game_Type'][index] == 'Blitz':
        total_blitz_accuracy += final_accuracy['My_Accuracy'][index]
        total_blitz_games += 1
    elif final_accuracy['Game_Type'][index] == 'Rapid':
        total_rapid_accuracy += final_accuracy['My_Accuracy'][index]
        total_rapid_games += 1
    elif final_accuracy['Game_Type'][index] == 'Lightning':
        total_lightning_accuracy += final_accuracy['My_Accuracy'][index]
        total_lightning_games += 1

#dividing total accuracy by number of games played
avg_blitz_accuracy = total_blitz_accuracy / total_blitz_games
avg_lightning_accuracy = total_lightning_accuracy / total_lightning_games
avg_rapid_accuracy = total_rapid_accuracy / total_rapid_games

#creating a data frame to be used for the average accuracy by game type bar plot
avg_data = {'Game_Type': ['Blitz', 'Lightning', 'Rapid'],
            'Average_Accuracy': [avg_blitz_accuracy, avg_lightning_accuracy, avg_rapid_accuracy]}

avg_accuracy = pd.DataFrame(avg_data)

# creating the bar plot
avg_acc_bar = avg_accuracy.plot.bar(x = 'Game_Type',
                      color = 'green',
                      edgecolor = 'black',
                      title = 'Bar Plot of Average Accuracy by Game Type',
                      xlabel = 'Game Type',
                      ylabel = 'Average Accuracy',
                      rot = 0,
                      legend = '')

caption = 'Viz by Robbie Goss | Data Collected from chess.com'
avg_acc_bar.text(0,-17, caption, ha='center')

plt.show()


# In[24]:


#Do I play to my competition level? 

#getting the counts of the wins, draws, and losses when my rating is better and worse (equal n/a due to sample size)
grouped = new_archive.groupby(["Better_Worse", "My_Result"]).size()

# creating the grouped bar plot
index = np.arange(2)

losses = [965, 407]
first_bar = plt.bar(index-0.25, 
                    losses, 
                    width=0.25, 
                    color = 'black', 
                    label = 'Losses')

wins = [460, 1163]
second_bar = plt.bar(index, 
                     wins, 
                     width=0.25, 
                     color = 'green', 
                     label='Wins', 
                     edgecolor = 'black')

draws = [92, 88]
second_bar = plt.bar(index+0.25, 
                     draws, 
                     width=0.25, 
                     color='gray', 
                     label='Draws', 
                     edgecolor = 'black')

plt.xticks(index, ['Worse', 'Better'])
plt.legend(loc = 'best')
plt.xlabel('Rating Difference')
plt.ylabel('Total L/W/D')

caption = 'Viz by Robbie Goss | Data Collected from chess.com'
plt.text(0.5,-250, caption, ha='right')

plt.show()


# In[25]:


'''
Creating 2 logistic regression models to find my probability of winning.

The first regression model is with 3199 games not analyzed for accuracy, and the second consists of 146 games that
I was able to analyze for accuracy.
'''
#making dummy variables to use in my model

# dummy variables for the game type (Rapid, Blitz, Lightning)
type_dummies = pd.get_dummies(new_archive.Game_Type)

# dummy variables for my piece color in a game
color_dummies = pd.get_dummies(new_archive.My_Color)

# dummy variables for the quarters of the year to see if I play better at different times of the year
quarter_dummies = pd.get_dummies(new_archive.Quarter)

#creating the df I will use for my model
dfs = [new_archive.My_Result, new_archive.Moves, new_archive.Rating_Diff, color_dummies, new_archive.GameId, quarter_dummies]
pre_log1 = pd.concat(dfs, axis=1)

# using pandasql to merge the dummy_archive data set with the final_accuracy dataset.


query = """
SELECT *
FROM analysis as a
JOIN pre_log1 as l
ON a.GameId = l.GameId;
"""

sqldf(query).head()

pre_log2 = sqldf(query)

# creating a variable to show my accuracy and a variable to show my opponents accuracy for this logistic regression 

conditions = [
    pre_log2['White'] == 1,
    pre_log2['White'] == 0
]

conditions2 = [
    pre_log2['White'] == 0,
    pre_log2['White'] == 1
]

values = [pre_log2['White_Accuracy'], pre_log2['Black_Accuracy']]
values2 = [pre_log2['White_Accuracy'], pre_log2['Black_Accuracy']]

my_accuracy = np.select(conditions, values)
opp_accuracy = np.select(conditions2, values2)

pre_log2['My_Accuracy'] = my_accuracy
pre_log2['Opp_Accuracy'] = opp_accuracy

#creating final dfs to be used for the regression models
log1 = pre_log1[['My_Result', 'Moves', 'Rating_Diff', 'Black', 'White']]
log2 = pre_log2[['My_Result', 'Moves', 'Rating_Diff', 'Black', 'White', 'My_Accuracy', 'Opp_Accuracy']]


# In[26]:


# My previous regression experience was with R, so I spent time learning to use sklearn in python for my model. 

# x1 and y1 are for the first logistic regression, x2 and y2 are for the second

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils

y1 = log1.loc[:, log1.columns == 'My_Result']
x1 = log1.loc[:, log1.columns != 'My_Result']

y2 = log2.loc[:, log2.columns == 'My_Result']
x2 = log2.loc[:, log2.columns != 'My_Result']

# fit_transform scales the data, making them usable for the regression models

lab = preprocessing.LabelEncoder()
y1_transformed = lab.fit_transform(y1)

y2_transformed = lab.fit_transform(y2)

logistic1 = LogisticRegression()
logistic1.fit(x1,y1_transformed)

logistic2 = LogisticRegression()
logistic2.fit(x2,y2_transformed)

'''
While I had not used anything similar to recursive feature estimation, it was interesting to try it out on my models to
see what appear to better 'features' for my model. 

Interestingly, the three best ranked 'features' for this regression are me playing as black, my accuracy and my opponent's
accuracy, but me playing black is not significant in the model while the 2 of the 3 features ranked lower are significant.
'''

# the four variables included were all highly ranked by the RFE
rfe1 = RFE(logistic1)
rfe1 = rfe1.fit(x1, y1_transformed)
print(rfe1.support_)
print(rfe1.ranking_)

rfe2 = RFE(logistic2)
rfe2 = rfe2.fit(x2, y2_transformed)
print(rfe2.support_)
print(rfe2.ranking_)


# In[27]:


import statsmodels.api as sm

#summary of logistic model 1
logit_model1 = sm.Logit(y1,x1)
result1 = logit_model1.fit()
print(result1.summary2())

#summary of logistic model 2
logit_model2 = sm.Logit(y2,x2)
result2 = logit_model2.fit()
print(result2.summary2())

'''
Some interesting things to note was that there were multiple different variables that I thought might be significant in
a model or meaningful in some way, but did not end up having much/any impact on the models. For example, I explored
seeing if the time of year impacted my win probability (breaking up the year month by month and also into quarters)
but that did not matter. I thought I would be more motivated in months when I had chess club during school, but I was
wrong. Additionally, once my accuracy and my opponent's accuracy were taken into account in my second regression model,
my starting color was no longer statistically significant. 
'''

