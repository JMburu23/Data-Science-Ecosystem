#!/usr/bin/env python
# coding: utf-8

# In[1]:


def one_dict(list_dict):
    keys=list_dict[0].keys()
    out_dict={key:[] for key in keys}
    for dict_ in list_dict:
        for key, value in dict_.items():
            out_dict[key].append(value)
    return out_dict    


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


dict_={'a':[11,21,31],'b':[12,22,32]}


# In[4]:


df=pd.DataFrame(dict_)
type(df)


# In[5]:


df.head()


# In[6]:


get_ipython().system('pip install nba_api')


# In[7]:


from nba_api.stats.static import teams
import matplotlib.pyplot as plt


# In[8]:


#https://pypi.org/project/nba-api/


# In[9]:


nba_teams = teams.get_teams()


# In[10]:


nba_teams[0:3]


# In[11]:


dict_nba_team=one_dict(nba_teams)
df_teams=pd.DataFrame(dict_nba_team)
df_teams.head()


# In[12]:


df_warriors=df_teams[df_teams['nickname']=='Warriors']
df_warriors


# In[13]:


id_warriors=df_warriors[['id']].values[0][0]
# we now have an integer that can be used to request the Warriors information 
id_warriors


# In[14]:


from nba_api.stats.endpoints import leaguegamefinder


# In[15]:


# Since https://stats.nba.com does not allow api calls from Cloud IPs and Skills Network Labs uses a Cloud IP.
# The following code is commented out, you can run it on jupyter labs on your own computer.
# gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=id_warriors)


# In[16]:


# Since https://stats.nba.com does not allow api calls from Cloud IPs and Skills Network Labs uses a Cloud IP.
# The following code is commented out, you can run it on jupyter labs on your own computer.
# gamefinder.get_json()


# In[17]:


# Since https://stats.nba.com does not allow api calls from Cloud IPs and Skills Network Labs uses a Cloud IP.
# The following code is comment out, you can run it on jupyter labs on your own computer.
# games = gamefinder.get_data_frames()[0]
# games.head()


# In[18]:


import requests

filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Labs/Golden_State.pkl"

def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)

download(filename, "Golden_State.pkl")


# In[19]:


file_name = "Golden_State.pkl"
games = pd.read_pickle(file_name)
games.head()


# In[20]:


games_home=games[games['MATCHUP']=='GSW vs. TOR']
games_away=games[games['MATCHUP']=='GSW @ TOR']


# In[21]:


games_home['PLUS_MINUS'].mean()


# In[22]:


games_away['PLUS_MINUS'].mean()


# In[23]:


fig, ax = plt.subplots()

games_away.plot(x='GAME_DATE',y='PLUS_MINUS', ax=ax)
games_home.plot(x='GAME_DATE',y='PLUS_MINUS', ax=ax)
ax.legend(["away", "home"])
plt.show()


# In[24]:


games_home['PTS'].mean()

games_away['PTS'].mean()


# In[ ]:




