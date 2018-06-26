# coding: utf-8

#
# ## International football results from 1872 to 2018
#
# An up-to-date dataset of nearly 40,000 international football results

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sn
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

# In[2]:

df = pd.read_csv('results.csv')
df.head()

# In[3]:

# df[df['tournament'].str.contains('FIFA', regex=True)]

# ### 获取所有的世界杯数据

# In[4]:

# df_FIFA_all including "FIFA World Cup" and "FIFA World Cup qualification"

df_FIFA_all = df[df['tournament'].str.contains('FIFA', regex=True)]
df_FIFA_all.head()

# In[5]:

df_FIFA = df_FIFA_all[df_FIFA_all['tournament'] == 'FIFA World Cup']
df_FIFA.head()

# In[6]:

df_FIFA.dtypes

# In[7]:

df_FIFA.loc[:, 'date'] = pd.to_datetime(df_FIFA.loc[:, 'date'])
df_FIFA.dtypes

# In[8]:

df_FIFA['year'] = df_FIFA['date'].dt.year
df_FIFA.head()

# In[9]:

df_FIFA.dtypes

# In[10]:

df_FIFA['diff_score'] = df_FIFA['home_score'] - df_FIFA['away_score']
df_FIFA.head()

# In[11]:

df_FIFA.dtypes

# In[12]:

df_FIFA['win_team'] = ''
df_FIFA.head()

# In[13]:

df_FIFA['diff_score'] = pd.to_numeric(df_FIFA['diff_score'])
df_FIFA.head()

# ### 创建一个新的列数据，包含获胜队伍的信息

# In[14]:

# The first method to get the winners

df_FIFA.loc[df_FIFA['diff_score'] > 0, 'win_team'] = df_FIFA.loc[
    df_FIFA['diff_score'] > 0, 'home_team']
df_FIFA.loc[df_FIFA['diff_score'] < 0, 'win_team'] = df_FIFA.loc[
    df_FIFA['diff_score'] < 0, 'away_team']
df_FIFA.loc[df_FIFA['diff_score'] == 0, 'win_team'] = 'Draw'

df_FIFA.head()

# In[15]:

# The second method to get the winners


def find_win_team(df):
    winners = []
    for i, row in df.iterrows():
        if row['home_score'] > row['away_score']:
            winners.append(row['home_team'])
        elif row['home_score'] < row['away_score']:
            winners.append(row['away_team'])
        else:
            winners.append('Draw')
    return winners


df_FIFA['winner'] = find_win_team(df_FIFA)
df_FIFA.head()

# ### 获取世界杯所有比赛获胜场数量最多的前20强数据
#
# 不含预选赛

# In[16]:

s = df_FIFA.groupby('win_team')['win_team'].count()
s.sort_values(ascending=False, inplace=True)
s

# In[17]:

s.drop(labels=['Draw'], inplace=True)
s

# In[18]:

s.head(5)

# In[19]:

s.head(20).plot(
    kind='bar', figsize=(10, 6), title='Top 20 Winners of World Cup')

# In[20]:

# s.sort_values(ascending=True,inplace=True)
# s

# In[21]:

s.sort_values(ascending=True, inplace=True)
s.tail(20).plot(
    kind='barh', figsize=(10, 6), title='Top 20 Winners of World Cup')

# In[22]:

s_percentage = s / s.sum()
s_percentage
s_percentage.tail(20).plot(
    kind='pie',
    figsize=(10, 10),
    autopct='%.1f%%',
    startangle=173,
    title='Top 20 Winners of World Cup',
    label='')

# ## 来查看部分国家的获胜情况

# In[23]:

s.get('China', default='NA')

# China，呵呵

# In[24]:

s.get('Japan', default='NA')

# In[25]:

s.get('Korea DPR', default='NA')

# In[26]:

s.get('Korea Republic', default='NA')

# In[27]:

s.get('Egypt', default='NA')

# 埃及，本次世界杯的黑马，之前在世界杯上也没有赢过~~

# In[28]:

# s.index

# ### 各个国家队进球总数量情况

# In[29]:

df_score_home = df_FIFA[['home_team', 'home_score']]
column_update = ['team', 'score']
df_score_home.columns = column_update
df_score_home

# In[30]:

df_score_away = df_FIFA[['away_team', 'away_score']]
df_score_away.columns = column_update
df_score_away

# In[31]:

df_score = pd.concat([df_score_home, df_score_away], ignore_index=True)
df_score

# In[32]:

s_score = df_score.groupby('team')['score'].sum()
s_score.sort_values(ascending=False, inplace=True)
s_score

# In[33]:

s_score.sort_values(ascending=True, inplace=True)
s_score.tail(20).plot(
    kind='barh', figsize=(10, 6), title='Top 20 in Total Scores of World Cup')

# ## 2018年世界杯32强分析

# 第一组：俄罗斯、德国、巴西、葡萄牙、阿根廷、比利时、波兰、法国
#
# 第二组：西班牙、秘鲁、瑞士、英格兰、哥伦比亚、墨西哥、乌拉圭、克罗地亚
#
# 第三组：丹麦、冰岛、哥斯达黎加、瑞典、突尼斯、埃及、塞内加尔、伊朗
#
# 第四组：塞尔维亚、尼日利亚、澳大利亚、日本、摩洛哥、巴拿马、韩国、沙特阿拉伯
#
#

# ** 判断是否有队伍首次打入世界杯 **

# In[50]:

team_list = [
    'Russia', 'Germany', 'Brazil', 'Portugal', 'Argentina', 'Belgium',
    'Poland', 'France', 'Spain', 'Peru', 'Switzerland', 'England', 'Colombia',
    'Mexico', 'Uruguay', 'Croatia', 'Denmark', 'Iceland', 'Costa Rica',
    'Sweden', 'Tunisia', 'Egypt', 'Senegal', 'Iran', 'Serbia', 'Nigeria',
    'Australia', 'Japan', 'Morocco', 'Panama', 'Korea Republic', 'Saudi Arabia'
]
for item in team_list:
    if item not in s_score.index:
        print(item)

# In[35]:

# 1. Iceland and Panama, the first time to top 32 of World Cup
# 2. Egypt, no win of tournament in World Cup

df_top32 = df_FIFA[(df_FIFA['home_team'].isin(team_list))
                   & (df_FIFA['away_team'].isin(team_list))]
df_top32.head()

# In[36]:

# s_score.index

# ### 32强赢球场数情况

# In[37]:

s_32 = df_top32.groupby('win_team')['win_team'].count()
s_32.sort_values(ascending=False, inplace=True)
s_32.drop(labels=['Draw'], inplace=True)
s_32.sort_values(ascending=True, inplace=True)
s_32.plot(
    kind='barh', figsize=(8, 12), title='Top 32 of World Cup since year 1872')

# ### 32强进球数量情况

# In[38]:

df_score_home_32 = df_top32[['home_team', 'home_score']]
column_update = ['team', 'score']
df_score_home_32.columns = column_update
df_score_away_32 = df_top32[['away_team', 'away_score']]
df_score_away_32.columns = column_update
df_score_32 = pd.concat(
    [df_score_home_32, df_score_away_32], ignore_index=True)
s_score_32 = df_score_32.groupby('team')['score'].sum()
s_score_32.sort_values(ascending=False, inplace=True)
s_score_32

# In[39]:

s_score_32.sort_values(ascending=True, inplace=True)
s_score_32.plot(
    kind='barh',
    figsize=(8, 12),
    title='Top 32 in Total Scores of World Cup since year 1872')

# ### 32强在最近10届世界杯的表现数据

# In[40]:

df_top32_10 = df_top32[df_top32['year'] >= 1978]
df_top32_10.head()

# #### 32强在最近10届世界杯的赢球场数情况

# In[41]:

s_32_10 = df_top32_10.groupby('win_team')['win_team'].count()
s_32_10.sort_values(ascending=False, inplace=True)
s_32_10.drop(labels=['Draw'], inplace=True)
s_32_10.sort_values(ascending=True, inplace=True)
s_32_10.plot(
    kind='barh', figsize=(8, 12), title='Top 32 of World Cup since year 1978')

# #### 32强在最近10届世界杯的进球总数量情况

# In[42]:

df_score_home_32_10 = df_top32_10[['home_team', 'home_score']]
column_update = ['team', 'score']
df_score_home_32_10.columns = column_update
df_score_away_32_10 = df_top32_10[['away_team', 'away_score']]
df_score_away_32_10.columns = column_update
df_score_32_10 = pd.concat(
    [df_score_home_32_10, df_score_away_32_10], ignore_index=True)
s_score_32_10 = df_score_32_10.groupby('team')['score'].sum()
s_score_32_10.sort_values(ascending=False, inplace=True)
s_score_32_10.head()

# In[43]:

s_score_32_10.sort_values(ascending=True, inplace=True)
s_score_32_10.plot(
    kind='barh',
    figsize=(8, 12),
    title='Top 32 in Total Scores of World Cup since year 1978')

# ### 32强在最近4届世界杯的表现数据

# #### 32强在最近4届世界杯的赢球场数情况

# In[44]:

df_top32_2002 = df_top32[df_top32['year'] >= 2002]
df_top32_2002.head()

# In[45]:

s_32_2002 = df_top32_2002.groupby('win_team')['win_team'].count()
s_32_2002.sort_values(ascending=False, inplace=True)
s_32_2002.drop(labels=['Draw'], inplace=True)
s_32_2002.sort_values(ascending=True, inplace=True)
s_32_2002.plot(
    kind='barh', figsize=(8, 12), title='Top 32 of World Cup since year 2002')

# #### 32强在最近4届世界杯的进球总数量情况

# In[46]:

df_score_home_32_2002 = df_top32_2002[['home_team', 'home_score']]
column_update = ['team', 'score']
df_score_home_32_2002.columns = column_update
df_score_away_32_2002 = df_top32_2002[['away_team', 'away_score']]
df_score_away_32_2002.columns = column_update
df_score_32_2002 = pd.concat(
    [df_score_home_32_2002, df_score_away_32_2002], ignore_index=True)
s_score_32_2002 = df_score_32_2002.groupby('team')['score'].sum()
s_score_32_2002.sort_values(ascending=False, inplace=True)
s_score_32_2002.head()

# In[47]:

s_score_32_2002.sort_values(ascending=True, inplace=True)
s_score_32_2002.plot(
    kind='barh',
    figsize=(8, 12),
    title='Top 32 in Total Scores of World Cup since year 2002')
