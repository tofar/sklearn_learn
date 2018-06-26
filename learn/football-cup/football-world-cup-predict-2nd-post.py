# coding: utf-8

# ## 作者相关信息
#
# * 作者：Lemonbit
# * 个人官网： **liyangbit.com**
# * 个人公众号： Python数据之道
#
# 文章地址：
#
# * [2018世界杯：用Python分析热门夺冠球队](http://liyangbit.com/projects/projects-world-cup-top3-predict/)
#
# * [第二波分析：德国是2018世界杯夺冠最大热门？ Python数据分析来揭开神秘面纱…](http://liyangbit.com/projects/projects-world-cup-predict-2nd-post)

#
# ## International football results from 1872 to 2018
#
# An up-to-date dataset of nearly 40,000 international football results

# In[139]:

import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sn
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

# In[140]:

# 解决matplotlib显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# In[141]:

df = pd.read_csv('results.csv')
df.head()

# In[142]:

mask = df['home_score'] - df['away_score']
df.loc[mask > 0, 'win_team'] = df.loc[mask > 0, 'home_team']
df.loc[mask < 0, 'win_team'] = df.loc[mask < 0, 'away_team']
df.loc[mask == 0, 'win_team'] = 'Draw'

df.head()

# In[143]:

df_FIFA_all = df[df['tournament'].str.contains('FIFA', regex=True)]
df_FIFA_all.head()

# ## 世界杯战绩 （含预选赛）

# **从前文来看**， 在世界杯历史上，实力最强的5支球队是 德国、阿根廷、巴西、法国、西班牙。
#
# 接下来，我们将比赛的范围扩大至包含世界杯预选赛，通过5支球队之间的比赛情况来进行分析。
#
# Top five countries: Germany, Argentina, Brazil, France, Spain

# In[144]:

team_top5 = ['Germany', 'Argentina', 'Brazil', 'France', 'Spain']
df_FIFA_top5 = df_FIFA_all[(df_FIFA_all['home_team'].isin(team_top5))
                           & (df_FIFA_all['away_team'].isin(team_top5))]
df_FIFA_top5

# In[145]:

df_FIFA_top5.shape

# 在世界杯历史上，5支球队在共有43场比赛相遇。
#
# 通过这43场比赛分析后，5支球队的胜负场数排名如下：

# In[146]:

s_FIFA_top5 = df_FIFA_top5.groupby('win_team')['win_team'].count()
s_FIFA_top5.drop('Draw', inplace=True)
s_FIFA_top5.sort_values(ascending=False, inplace=True)
s_FIFA_top5.plot(kind='bar', figsize=(10, 6), title='Top Five in World Cup')

# 下面，着重来分析下这5支球队，在世界杯上，两两对阵时的胜负情况。
#
# 首先自定义两个函数，分别获得两支球队获胜场数情况以及自定义绘图函数

# In[147]:


# 自定义函数，返回两支球队获胜场数情况
def team_vs(df, team_A, team_B):
    df_team_A_B = df[(df['home_team'].isin([team_A, team_B]))
                     & (df['away_team'].isin([team_A, team_B]))]
    s_win_team = df_team_A_B.groupby('win_team')['win_team'].count()
    return s_win_team


# 如需获取本文源代码，请关注公众号“Python数据之道”，
# 在公众号后台回复 “code” ，谢谢大家支持。

# In[148]:


# 自定义函数，两支球队在世界杯的对阵胜负情况制图
def team_vs_plot(df, team_A, team_B, ax):
    s_win_FIFA = team_vs(df, team_A, team_B)
    title = team_A + ' vs ' + team_B
    s_win_FIFA.plot(kind='bar', ax=ax)
    ax.set_xlabel('')
    ax.set_title(title, fontdict={'fontsize': 10})
    ax.set_xticklabels(s_win_FIFA.index, rotation=20)


# 基于上述函数，分析结果如下：

# ### 世界杯战绩：巴西 vs 其他4支球队

# In[149]:

f, axes = plt.subplots(figsize=(10, 10), ncols=2, nrows=2)
ax1, ax2, ax3, ax4 = axes.ravel()
team_vs_plot(df_FIFA_all, 'Brazil', 'Germany', ax=ax1)
team_vs_plot(df_FIFA_all, 'Brazil', 'Argentina', ax=ax2)
team_vs_plot(df_FIFA_all, 'Brazil', 'France', ax=ax3)
team_vs_plot(df_FIFA_all, 'Brazil', 'Spain', ax=ax4)

# 如需获取本文源代码，请关注公众号“Python数据之道”，
# 在公众号后台回复 “code” ，谢谢大家支持。

# set main title of the figure
plt.suptitle(
    'Brazil vs other Top 4 teams in World Cup',
    fontsize=14,
    fontweight='bold',
    x=0.5,
    y=0.94)
plt.show()

# **统计现象1：**
#
# 在世界杯上的战绩，统计获胜场数如下（不含平局）：
#
# 巴西1：1德国，巴西6：3阿根廷，巴西1：2法国，巴西3：1西班牙
#
# 巴西队，输赢不好判断……

# ### 世界杯战绩：德国 vs 其他3支球队

# In[150]:

f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=3)
ax1, ax2, ax3 = axes.ravel()

team_vs_plot(df_FIFA_all, 'Germany', 'Argentina', ax=ax1)
team_vs_plot(df_FIFA_all, 'Germany', 'France', ax=ax2)
team_vs_plot(df_FIFA_all, 'Germany', 'Spain', ax=ax3)

plt.subplots_adjust(hspace=0.4)
# subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
# default values
# left  = 0.125  # the left side of the subplots of the figure
# right = 0.9    # the right side of the subplots of the figure
# bottom = 0.1   # the bottom of the subplots of the figure
# top = 0.9      # the top of the subplots of the figure
# wspace = 0.2   # the amount of width reserved for blank space between subplots
# hspace = 0.2   # the amount of height reserved for white space between subplots

# 如需获取本文源代码，请关注公众号“Python数据之道”，
# 在公众号后台回复 “code” ，谢谢大家支持。

# set main title of the figure
plt.suptitle(
    'Germany vs other Top 3 teams in World Cup',
    fontsize=14,
    fontweight='bold',
    x=0.5,
    y=0.94)
plt.show()

# **统计现象2：**
#
# 在世界杯上的战绩，统计获胜场数如下（不含平局）：
#
# 德国4：1阿根廷，德国2：1法国，德国2：1西班牙
#
# 德国在这5支持球队里，获胜的优势相对比较明显。

# ### 世界杯战绩：阿根廷 vs 其他2支球队

# In[151]:

f, axes = plt.subplots(figsize=(10, 5), ncols=2, nrows=1)
ax1, ax2 = axes.ravel()

team_vs_plot(df_FIFA_all, 'Argentina', 'France', ax=ax1)
team_vs_plot(df_FIFA_all, 'Argentina', 'Spain', ax=ax2)

# plt.subplots_adjust(hspace = 0.4)
# subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
# default values
# left  = 0.125  # the left side of the subplots of the figure
# right = 0.9    # the right side of the subplots of the figure
# bottom = 0.1   # the bottom of the subplots of the figure
# top = 0.9      # the top of the subplots of the figure
# wspace = 0.2   # the amount of width reserved for blank space between subplots
# hspace = 0.2   # the amount of height reserved for white space between subplots

# set main title of the figure
plt.suptitle(
    'Argentina vs other Top 2 teams in World Cup',
    fontsize=14,
    fontweight='bold',
    x=0.5,
    y=0.99)
plt.show()

# **统计现象3：**
#
# 在世界杯上的战绩，统计获胜场数如下（不含平局）：
#
# 阿根廷2：0法国，阿根廷1：0西班牙
#
# 但阿根廷不敌巴西和德国

# ### 世界杯战绩：西班牙 vs 法国

# In[152]:

s_win_FIFA_Spain_France = team_vs(df_FIFA_all, 'Spain', 'France')
s_win_FIFA_Spain_France.plot(
    kind='bar', figsize=(10, 6), title='Spain vs France in World Cup')

# **结论：** 从历届世界杯上的表现情况来看，分析5强之间两两对阵后，发现德国队的表现是最好的。其次巴西和阿根廷的表现也不错。

# 考虑到，历届世界杯的数据，时间跨度很大，很多球队其实已经发生了很大变化。
#
# 球队真实的情况，可能选择近几年的比赛，以及包含不同级别的比赛，可能分析效果要更好些。
#
# 下面，重点来分析2014年以来包含所有比赛的情况。

# ## 2014年以来，所有比赛的战绩对比

# 首先，时间选择2014年之后（含2014年），距离现在的时间比较近，相对来说，球队人员的组成变化小一些。
#
# 当然，这里的时间选择，对于结果是有影响的。 大家可以探讨下这个因素带来的影响。
#

# ### 2014年以来所有球队所有比赛胜负情况概览

# 首先，我们来看看自2014年以来所有球队（国家队）所有比赛的胜负情况（以场数计），以便有一个大概的预期。

# In[153]:

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df_since_2014 = df[df['year'] >= 2014]
df_since_2014

# In[154]:

df_since_2014.shape

# 2014年以来，共有3600多场比赛。
#
# 针对3600多场比赛分析后，胜负场数情况如下：

# In[155]:

s_all = df_since_2014.groupby('win_team')['win_team'].count()
s_all.drop('Draw', inplace=True)
s_all.sort_values(ascending=True, inplace=True)
s_all.tail(50).plot(
    kind='barh',
    figsize=(8, 16),
    tick_label='',
    title='Top 50 in all tournament since 2014')

# 从上图来看，**2014年以来，墨西哥，法国，德国、葡萄牙、巴西、比利时、韩国和西班牙表现相对较好。**
#
# 结果是不是跟想象中的有些差异？
#
# 6月17日的小组赛，德国不敌墨西哥，看来也不是全无理由的。
#
# 但是，**本次我们主要还是要考虑32强之间的对阵，这样更能反映现实情况。**

# ### 2014年以来32强相互之间在所有比赛中的概览情况

# In[156]:

team_list = [
    'Russia', 'Germany', 'Brazil', 'Portugal', 'Argentina', 'Belgium',
    'Poland', 'France', 'Spain', 'Peru', 'Switzerland', 'England', 'Colombia',
    'Mexico', 'Uruguay', 'Croatia', 'Denmark', 'Iceland', 'Costa Rica',
    'Sweden', 'Tunisia', 'Egypt', 'Senegal', 'Iran', 'Serbia', 'Nigeria',
    'Australia', 'Japan', 'Morocco', 'Panama', 'Korea Republic', 'Saudi Arabia'
]

df_top32 = df_since_2014[(df_since_2014['home_team'].isin(team_list))
                         & (df_since_2014['away_team'].isin(team_list))]
df_top32

# In[157]:

s_top32 = df_top32.groupby('win_team')['win_team'].count()
s_top32.drop('Draw', inplace=True)
s_top32.sort_values(ascending=True, inplace=True)
s_top32.plot(
    kind='barh',
    figsize=(8, 12),
    tick_label='',
    title='Top 32 in all tournament since 2014')
# plt.ylabel('')

# 从上图来看，自2014年以来，**巴西、法国、葡萄牙、阿根廷、墨西哥、比利时、德国、西班牙、英国为前9强**。下面我们来分析top9之间的胜负情况。

# In[158]:

team_top9 = [
    'Brazil', 'France', 'Portugal', 'Argentina', 'Mexico', 'Belgium',
    'Germany', 'Spain', 'England'
]
df_top9 = df_since_2014[(df_since_2014['home_team'].isin(team_top9))
                        & (df_since_2014['away_team'].isin(team_top9))]
df_top9

# In[159]:

df_top9.shape

# 2014年以来，top 9 之间一共踢了44场比赛（包括友谊赛）。
#
# 总体来说，比赛的场数不是太多，基于这些数据来分析，可能对结果会有较大的影响。

# 九强排名如下：

# In[160]:

s_top9 = df_top9.groupby('win_team')['win_team'].count()
s_top9.drop('Draw', inplace=True)
s_top9.sort_values(ascending=False, inplace=True)
s_top9.plot(
    kind='bar', figsize=(10, 6), title='Top 9 in all tournament since 2014')

# **来查看下都统计了哪些类型的比赛**

# In[162]:

s_tournament = df_top9.groupby('tournament')['tournament'].count()
s_tournament_percentage = s_tournament / s_tournament.sum()
# s_tournament_percentage.sort_values(ascending=False, inplace=True)
s_tournament_percentage.tail(20).plot(
    kind='pie',
    figsize=(10, 10),
    autopct='%.1f%%',
    startangle=150,
    title='percentage of tournament',
    label='',
    labeldistance=1.25,
    pctdistance=1.08)

# 考虑到友谊赛在有些情况下可能不能比较准确的反映出球队的真实水平，且友谊赛站的场数比例较大，我们剔除友谊赛再来看看结果情况。

# ### 2014年以来32强剔除友谊赛后的胜负情况概览

# In[165]:

df_top9_no_friendly = df_top9[df_top9['tournament'] != 'Friendly']
df_top9_no_friendly.groupby('tournament')['tournament'].count()

# In[166]:

s_tournament_exclude_friendly = df_top9_no_friendly.groupby('tournament')[
    'tournament'].count()
s_tournament_percentage_exclude_friendly = s_tournament_exclude_friendly / s_tournament_exclude_friendly.sum(
)
s_tournament_percentage_exclude_friendly.sort_values(
    ascending=True, inplace=True)
s_tournament_percentage_exclude_friendly.plot(
    kind='pie',
    figsize=(10, 10),
    autopct='%.1f%%',
    startangle=150,
    title='percentage of tournament exclude friendly',
    label='',
    labeldistance=1.2,
    pctdistance=1.08)

# In[167]:

df_top9_no_friendly

# In[168]:

df_top9_no_friendly.shape

# In[169]:

s_top9_no_friendly = df_top9_no_friendly.groupby('win_team')[
    'win_team'].count()
s_top9_no_friendly.drop('Draw', inplace=True)
s_top9_no_friendly.sort_values(ascending=False, inplace=True)
s_top9_no_friendly.plot(
    kind='bar',
    figsize=(10, 6),
    title='Top 9 in all tournament(Friendly excluded) since 2014')

# 在概览中可以看出，是否剔除友谊赛（Friendly），对排名还是有影响的。
#
# 另外，剔除友谊赛后，总的比赛场数更少了（只有13场），9强之间有些队伍没有比赛，或者没有赢过，这个数据用来分析的作用更有限。
#
# 当然，在分析中**是否要剔除友谊赛，应该是值得商榷的。**

# 这里，我们后续分析采用包含友谊赛的数据，来分别分析9强之间两两对阵的情况，看看哪支球队的胜率更高些。

# **自定义获取球队某年至今获胜比例函数**

# In[170]:


# 自定义获取球队某年至今获胜比例函数
def probability(df, year, team_A, team_B):
    prob = []
    df_year = df[df['year'] >= year]
    s = team_vs(df_year, team_A, team_B)
    s_team_A = 0 if s.get(team_A) is None else s.get(team_A)
    s_A_win = s_team_A / s.sum()
    s_team_B = 0 if s.get(team_B) is None else s.get(team_B)
    s_B_win = s_team_B / s.sum()
    s_draw = 1 - s_A_win - s_B_win
    prob.append(year)
    prob.append(s_A_win)
    prob.append(s_B_win)
    prob.append(s_draw)
    #     print('{0} vs {1} since {2}'.format(team_A,team_B,year))
    #     print('Probability of {0} winning: {1:.2%}'.format(team_A, s_A_win))
    #     print('Probability of {0} winning: {1:.2%}'.format(team_B, s_B_win))
    #     print('Probability of draw: {0:.2%}'.format(s_draw))
    return prob


# **自定义获取两支球队历史获胜情况对比函数**

# In[171]:


# 自定义获取两支球队历史获胜情况对比函数
def his_team_data(df, year_start, year_end, team_A, team_B):
    row_team = []
    #     df_team = pd.DataFrame(columns=('year', 'team_A_win', 'team_B_win', 'draw'))
    for yr in list(range(year_start, year_end + 1)):
        team_A_vs_team_B = probability(df, yr, team_A, team_B)
        row_team.append(team_A_vs_team_B)
    team_A_win = team_A + '_win_percentage'
    team_B_win = team_B + '_win_percentage'
    df_team = pd.DataFrame(
        row_team, columns=('year', team_A_win, team_B_win, 'draw_percentage'))
    return df_team


# **自定义两支球队历史获胜情况制图函数**

# In[172]:


# 自定义两支球队历史获胜情况制图函数
def team_plot(df, year_start, year_end, team_A, team_B, ax):
    team_A_vs_team_B = team_A + '_vs_' + team_B
    team_A_vs_team_B = his_team_data(df, year_start, year_end, team_A, team_B)
    title = team_A + ' vs ' + team_B
    columns = [
        team_A + '_win_percentage', team_B + '_win_percentage',
        'draw_percentage'
    ]
    team_A_vs_team_B.set_index('year')[columns].plot(
        kind='line', figsize=(10, 6), title=title, ax=ax)


# ### 巴西 vs 德国

# In[173]:

team_A = 'Brazil'
team_B = 'Germany'

f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
ax1, ax2 = axes.ravel()

team_plot(df, 1930, 2016, team_A, team_B, ax1)
ax1.set_xlabel('')
team_plot(df, 2000, 2016, team_A, team_B, ax2)
ax2.set_title('')

plt.show()

# ### 德国 vs 8支球队

# In[174]:

team_A = 'Germany'

for team in [
        'France', 'Portugal', 'Argentina', 'Mexico', 'Belgium', 'Brazil',
        'Spain', 'England'
]:
    team_B = team

    f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    team_plot(df, 1930, 2016, team_A, team_B, ax1)
    ax1.set_xlabel('')
    team_plot(df, 2000, 2016, team_A, team_B, ax2)
    ax2.set_title('')

    plt.show()

# ### 巴西 vs 8支球队

# In[175]:

team_A = 'Brazil'

for team in [
        'France', 'Portugal', 'Argentina', 'Mexico', 'Belgium', 'Germany',
        'Spain', 'England'
]:
    team_B = team

    f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    team_plot(df, 1930, 2016, team_A, team_B, ax1)
    ax1.set_xlabel('')
    team_plot(df, 2000, 2016, team_A, team_B, ax2)
    ax2.set_title('')

    plt.show()

# ### 法国 vs 7支球队

# In[176]:

team_A = 'France'

for team in [
        'Portugal', 'Argentina', 'Mexico', 'Belgium', 'Germany', 'Spain',
        'England'
]:
    team_B = team

    f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    team_plot(df, 1930, 2016, team_A, team_B, ax1)
    ax1.set_xlabel('')
    team_plot(df, 2000, 2016, team_A, team_B, ax2)
    ax2.set_title('')

    plt.show()

# ### 葡萄牙 vs 6支球队

# In[177]:

team_A = 'Portugal'

for team in ['Argentina', 'Mexico', 'Belgium', 'Germany', 'Spain', 'England']:
    team_B = team

    f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    team_plot(df, 1930, 2016, team_A, team_B, ax1)
    ax1.set_xlabel('')
    team_plot(df, 2000, 2016, team_A, team_B, ax2)
    ax2.set_title('')

    plt.show()

# ### 阿根廷 vs 5支球队

# In[178]:

team_A = 'Argentina'

for team in ['Mexico', 'Belgium', 'Germany', 'Spain', 'England']:
    team_B = team

    f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    team_plot(df, 1930, 2016, team_A, team_B, ax1)
    ax1.set_xlabel('')
    team_plot(df, 2000, 2016, team_A, team_B, ax2)
    ax2.set_title('')

    plt.show()

# ### 墨西哥 vs 4支球队

# In[179]:

team_A = 'Mexico'

for team in ['Belgium', 'Germany', 'Spain', 'England']:
    team_B = team

    f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    team_plot(df, 1930, 2016, team_A, team_B, ax1)
    ax1.set_xlabel('')
    team_plot(df, 2000, 2016, team_A, team_B, ax2)
    ax2.set_title('')

    plt.show()

# ### 比利时 vs 3支球队

# In[180]:

team_A = 'Belgium'

for team in ['Germany', 'Spain', 'England']:
    team_B = team

    f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    team_plot(df, 1930, 2016, team_A, team_B, ax1)
    ax1.set_xlabel('')
    team_plot(df, 2000, 2016, team_A, team_B, ax2)
    ax2.set_title('')

    plt.show()

# ### 德国 vs 2支球队

# In[181]:

team_A = 'Germany'

for team in ['Spain', 'England']:
    team_B = team

    f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    team_plot(df, 1930, 2016, team_A, team_B, ax1)
    ax1.set_xlabel('')
    team_plot(df, 2000, 2016, team_A, team_B, ax2)
    ax2.set_title('')

    plt.show()

# ### 西班牙 vs 英国

# In[182]:

team_A = 'Spain'

for team in ['England']:
    team_B = team

    f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    team_plot(df, 1930, 2016, team_A, team_B, ax1)
    ax1.set_xlabel('')
    team_plot(df, 2000, 2016, team_A, team_B, ax2)
    ax2.set_title('')

    plt.show()

# In[183]:

# team_list = ['Russia', 'Germany', 'Brazil', 'Portugal', 'Argentina', 'Belgium', 'Poland', 'France',
#              'Spain', 'Peru', 'Switzerland', 'England', 'Colombia', 'Mexico', 'Uruguay', 'Croatia',
#             'Denmark', 'Iceland', 'Costa Rica', 'Sweden', 'Tunisia', 'Egypt', 'Senegal', 'Iran',
#             'Serbia', 'Nigeria', 'Australia', 'Japan', 'Morocco', 'Panama', 'Korea Republic', 'Saudi Arabia']

team_A = 'Argentina'
team_B = 'Croatia'

f, axes = plt.subplots(figsize=(6, 12), ncols=1, nrows=2)
ax1, ax2 = axes.ravel()

team_plot(df, 1930, 2016, team_A, team_B, ax1)
ax1.set_xlabel('')
team_plot(df, 2000, 2016, team_A, team_B, ax2)
ax2.set_title('')

plt.show()
