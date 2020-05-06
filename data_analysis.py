import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot 
import plotly.express as px

df = pd.read_csv('data.csv')

# comparison of preferred foot over the different players
plt.rcParams['figure.figsize'] = (10, 5)
sns.countplot(df['Preferred Foot'], palette = 'pink')
plt.title('Most Preferred Foot of the Players', fontsize = 20)
plt.show()

df_overalll_analysis = df[['Name','Age', 'Club', 'Value', 'Wage','Position', 'Weak Foot', 'Acceleration', 'Aggression', 'Agility', 
                   'Preferred Foot', 'Balance', 'BallControl', 'Composure', 
                   'Crossing', 'Dribbling', 'FKAccuracy', 
                   'Finishing','Work Rate', 'GKDiving', 'GKHandling', 
                   'GKKicking', 'GKPositioning', 'GKReflexes', 
                   'HeadingAccuracy', 'Interceptions', 'Jumping', 
                   'LongPassing', 'LongShots', 'Marking', 'Penalties','Overall', 'Potential']]

h_labels = [x.replace('_', ' ').title() for x in 
            list(df_overalll_analysis.select_dtypes(include=['number', 'bool']).columns.values)]
corr = df_overalll_analysis.corr()
fig, ax = plt.subplots(figsize=(10,6))
sns_plot = sns.heatmap(corr, annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
sns_plot.figure.savefig('heatmap.png', dpi = 200)

df_composure = pd.pivot_table(df, index = 'Composure', values = ['Overall', 'Potential']).sort_values('Composure', ascending = False)
df_composure = df_composure.reset_index()


df_overalll_analysis['Value'] = df_overalll_analysis['Value'].apply(lambda x: x.split('€')[1])
df_overalll_analysis['Value'] = df_overalll_analysis['Value'].apply(lambda x: x.split('M')[0])
df_overalll_analysis['Value'] = df_overalll_analysis['Value'].apply(lambda x: x.split('K')[0])
df_overalll_analysis['Wage'] = df_overalll_analysis['Wage'].apply(lambda x: x.split('€')[1])
df_overalll_analysis['Wage'] = df_overalll_analysis['Wage'].apply(lambda x: x.split('K')[0])
df_overalll_analysis['Value'] = df_overalll_analysis['Value'].astype(float)
df_overalll_analysis['Wage'] = df_overalll_analysis['Wage'].astype(float)
df_overalll_analysis['Value'].sum()
df_overalll_analysis['Age'].sum()

df_overalll_analysis['Potential'].apply(lambda x: x>90)


# different positions acquired by the players 
plt.figure(figsize = (18, 8))
plt.style.use('fivethirtyeight')
ax = sns.countplot('Position', data = df, palette = 'bone')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
plt.savefig('positionsbar.png', dpi = 100)
plt.show()

# players work rate
plt.rcParams['figure.figsize'] = (10, 5)
sns.countplot(df['Work Rate'], palette = 'pink')
plt.xticks(rotation='vertical')
plt.title('players work rate', fontsize = 20)
plt.savefig('workrate.png', dpi = 100)
plt.show()

# best players per each position with their age, club, and nationality based on their overall scores
df.iloc[df.groupby(df['Position'])['Overall'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']]

df_club = df[df['Club'] == 'FC Barcelona']
df_club = df_club[['Name','Jersey Number','Position','Overall','Potential', 'Nationality','Age','Wage',
                                    'Value','Contract Valid Until', 'Crossing', 'Finishing',
                                    'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
                                    'LongPassing', 'BallControl', 'Acceleration','SprintSpeed',
                                    'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
                                    'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
                                    'Composure', 'Marking', 'StandingTackle', 'GKDiving', 'GKHandling', 'GKKicking',
                                    'GKPositioning', 'GKReflexes', 'Release Clause']]
df_club.shape
desc = pd.DataFrame(df_club.describe())

#finding suarez replcement
df_striker = df_overalll_analysis[(df_overalll_analysis['Potential'] > 85)&
                                           (df_overalll_analysis['Value'] < 50)&
                                           (df_overalll_analysis['Age'] <= 30)&
                                           (df_overalll_analysis['Position'] =='ST')
                                           ]




player_features = ('Acceleration', 'Aggression', 'Agility', 
                   'Balance', 'BallControl', 'Composure', 
                   'Crossing', 'Dribbling', 'FKAccuracy', 
                   'Finishing', 'GKDiving', 'GKHandling', 
                   'GKKicking', 'GKPositioning', 'GKReflexes', 
                   'HeadingAccuracy', 'Interceptions', 'Jumping', 
                   'LongPassing', 'LongShots', 'Marking', 'Penalties')

# Top four features for every position in football
for i, val in df.groupby(df['Position'])[player_features].mean().iterrows():
    print('Position {}: {}, {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))

from math import pi
idx = 1
plt.figure(figsize=(15,45))
for position_name, features in df_club.groupby(df_club['Name'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))
    
    # number of variable
    categories=top_features.keys()
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(9, 3, idx, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1
    
pd.pivot_table(df, index = 'Name', values = ['Overall', 'Potential']).sort_values('Overall', ascending = False)
 
df_youth_prodigy = df_overalll_analysis[(df_overalll_analysis['Potential'] > 85)&
                                           (df_overalll_analysis['Value'] < 50)&
                                           (df_overalll_analysis['Age'] <= 25)
                                           ]

fig = px.bar(df_youth_prodigy[['Name', 'Potential']].sort_values('Potential', ascending = 'False'),
            y = 'Potential', x = 'Name', color = 'Name', log_y = True,
             template = 'ggplot2', title = 'Youth Potential')
plot(fig)

fig = px.bar(df_youth_prodigy[['Name', 'Age']].sort_values('Age', ascending = 'False'),
            y = 'Age', x = 'Name', color = 'Name', log_y = True,
             template = 'ggplot2', title = 'Youth Age')
plot(fig)
player_features = ('Acceleration', 'Aggression', 'Agility', 
                   'Balance', 'BallControl', 'Composure', 
                   'Crossing', 'Dribbling', 'FKAccuracy', 
                   'Finishing', 'GKDiving', 'GKHandling', 
                   'GKKicking', 'GKPositioning', 'GKReflexes', 
                   'HeadingAccuracy', 'Interceptions', 'Jumping', 
                   'LongPassing', 'LongShots', 'Marking', 'Penalties')

# comparison of preferred foot over the different players
plt.rcParams['figure.figsize'] = (10, 5)
sns.countplot(df_youth_prodigy['Preferred Foot'], palette = 'pink')
plt.title('Most Preferred Foot of the Players', fontsize = 20)
plt.show()

# different positions acquired by the players 
plt.figure(figsize = (18, 8))
plt.style.use('fivethirtyeight')
ax = sns.countplot('Position', data = df_youth_prodigy, palette = 'bone')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
plt.savefig('positionsbar_youth.png', dpi = 100)
plt.show()


# youth players work rate
plt.rcParams['figure.figsize'] = (10, 5)
sns.countplot(df_youth_prodigy['Work Rate'], palette = 'pink')
plt.xticks(rotation='vertical')
plt.title('Youth players work rate', fontsize = 20)
plt.savefig('workrate_youth.png', dpi = 100)
plt.show()


# best players per each position with their age, club, and nationality based on their overall scores
df_bestinposition = df.iloc[df_youth_prodigy.groupby(df_youth_prodigy['Position'])['Potential'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality', 'Potential']]


# Top four features for every position in football
for i, val in df_youth_prodigy.groupby(df_youth_prodigy['Position'])[player_features].mean().iterrows():
    print('Position {}: {}, {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))


from math import pi
idx = 1
plt.figure(figsize=(15,45))
for position_name, features in df_youth_prodigy.groupby(df_youth_prodigy['Name'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))
    
    # number of variable
    categories=top_features.keys()
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = list(top_features.values())
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(9, 3, idx, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=11, y=1.1)
    
    idx += 1









