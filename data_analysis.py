import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
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
df_overalll_analysis = df[['Acceleration', 'Aggression', 'Agility', 
                   'Balance', 'BallControl', 'Composure', 
                   'Crossing', 'Dribbling', 'FKAccuracy', 
                   'Finishing', 'GKDiving', 'GKHandling', 
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



