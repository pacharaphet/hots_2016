import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/hots_base_df.csv').drop(columns='Unnamed: 0')
maps = df['Map'].unique().tolist()
labels = ['very short', 'short', 'average', 'long', 'very long']
print(maps, labels)



def create_hero_df(df, hero_name):
    """Create a df with records for the hero specified
       Hero name must be a string"""
    return  df[df['Hero']==hero_name]

def best_game_length(df):
    """Returns a list of winrates for a given player based on game length"""
    best_winrate = 0
    winrates = []
    best_bin = ''
    for label in labels: 
        #print(label)
        length_df = df[df['Binned Length'] == label]
        df_wins = length_df[length_df['Is Winner'] == True]
        no_of_games = len(length_df)
        no_of_wins = len(df_wins)
        #print('Stats: ', no_of_games, no_of_wins)
        winrate = round((no_of_wins/no_of_games)*100, 2)
        #print(f'winrate = {winrate}')
        winrates.append(winrate)
        if winrate > best_winrate: 
            best_winrate = winrate
            best_bin = label
            #print('best: ', best_winrate, best_bin, '\n') 
        else: 
            continue
            #print('best: ', best_winrate, best_bin, '\n')    
        
        
    return winrates

def best_lengths_by_map(df):
    """creates a list of 7 lists with 3 elements each, showing the best maps and game lengths for a given hero"""
    winrates_by_map = []
    map_labels = []
    for map in maps: 
        try: 
            map_df = df[df['Map'] == map]
            winrates = best_game_length(map_df)
            winrates_by_map.append(winrates)
            map_labels.append(map)
        except ZeroDivisionError: 
            continue
    return winrates_by_map, map_labels

def hero_winrates(df, hero_name):
    hero_df = create_hero_df(df, hero_name)
    winrates, map_labels = best_lengths_by_map(hero_df)
    hero_wr_df = pd.DataFrame(winrates, columns=labels, index=map_labels)
    hero_wr_df['mean'] = hero_wr_df.mean(axis=1).round(2)
    return hero_wr_df

def plot_winrates(df, hero_name):
    hero_df = hero_winrates(df, hero_name)
    hero_df.plot(kind='bar')
    plt.ylim(45,70)
    plt.title(f'{hero_name} winrates by map and game length')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    

    return plt.show()

def highlight_max(df):
    return df.style.apply(lambda x: ["background: lightblue" if v == x.max() else "" for v in x], axis = 1)
    
