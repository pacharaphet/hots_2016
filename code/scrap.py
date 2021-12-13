import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


naz_df = pd.read_csv('../data/naz_df.csv')

def plot_winrates(df, hero_name):
    df.plot(kind='bar')
    plt.ylim(45,60)
    plt.title(f'{hero_name} winrates by map and game length')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()

    return True

plot_winrates(naz_df, 'Nazeebo')
