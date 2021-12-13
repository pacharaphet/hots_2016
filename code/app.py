import sys, os 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import chart_studio.plotly as py 
import plotly.express as px
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
from PIL import Image
from hero_functions import create_hero_df, best_game_length, best_lengths_by_map, hero_winrates, plot_winrates, highlight_max


header = st.container()
heroes = st.container()
dataset = st.container()
winrates = st.container()
model_training = st.container()
interactive = st.container()



def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    with header:
        title = """<div align="center">
        <p style="font-family: Garamond; color:Blue; font-size: 46px;">HotS Hero Data Analyzer </p>
        </div>"""
        st.markdown(title, unsafe_allow_html=True)
        subtitle = """<div align="center">
        <p style="font-family: Garamond; align: center; color:Green; font-size: 30px;">
        Hero winrates by map and game length</p>
        </div>"""
        st.markdown(subtitle, unsafe_allow_html=True)

        source = """<div align="right">
        <p style="font-family: Garamond; align: center; color:White; font-size: 20px;">
        Source: www.hotslogs.com</p>
        </div>"""
        st.markdown(source, unsafe_allow_html=True)

    with heroes: 
        my_data = pd.read_csv('../data/hots_base_df.csv')
        my_data.drop(columns=['Unnamed: 0', 'index'], inplace=True)
        heroes_list = my_data['Hero'].unique().tolist()
        heroes_list.sort()
        heroes_list.insert(0, '(--return home--)')
        result = st.sidebar.selectbox('Select a hero to see their stats', heroes_list)
        if result == '(--return home--)': 
            path = '../data/images/'
            dirs = os.listdir(path)
            image_list = []
            for item in dirs: 
                image_list.append(path+item)
            st.image(image_list, width=100, caption=heroes_list[1:])
        else: 
            
            st.image(f'../data/images/{result.lower()}resized.jpg', width=400, caption=result)        
            hero_df = hero_winrates(my_data, result)
            st.write('Overall winrate: ', round(hero_df['mean'].mean(),2))
            st.dataframe(highlight_max(hero_df))
            st.pyplot(plot_winrates(my_data, result))

    with winrates: 
        st.header('Heroes by overall winrates')
        dfHero = my_data[['Hero', 'Is Winner', 'Group']]
        dfHero = dfHero.groupby(["Hero","Is Winner"])["Hero"].count()
        dfHero = dfHero.unstack('Is Winner')
        dfHero.rename(columns={False : 'False', True:'True', 'winrate':'winrate', 'Group' : 'Group'}, inplace=True)
        dfHero['winrate'] = round(dfHero['True']/(dfHero['False']+dfHero['True']) * 100, 2)
        #st.write(dfHero)
        winrate_plot = dfHero['winrate'].plot(kind='bar', figsize=(12,10), ylim = (40, 61), color=['y', 'g', 'g', 'y', 'm',
                                                           'g','g', 'g', 'r', 'y', 'r',
                                                           'r', 'r', 'r', 'm', 'm', 'g',
                                                           'y', 'y', 'r', 'r', 'm', 'y',
                                                           'g', 'g', 'y', 'm', 'y', 'r',
                                                           'r', 'g', 'm', 'm', 'r', 'y', 'r'
                                                          ], title='Heroes by Winrate')    
        st.pyplot(winrate_plot.figure)
        st.pyplot()

if __name__ == '__main__': 
    main()
