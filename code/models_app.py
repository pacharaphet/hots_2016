import pandas as pd 
import numpy as np
import pickle 
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time 
start_time = time.time()
@st.cache(allow_output_mutation=True)
def load_models():
    """"Load in each of the 3 models for each of the 3 feature sets(9 total)"""
    pickle_in = open('../data/models/base_model.pkl', 'rb')
    pickle_in_rfb = open('../data/models/rf_baseline.pkl', 'rb')
    pickle_in_nbb = open('../data/models/nb_baseline.pkl', 'rb')
    pickle_in_log_maps = open('../data/models/logreg_maps.pkl', 'rb')
    pickle_in_rf_maps = open('../data/models/rf_maps.pkl', 'rb')
    pickle_in_nb_maps = open('../data/models/nb_maps.pkl', 'rb')
    pickle_in_log_length = open('../data/models/logreg_length.pkl', 'rb')
    pickle_in_rf_length = open('../data/models/rf_length.pkl', 'rb')
    pickle_in_nb_length = open('../data/models/nb_length.pkl', 'rb')
    base_model = pickle.load(pickle_in)
    rf_baseline = pickle.load(pickle_in_rfb)
    nb_baseline = pickle.load(pickle_in_nbb)
    logreg_maps = pickle.load(pickle_in_log_maps)
    rf_maps = pickle.load(pickle_in_rf_maps)
    nb_maps = pickle.load(pickle_in_nb_maps)
    logreg_length = pickle.load(pickle_in_log_length)
    rf_length = pickle.load(pickle_in_rf_length)
    nb_length = pickle.load(pickle_in_nb_length)

    return [base_model, rf_baseline, nb_baseline, logreg_maps, rf_maps, nb_maps, logreg_length, rf_length, nb_length]
@st.cache(allow_output_mutation=True)
def load_dfs():
    """Load the dfs with three different feature sets 
    baseline_df = heroes only
    maps_df = heroes and maps
    length_df = heroes, maps and gamelength"""
    df = pd.read_csv('../data/base_df.csv').drop(columns=['Unnamed: 0'])
    maps_df = pd.read_csv('../data/maps_df.csv').drop(columns='Unnamed: 0')
    length_df = pd.read_csv('../data/length_df.csv').drop(columns='Unnamed: 0')
    return [df, maps_df, length_df]


# DF_LIST = ['Select dataframe', 'baseline', 'maps_df', 'lengths_df']
# DF_DICT = {'baseline':dfs[0], 'maps_df': dfs[1], 'length_df':dfs[2]} 
# model_list = ['Select model', 'logistic regression', 'random forest', 'naive bayes']
# model_dict_baseline = {'logistic regression':models[0], 'random forest': models[1], 'naive bayes':models[2]}
# model_dict_maps = {'logistic regression':models[3], 'random forest': models[4], 'naive bayes':models[5]}
# model_dict_length = {'logistic regression':models[6], 'random forest': models[7], 'naive bayes':models[8]}

def get_train_test(df): 
    X = df.drop(columns='Is Winner')
    y = df['Is Winner']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    return X_train, X_test, y_train, y_test 


def main():
    models = load_models()
    dfs = load_dfs()
    DF_LIST = ['Select dataframe', 'baseline', 'maps_df', 'length_df']
    DF_DICT = {'baseline':dfs[0], 'maps_df': dfs[1], 'length_df':dfs[2]} 
    MODEL_LIST = ['Select model', 'logistic regression', 'random forest', 'naive bayes']
    BASELINE_MODELS = {'logistic regression':models[0], 'random forest': models[1], 'naive bayes':models[2]}
    MAP_MODELS = {'logistic regression':models[3], 'random forest': models[4], 'naive bayes':models[5]}
    LENGTH_MODELS = {'logistic regression':models[6], 'random forest': models[7], 'naive bayes':models[8]}
    title = """<div align="center">
        <p style="font-family: Garamond; color:Magenta; font-size: 46px;">Hots Predictor Models</p>
        </div>"""
    st.markdown(title, unsafe_allow_html=True)
    # title2 = """<div align="center">
    #     <p style="font-family: Garamond; color:Lightblue; font-size: 54px;">Predictor Models</p>
    #     </div>"""
    # st.markdown(title2, unsafe_allow_html=True)
     


    result = st.sidebar.selectbox('Please chooose a feature set:', DF_LIST)
    if result == 'Select dataframe': 
            st.write('Here you can select the feature set you wish to use for predictions as well as the model to be run.')
            st.write('')
            st.write('Select a feature set and a model to get started!')
            st.write('')
            st.write('baseline: heroes only')
            st.write('maps_df: heroes and maps')
            st.write('length_df: heroes, maps and binned game lengths')
    else:   
        
        X_train, X_test, y_train, y_test = get_train_test(DF_DICT[result])
        model_result = st.sidebar.selectbox('Please choose a model for predictions: ', MODEL_LIST)
        if model_result != 'Select model':

            try: 
                st.success(f'You selected the model: {model_result}')
                if result == 'baseline':
                    preds = BASELINE_MODELS[model_result].predict(X_test)
                elif result == 'maps_df':
                    preds = MAP_MODELS[model_result].predict(X_test)
                else: 
                    preds = LENGTH_MODELS[model_result].predict(X_test)
                st.write("Accuracy: ", round(accuracy_score(y_test, preds),3))
                st.write("Confusion matrix: \n", confusion_matrix(y_test, preds))
                st.write("Classification report:\n ", classification_report(y_test, preds))
            except KeyError: 
                pass
        else: 
            st.success(f'You selected the following dataframe: {result}')
            st.write('Select a model from the sidebar on the left:') 

if __name__ == '__main__':
    main()