import math
import pandas as pd
from requests import session
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import roc_curve

@st.cache
def load_titanic() -> pd.DataFrame:
    """load titanic data

    Returns:
        pd.DataFrame: titanic data
    """
    return sns.load_dataset('titanic')

@st.cache
def load_iris() -> pd.DataFrame:
    """load iris data

    Returns:
        pd.DataFrame: iris data
    """
    df = px.data.iris()
    return df[df.species != "setosa"]

def show_df_with_padination(df:pd.DataFrame):
    """show data frame

    Args:
        df (pd.DataFrame): input data frame
    """
    rows_per_page = 10
    total_pages = math.ceil(len(df) / rows_per_page)
    
    if "page_df" not in st.session_state:
        st.session_state["page_df"] = 1
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        def minus_one_page() -> None:
            """minus one page
            """
            st.session_state['page_df'] -= 1
        if st.session_state['page_df'] > 1:
            st.button(label='<< Prev', on_click=minus_one_page)
    
    with col2:
        st.write(f"Page: {st.session_state['page_df']} / {total_pages}")
    
    with col3:
        def plus_one_page() -> None:
            """plus one page
            """
            st.session_state['page_df'] += 1
        if st.session_state['page_df'] < total_pages:
            st.button(label='>> Next', on_click=plus_one_page)
            
    start_iloc =(st.session_state['page_df'] - 1) * rows_per_page
    end_iloc = start_iloc + rows_per_page + 1
    st.write(df.iloc[start_iloc:end_iloc, :])
    
def main():
    # sidebar
    st.sidebar.title("data")
    option = st.sidebar.selectbox('select data', ('Iris', 'Titanic'))
    
    
    st.markdown("# " + option)
    if option == "Iris":
        df = load_iris()
    else:
        df = load_titanic()
    
    # pagination
    show_df_with_padination(df)
    
    # 3d visualization
    st.markdown("# Visualization")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_column = st.selectbox('x: ', (df.columns), 0)
    with col2:
        y_column = st.selectbox('y: ', (df.columns), 1)
    with col3:
        color_column = st.selectbox('color: ', (df.columns), 2)
    
    height, width = 700, 800
    
    fig = px.scatter(
        df, 
        x=x_column,
        y=y_column,
        color=color_column,
        height=height,
        width=width
    )
    st.plotly_chart(fig, height=height, width=width)
    
    # ml progress
    st.markdown("# Machine Learning")
    if option == "Iris":
        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = df['species_id'].replace(2, 0).replace(3, 1)
    else:
        df_ = df.dropna()
        X = df_[['pclass', 'age', 'sibsp', 'parch', 'fare', 'alone']]
        y = df_['survived']
        
    with st.spinner(text='Training...'):
        rf = RandomForestRegressor(n_estimators=100, random_state=0)
        y_pred = cross_val_predict(rf, X=X, y=y, cv=5)
        
        # evaluation
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        auc = metrics.auc(fpr, tpr)
        fig_eval = plt.figure(figsize=(12, 9))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
        plt.legend()
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.grid()
        st.pyplot(fig_eval)
        
if __name__ == '__main__':
    main()