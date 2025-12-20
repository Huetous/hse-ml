import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from constants import *


@st.cache_resource
def load_model_dict():
    """Load model via pickle"""
    with open(MODEL_DICT_PKL, "rb") as f:
        model_dict = pickle.load(f)
    return model_dict

def process_tab_model():
    try:
        model = load_model_dict()["model"]
    except Exception as e:
        st.write(f"Couldn't load the model: {e}")
        st.stop()

    # Model Description Container
    with st.container(border=True):
        st.header("Model Description") 
        st.write(model.__class__)
        st.write("Hyperparameters")
        st.write(model.get_params())

    # Weights Visualization Container
    with st.container(border=True):
        st.header("Weights Visualization") 
        fig = plt.figure(figsize=(10, 5))
        coef_ = model.coef_
        xticks = np.arange(len(coef_))
        ax = sns.barplot(x=xticks, y=coef_)
        fig.suptitle("Распределение весов модели")
        ax.set_xlabel("Номер веса")
        ax.set_ylabel("Величина веса")
        plt.xticks(rotation=45)
        plt.xticks(np.asarray(xticks[::10]))
        st.pyplot(fig, width="content")
