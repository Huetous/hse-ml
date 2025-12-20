import streamlit as st
from constants import *
from tabs.eda import process_tab_eda
from tabs.model_info import process_tab_model
from tabs.prediction import process_tab_predict


st.set_page_config(
    page_title=TITLE,
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title(TITLE)

TABS = {
    "Tabs": "Select a tab",
    "📊 EDA": process_tab_eda,
    "ℹ️ Model Info": process_tab_model,
    "🔮 Make Predictions": process_tab_predict
}
TABS_TITLES = list(TABS.keys())

selected_tab = st.radio(
    "Select Tab",
    TABS_TITLES,
    horizontal=True,
    label_visibility="collapsed"
)

if selected_tab == TABS_TITLES[0]:
    st.write(TABS[TABS_TITLES[0]])
elif selected_tab == TABS_TITLES[1]:
    TABS[TABS_TITLES[1]]()
elif selected_tab == TABS_TITLES[2]:
    TABS[TABS_TITLES[2]]()
elif selected_tab == TABS_TITLES[3]:
    TABS[TABS_TITLES[3]]()