import phik
import re
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport

from constants import *
from tabs.model_info import load_model_dict


@st.cache_data
def load_data_for_eda():
    df_train = pd.read_csv(EDA_DATA_ROOT_URL + 'cars_train.csv')
    df_test = pd.read_csv(EDA_DATA_ROOT_URL + 'cars_test.csv')
    return df_train, df_test 

def clip_outliers_iqr(df, column, lower=None, upper=None):
    if (lower is None) and (upper is None):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

    df[column] = df[column].clip(lower, upper)
    return df, lower, upper

def process_torque(s):
    if not isinstance(s, str):
        return (None, None)
    
    s = s.lower()
    # replace all "at" with @
    s = re.sub(r"at", "@", s)
    # remove '(kgm @ rpm)'-type of substring
    s = re.sub(r"\([^)]*\)", "", s)
    parts = s.split("@")

    torque = None
    if len(parts) > 0:
        torque = parts[0]

        ratio = 1
        if "nm" in torque:
            # convert Nm to kgm
            ratio = 9.81

        # remove measures for torque (KGM, kgm)
        torque = re.sub(r"[^0-9.,]", "", torque)
        torque = float(torque) / ratio

    max_torque_rpm = None
    if len(parts) > 1:
        max_torque_rpm = parts[1]

        # remove measures for max_torque_rpm (rpm)
        max_torque_rpm = re.sub(r"[^0-9.,-]", "", max_torque_rpm)

        # if we have 1750-2500, take the maximum - 2500
        max_torque_rpm = max_torque_rpm.split("-")[-1]

        # remove commas
        max_torque_rpm = re.sub(r",", "", max_torque_rpm)
        max_torque_rpm = int(max_torque_rpm)

    return torque, max_torque_rpm

def process_max_power(s):
    if not isinstance(s, str):
        return None
    try:
        s = float(s.split()[0])
    except Exception as e:
        s = None
    return s

def prepare_data_for_eda(df_train, df_test, cols_to_impute):
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()
    df_train_copy["is_test"] = 0
    df_test_copy["is_test"] = 1
    df = pd.concat([df_train_copy, df_test_copy], ignore_index=True)
    
    df = clean_data(df)
    medians = df.loc[df.is_test == 0, cols_to_impute].median()
    df[cols_to_impute] = df[cols_to_impute].fillna(medians)
    df.engine = df.engine.astype(int)
    df.seats = df.seats.astype(int)

    df_train = df[df.is_test == 0].drop("is_test", axis=1)
    df_test = df[df.is_test == 1].drop("is_test", axis=1)

    return df_train, df_test

def clean_data(df, drop_dups=True):
    if drop_dups:
        columns = list(df.columns)
        if "selling_price" in columns:
            columns.remove("selling_price")
        df = df.drop_duplicates(
            subset=columns,
            keep='first'
        )
        df = df.reset_index(drop=True)

    df["mileage"] = df.mileage.apply(
        lambda x: float(x.split()[0]) if type(x) != float else x
    )
    df["engine"] = df.engine.apply(
        lambda x: float(x.split()[0]) if type(x) == str else x
    )
    df["max_power"] = df.max_power.apply(process_max_power)

    df[["torque_value", "max_torque_rpm"]] = (
        df["torque"].apply(process_torque).apply(pd.Series)
    )
    df.drop("torque", axis=1, inplace=True)
    
    return df

def process_tab_eda():
    try:
        df_train, df_test = load_data_for_eda()
    except Exception as e:
        st.write(f"Couldn't load data for EDA: {e}")
        st.stop()
    

    # Dataset Description Container
    with st.container(border=True):
        st.header("Dataset Description") 

        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.subheader("Original Dataframes")
            st.write("Ниже привидены исходные, неизмененные данные")
            st.markdown("**Original Train Dataframe**")
            st.dataframe(df_train)
            st.markdown("**Original Test Dataframe**")
            st.dataframe(df_test)
        
        with col2:
            st.subheader("Features Description")
            st.markdown("""
                | Признак | Описание |
                |------|-------------|
                | `name` | Бренд авто|
                | `year` | Год производства |
                | `selling_price` | Цена авто|
                | `km_driven` | Пробег (в км) |
                | `fuel` | Тип топлива |
                | `seller_type` | Тип продавца |
                | `transmission` | Тип трансмиссии |
                | `owner` | Кол-во бывших владельцев |
                | `mileage` | Расход топлива (литр/км) |
                | `engine` | Тип двигателя |
                | `max_power` | Максимальная мощность двигателя |
                | `torque` | Крутящий момент двигателя |
                | `seats` | Кол-во сидений |
            """)
        
        st.subheader("Initial Analysis")
        st.write("Базовые статистики для train датасета")
        profile = ProfileReport(df_train)
        components.html(
            profile.to_html(),
            scrolling=True,
            height=600,
        )


    # Cleaned Dataset Container
    with st.container(border=True):
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            model_dict = load_model_dict()
            cols_to_impute = model_dict["cols_to_impute"]
            df_train_clean, df_test_clean = prepare_data_for_eda(df_train, df_test, cols_to_impute)
            st.header("Cleaned Dataset") 
            st.write("Ниже привидены очищенные данные")
            st.markdown("**Clean Train Dataframe**")
            st.dataframe(df_train_clean)
            st.markdown("**Clean Test Dataframe**")
            st.dataframe(df_test_clean)
        with col2:
            st.subheader("Описание очистки данных")
            st.markdown("""
                Были выполнены следующие шаги по очистке данных:
                1. Удалены дубликаты в train и test
                2. Для признаков mileage, engine, max_power - извлечены числовые значения и удалены меры измерений
                3. Из признака torque были выделены два новых признака - torque value и  max_torque_rpm
                4. Пропущенные значения были заполнены медианами
                5. Признаки engine и seats были привидены к целочисленному типу (int) 
            """)


    # Numerical Features Analysis Container
    with st.container(border=True):
        st.header("Numerical Features Analysis") 
        st.subheader("Main Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Cleaned Train Dataframe")
            st.table(df_train_clean.describe())
        with col2:
            st.markdown("Cleaned Test Dataframe")
            st.table(df_test_clean.describe())
        
        st.subheader("Pair Plot")
        df_train_clean_clip = df_train_clean.copy() 
        df_test_clean_clip = df_test_clean.copy() 
        for col in ["km_driven", "torque_value", "max_torque_rpm"]:
            df_train_clean_clip, lower, upper = clip_outliers_iqr(df_train_clean_clip, col)
            df_test_clean_clip = clip_outliers_iqr(df_test_clean_clip, col, lower, upper)[0]

        st.markdown("""
            * В датасете есть аномальные значения у признаков 'km_driven', 'torque_value',
            'max_torque_rpm'. 
            * Для более наглядной визуализации, эти значения были clip'нуты 
            с помощью квантилей. 
            * Также, для производительности, графики построены с помощью подмножеств
            размера не более 2000 объектов.
            * Возможна задержка в 1-3 сек., прежде чем появятся графики.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Cleaned Clipped Train Dataframe")
            sns_plot = sns.pairplot(
                df_train_clean_clip.sample(2000),
                diag_kind="kde"
            )
            st.pyplot(sns_plot, width="content")
        with col2:
            st.markdown("Cleaned Clipped Test Dataframe")
            sns_plot = sns.pairplot(
                df_test_clean_clip,
                diag_kind="kde"
            )
            st.pyplot(sns_plot, width="content")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Pearson Correlation")
            st.markdown("* Корреляции построены для train датасета")
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(df_train_clean.corr(numeric_only=True), annot=True)
            st.pyplot(fig, width="content")
        with col2:
            st.subheader("Train/Test Distribution Comparison")
            st.markdown("""
                * В датасете есть аномальные значения у признаков 'km_driven', 'torque_value',
                'max_torque_rpm'. 
                * Для более наглядной визуализации, эти значения были clip'нуты 
                с помощью квантилей. 
            """)
            num_cols = [
                col for col in df_train_clean_clip.columns 
                if df_train_clean_clip[col].dtype != "object"
            ]
            fig, ax = plt.subplots(3, 3, figsize=(10, 10))
            ax = ax.flatten()
            for i, col in enumerate(num_cols):
                ax[i].hist(df_train_clean_clip[col], label="train", bins=20)
                ax[i].hist(df_test_clean_clip[col], label="test", bins=20)
                ax[i].set_title(col)
                ax[i].legend()
            plt.tight_layout()
            st.pyplot(fig, width="content")


    # Categorical Features Analysis Container
    with st.container(border=True):
        st.header("Categorical Features Analysis") 
        st.subheader("Main Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Cleaned Train Dataframe")
            st.table(df_train_clean.describe(include="object"))
        with col2:
            st.markdown("Cleaned Test Dataframe")
            st.table(df_test_clean.describe(include="object"))

    
    # Correlation Between All Features Container
    with st.container(border=True):
        st.header("Correlation Between All Features") 
        fig = plt.figure(figsize=(10, 10))
        ph = df_train_clean.phik_matrix()
        sns.heatmap(ph, cmap="Blues", annot=True)
        st.pyplot(fig, width="content")


    # Main Observations Container
    with st.container(border=True):
        st.header("Main Observations") 
        with open(ROOT_DIR / "observations.txt") as f:
            obs = "".join(f.readlines())
        st.markdown(obs)
