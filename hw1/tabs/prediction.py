import re
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tabs.eda import load_data_for_eda, clean_data
from tabs.model_info import load_model_dict


def process_name(s, process_variant=True):
    maker, model, *variant = s.lower().split()
    variant = " ".join(variant) if variant else ""

    if not process_variant:
        return maker, model, variant

    # Remove fuel types (we have 'fuel' feature)
    for w in ['diesel', 'petrol', 'lpg', 'cng']:
        variant = variant.replace(w, "")

    # Remove transmission types (we have 'transmission' feature)
    for w in ["mt", "at"]:
        variant = variant.replace(w, "")

    # Remove year ranges (we have 'year' feature)
    variant = re.sub(r"[0-9]+-[0-9]+\s*", "", variant)

    # Remove decimal numbers
    variant = re.sub(r"[0-9]+\.[0-9]+\s*", "", variant)

    # Remove parentheses
    variant = re.sub(r"[\(\)]", "", variant)

    variant = " ".join(variant.split())

    return maker, model, variant

def get_top_values(train, col, top=50):
    return list(train[col].value_counts()[:top].index)

def process_with_other(df, col):
    allowed_values = get_top_values(df, col, 50)
    df[col] = df[col].apply(lambda x: x if x in allowed_values else "other")
    return df

def process_data_for_model(df, model_dict):
    df = clean_data(df, drop_dups=False)
    scaler = model_dict["scaler"]
    num_cols = model_dict["num_features"]
    df[num_cols] = scaler.transform(df[num_cols])

    df[["maker", "model", "variant"]] = df.name.apply(process_name).apply(pd.Series)
    df = process_with_other(df, "maker")
    df = process_with_other(df, "model")
    df = process_with_other(df, "variant")
    df.drop("name", axis=1, inplace=True)
    
    ohe = model_dict["ohe"]
    cat_cols = model_dict["cat_features"]
    cat_ohe = ohe.transform(df[cat_cols])
    cat_ohe = pd.DataFrame(cat_ohe.toarray(), columns=ohe.get_feature_names_out())
    df = pd.concat([df, cat_ohe], axis=1)
    to_drop = [col for col in df if df[col].dtype == "object"]
    to_drop.append("seats")
    df = df.drop(to_drop, axis=1)
    return df

def process_tab_predict():
    st.write("""
        Вы можете получить предсказания модели двумя способами:
        1. Вручную ввести значения признаков объекта
        2. Загрузить CSV-файл с одним или несколькими объектами
    """)
    model_dict = load_model_dict()    
    model, features = model_dict["model"], model_dict["features"]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Введите данные объекта")
        with st.form("prediction_form"):
            df_train = load_data_for_eda()[0]
            fuel_unique = sorted(df_train.fuel.unique().tolist())
            seller_type_unique = sorted(df_train.seller_type.unique().tolist())
            transmission_unique = sorted(df_train.transmission.unique().tolist())
            owner_unique = sorted(df_train.owner.unique().tolist())

            name = st.text_input(
                "Бренд авто",
                value=df_train.name.dropna().sample(1).values[0]
            )
            year = st.number_input(
                "Год производства", 
                value=df_train.year.dropna().sample(1).values[0]
            )
            km_driven = st.number_input(
                "Пробег (в км)", 
                value=df_train.km_driven.dropna().sample(1).values[0]
            )
            mileage = st.text_input(
                "Расход топлива (литр/км)", 
                value=df_train.mileage.dropna().sample(1).values[0]
            )
            max_power = st.text_input(
                "Максимальная мощность двигателя (bhp)",
                value=df_train.max_power.dropna().sample(1).values[0]
            )
            torque = st.text_input(
                "Крутящий момент двигателя (Nm или kgm @ rpm)",
                value=df_train.torque.dropna().sample(1).values[0]
            )
            seats = st.number_input(
                "Кол-во сидений ",
                value=df_train.seats.dropna().sample(1).values[0]
            )
        
            seller_type = st.selectbox("Тип продавца", seller_type_unique)
            owner = st.selectbox("Кол-во бывших владельцев", owner_unique)        
            fuel = st.selectbox("Тип топлива", fuel_unique)
            transmission = st.selectbox("Тип трансмиссии", transmission_unique)
            engine = st.text_input(
                "Тип двигателя (CC)",
                value=df_train.engine.dropna().sample(1).values[0]
            )
            
            submitted = st.form_submit_button("Предсказать")

    with col2:
        st.subheader("Загрузите CSV-файл")
        uploaded_file = st.file_uploader("Загрузить CSV-файл", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            processed_df = process_data_for_model(input_df, model_dict)
            input_df["selling_price"] = model.predict(processed_df[features])
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            st.stop()
        
        st.subheader("📊 Результаты")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего объектов", len(input_df))
        with col2:
            st.metric("Предсказанная цена (мин.)", input_df["selling_price"].min())
        with col3:
            st.metric("Предсказанная цена (среднее)", input_df["selling_price"].mean())
        with col4:
            st.metric("Предсказанная цена (макс.)", input_df["selling_price"].max())

        st.subheader("📈 Визуализации")
        fig = plt.figure(figsize=(5, 5))
        sns.histplot(input_df["selling_price"])
        st.pyplot(fig, width="content")

    if submitted:
        try:
            input_df = pd.DataFrame({
                "name": name,
                "year": year,
                "km_driven": km_driven,
                "mileage": mileage,
                "max_power": max_power,
                "torque": torque,
                "seats": seats,
                "seller_type": seller_type,
                "owner": owner,
                "fuel": fuel,
                "transmission": transmission,
                "engine": engine,
            }, index=[0])
            processed_df = process_data_for_model(input_df, model_dict)
            preds = model.predict(processed_df[features])
            st.metric("Предсказанная цена", preds[0])
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            st.stop()
