import streamlit as st
import pandas as pd
import joblib

# Cargar modelo
model = joblib.load("modelo_cripto.pkl")

# Cargar datos con caché
@st.cache_data
def load_data():
    return pd.read_csv("df_prueba.csv")

df = load_data()

# Título y descripción
st.title("📈 Recomendador de Criptomonedas con IA")
st.markdown("Esta herramienta predice criptomonedas de **baja capitalización** con alta probabilidad de crecer en los próximos meses.")

# Filtrar datos no válidos
df = df[(df["market_cap"] > 0) & (df["price_usd"] > 0)]

# Features usadas por el modelo
features = [
    "price_usd",
    "market_cap",
    "total_volume",
    "ai", "gaming", "memes", "rwa"
]

X = df[features]

# Predecir probabilidad de crecimiento
df["prob_crecimiento"] = model.predict_proba(X)[:, 1]

# Precio estimado más realista (crecimiento objetivo: 30%)
df["price_estimado"] = df["price_usd"] * (1 + df["prob_crecimiento"] * 0.30)

# Filtro: baja capitalización y alta probabilidad
umbral_lowcap = df["market_cap"].quantile(0.25)

recomendadas = df[
    (df["market_cap"] < umbral_lowcap) &
    (df["prob_crecimiento"] >= 0.5)
].copy()

# Ordenar por mayor probabilidad
recomendadas = recomendadas.sort_values("prob_crecimiento", ascending=False)

# Renombrar columnas para visualización
tabla_mostrar = recomendadas[[
    "id",
    "price_usd",
    "market_cap",
    "total_volume",
    "prob_crecimiento",
    "price_estimado"
]].head(10).rename(columns={
    "id": "Criptomoneda",
    "price_usd": "Precio Actual (USD)",
    "market_cap": "Capitalización de Mercado",
    "total_volume": "Volumen (24h)",
    "prob_crecimiento": "Prob. de Crecimiento",
    "price_estimado": "Precio Estimado"
})

# Mostrar resultados
st.subheader("🔝 Top 10 Criptos Recomendadas para Invertir")
st.dataframe(tabla_mostrar)

# Footer
st.markdown("---")
st.markdown("🔗 [GitHub](https://github.com/joyel124/TF_ML_GUI)")
