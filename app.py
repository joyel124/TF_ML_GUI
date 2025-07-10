import streamlit as st
import pandas as pd
import joblib

model = joblib.load("modelo_cripto.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("df_prueba.csv")

df = load_data()

st.title("📈 Recomendador de Criptomonedas con IA")
st.markdown("Esta herramienta predice criptomonedas de **baja capitalización** con alta probabilidad de crecer en los próximos meses.")

features = [
    "price_usd",
    "market_cap",
    "total_volume",
    "ai", "gaming", "memes", "rwa"
]

X = df[features]
df["prob_crecimiento"] = model.predict_proba(X)[:, 1]

umbral_lowcap = df["market_cap"].quantile(0.25)
recomendadas = df[
    (df["market_cap"] < umbral_lowcap) &
    (df["prob_crecimiento"] >= 0.5)
].copy()

recomendadas = recomendadas.sort_values("prob_crecimiento", ascending=False)

st.subheader("🔝 Top 10 Criptos Recomendadas para Invertir")
st.dataframe(recomendadas[["id", "market_cap", "prob_crecimiento"]].head(10))

st.download_button(
    label="⬇️ Descargar todas las recomendaciones",
    data=recomendadas.to_csv(index=False),
    file_name="criptos_recomendadas.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("🔗 [GitHub](https://github.com/joyel124/TF_ML_GUI)")