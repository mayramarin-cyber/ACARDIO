import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

# =====================================================
# CONFIGURACIÓN
# =====================================================
st.set_page_config(
    page_title="Predicción Cardiovascular",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Predicción de Riesgo Cardiovascular")

st.write("""
Esta aplicación utiliza un **modelo de Machine Learning (Random Forest calibrado)** 
para estimar la probabilidad de que una persona pertenezca al **grupo de riesgo cardiovascular**.

⚠ **Importante:**  
La probabilidad mostrada representa la **confianza del modelo**, NO un porcentaje médico real
de riesgo. El modelo clasifica según patrones aprendidos en los datos, pero **no reemplaza una
evaluación clínica profesional**.
""")


# =====================================================
# CARGAR MODELO
# =====================================================
MODEL_PATH = "Artefactos/v1/pipeline_RF_light.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encontró el modelo en: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()


# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predicción", "📊 Gráficos", "📘 Interpretación"])


# =====================================================
# TAB 1 - PREDICCIÓN
# =====================================================
with tab1:

    st.header("🔮 Predicción de riesgo cardiovascular")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Sexo", ["Hombre", "Mujer"])
        age = st.number_input("Edad (años)", 18, 100, 50)
        height = st.number_input("Altura (cm)", 120, 220, 165)
        weight = st.number_input("Peso (kg)", 40.0, 200.0, 70.0)
        ap_hi = st.number_input("Presión sistólica (ap_hi)", 80, 250, 120)

    with col2:
        ap_lo = st.number_input("Presión diastólica (ap_lo)", 50, 200, 80)
        cholesterol = st.selectbox("Colesterol", ["Normal", "Medio", "Alto"])
        gluc = st.selectbox("Glucosa", ["Normal", "Elevada", "Muy Elevada"])
        smoke = st.selectbox("Fuma", ["No fuma", "Fuma"])
        alco = st.selectbox("Consume alcohol", ["No consume alcohol", "Consume alcohol"])
        active = st.selectbox("Actividad física", ["Activo", "Inactivo"])

    # =====================================================
    # CREAR DATA
    # =====================================================
    input_data = pd.DataFrame({
        "gender": [gender],
        "age_years": [age],
        "height": [height],
        "weight": [weight],
        "ap_hi": [ap_hi],
        "ap_lo": [ap_lo],
        "cholesterol": [cholesterol],
        "gluc": [gluc],
        "smoke": [smoke],
        "alco": [alco],
        "active": [active],
    })

    input_data["BMI"] = input_data["weight"] / ((input_data["height"] / 100) ** 2)


    # =====================================================
    # BOTÓN DE PREDICCIÓN
    # =====================================================
    if st.button("Predecir riesgo", use_container_width=True):

        try:
            proba = float(model.predict_proba(input_data)[0][1])
            pred = 1 if proba >= 0.50 else 0

            # Resultado
            if pred == 1:
                st.error(f"⚠️ Riesgo cardiovascular — Probabilidad asignada: {proba:.2f}")
            else:
                st.success(f"✅ Sin riesgo — Probabilidad asignada: {proba:.2f}")

            # =====================================================
            # INFORME
            # =====================================================
            st.subheader("📄 Informe interpretado del resultado")

            st.info(f"""
### 📌 Interpretación de la probabilidad obtenida

El modelo asignó **{proba:.2f}**, lo que significa:

- 👉 **{proba:.0%} de confianza del modelo en su predicción actual**  
- ❗ **NO representa el porcentaje real de riesgo clínico**  
- Es una probabilidad basada en un modelo RandomForest calibrado  
""")

            # =====================================================
            # GAUGE
            # =====================================================
            st.subheader("📊 Indicador de riesgo (Gauge)")

            fig, ax = plt.subplots(figsize=(6, 2))
            ax.axis("off")

            colors = ["green", "yellow", "orange", "red"]
            thresholds = [0.25, 0.50, 0.75, 1.0]

            start = 0
            for c, t in zip(colors, thresholds):
                ax.barh(0, t - start, left=start, height=0.30, color=c)
                start = t

            ax.plot(proba, 0.15, marker="v", markersize=14, color="black")
            ax.text(proba, 0.42, f"{proba:.2f}", ha="center", fontsize=12)

            st.pyplot(fig)

            # =====================================================
            # RADIAL CORREGIDO
            # =====================================================
            st.subheader("📊 Perfil del paciente (Radial)")

            factor_labels = ["Edad", "PS Sistólica", "Colesterol", "Glucosa", "Fuma", "Actividad"]
            factor_vals = [
                age / 100,
                ap_hi / 200,
                ["Normal", "Medio", "Alto"].index(cholesterol) / 2,
                ["Normal", "Elevada", "Muy Elevada"].index(gluc) / 2,
                1 if smoke == "Fuma" else 0,
                1 if active == "Activo" else 0
            ]

            vals_closed = factor_vals + [factor_vals[0]]
            angles = np.linspace(0, 2*np.pi, len(vals_closed))

            fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            ax_r.plot(angles, vals_closed, "o-", linewidth=2)
            ax_r.fill(angles, vals_closed, alpha=0.25)
            ax_r.set_thetagrids(angles[:-1] * 180/np.pi, factor_labels)

            st.pyplot(fig_r)

        except Exception as e:
            st.error("Error durante la predicción.")
            st.code(str(e))



# =====================================================
# TAB 2 - GRÁFICOS DEL MODELO
# =====================================================
with tab2:

    st.header("📊 Gráficos del modelo entrenado")

    try:
        with open("Artefactos/v1/decision_policy.json") as f:
            dp = json.load(f)

        cm = np.array(dp["confusion_matrix"])
        labels = ["Sin riesgo", "Con riesgo"]

        fig1, ax1 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title("Matriz de Confusión")
        st.pyplot(fig1)

        metrics = dp["test_metrics"]
        fig2, ax2 = plt.subplots()
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax2)
        ax2.set_title("Métricas del Modelo")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    except Exception as e:
        st.warning("No se pudieron cargar los gráficos.")
        st.code(str(e))


# =====================================================
# TAB 3 - INTERPRETACIÓN
# =====================================================
with tab3:

    st.header("📘 Interpretación de métricas del modelo")

    st.write("""
    **Accuracy:** Qué porcentaje total de predicciones acertó el modelo.  
    **Precision:** Qué tan correctas son las predicciones positivas.  
    **Recall:** Capacidad del modelo para detectar casos de riesgo.  
    **F1-score:** Equilibrio entre precisión y recall.  
    **ROC-AUC:** Qué tan bien separa las clases.  
    """)

    try:
        st.json(dp["test_metrics"])
    except:
        st.info("No se pudieron cargar métricas.")
