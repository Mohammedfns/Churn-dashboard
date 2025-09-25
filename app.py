import os
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Charger les données
df_metrics = pd.read_csv(os.path.join(DATA_DIR, "metrics_monthly.csv"), parse_dates=["month"])
df_emp = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"), parse_dates=["hire_date","leave_date","dob"])
df_abs = pd.read_csv(os.path.join(DATA_DIR, "absences.csv"), parse_dates=["month"])

# Charger le modèle churn
MODEL_PATH = os.path.join(DATA_DIR, "churn_model.pkl")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Dashboard RH", layout="wide")

st.title("📊 Dashboard RH")
st.markdown("Bienvenue dans le tableau de bord RH interactif. Sélectionne un onglet pour explorer les KPIs.")

# ---- Onglets
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Effectif & Turnover",
    "📉 Absentéisme",
    "🧑‍💼 Recrutement",
    "🔮 Prédiction de churn"
])

# ---- Onglet 1 : Effectif & Turnover
with tab1:
    st.subheader("Évolution de l’effectif et du turnover")
    fig1 = px.line(df_metrics, x="month", y=["active_count","hires","leaves"], 
                   labels={"value":"Nombre","month":"Mois"}, 
                   title="Évolution de l’effectif, embauches et départs")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(df_metrics, x="month", y="turnover", 
                   title="Taux de turnover mensuel", 
                   labels={"turnover":"Turnover","month":"Mois"})
    st.plotly_chart(fig2, use_container_width=True)

# ---- Onglet 2 : Absentéisme
with tab2:
    st.subheader("Taux d’absentéisme par département")
    abs_by_dept = df_abs.groupby(["month","absence_type"]).size().reset_index(name="count")
    fig3 = px.area(abs_by_dept, x="month", y="count", color="absence_type", 
                   title="Types d’absences au fil du temps")
    st.plotly_chart(fig3, use_container_width=True)

    abs_dept = df_abs.merge(df_emp[["emp_id","department"]], on="emp_id")
    dept_abs = abs_dept.groupby("department")["absent_days"].mean().reset_index()
    fig4 = px.bar(dept_abs, x="department", y="absent_days", title="Jours moyens d’absence par département")
    st.plotly_chart(fig4, use_container_width=True)

# ---- Onglet 3 : Recrutement
with tab3:
    st.subheader("Efficacité du recrutement")
    fig5 = px.line(df_metrics, x="month", y="time_to_hire", 
                   title="Temps moyen pour recruter (time-to-hire)")
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.line(df_metrics, x="month", y="cost_per_hire", 
                   title="Coût moyen par embauche (€)")
    st.plotly_chart(fig6, use_container_width=True)

# ---- Onglet 4 : Prédiction churn
with tab4:
    st.subheader("Prédiction des employés à risque de départ")

    # Préparer features pour prédiction
    df_emp["tenure_days"] = (pd.to_datetime("today") - df_emp["hire_date"]).dt.days
    df_emp["age"] = (pd.to_datetime("today") - df_emp["dob"]).dt.days // 365

    features = df_emp[["gender","department","site","contract","salary","tenure_days","age"]].copy()
    features = pd.get_dummies(features, columns=["gender","department","site","contract"], drop_first=True)

    # Aligner colonnes avec modèle
    missing_cols = [c for c in model.feature_names_in_ if c not in features.columns]
    for c in missing_cols:
        features[c] = 0
    features = features[model.feature_names_in_]

    # Prédictions
    df_emp["churn_proba"] = model.predict_proba(features)[:,1]

    # Top 10 employés à risque
    top_risk = df_emp.sort_values("churn_proba", ascending=False).head(10)
    st.write("🔝 Top 10 employés à risque de départ :")
    st.dataframe(top_risk[["first_name","last_name","department","contract","salary","churn_proba"]])

    fig7 = px.histogram(df_emp, x="churn_proba", nbins=20, title="Distribution des probabilités de churn")
    st.plotly_chart(fig7, use_container_width=True)
