import os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = DATA_DIR

# Charger les données
EMP = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"), parse_dates=["hire_date","leave_date","dob"])
ABS = pd.read_csv(os.path.join(DATA_DIR, "absences.csv"), parse_dates=["month"])
REC = pd.read_csv(os.path.join(DATA_DIR, "recruiting.csv"), parse_dates=["open_date","close_date"])

# Définir la période
earliest = EMP["hire_date"].min().to_pydatetime().date()
latest = pd.to_datetime(ABS["month"].max() if not ABS.empty else EMP["hire_date"].max()).date()
first_month = earliest.replace(day=1)
last_month = latest.replace(day=1)
months = pd.date_range(first_month, last_month, freq="MS").to_pydatetime().tolist()

rows = []
for m in months:
    m_start = pd.Timestamp(m).date()
    m_end = (pd.Timestamp(m) + relativedelta(months=1) - relativedelta(days=1)).date()

    # Effectif actif
    active = EMP[(EMP["hire_date"].dt.date <= m_end) & ((EMP["leave_date"].isna()) | (EMP["leave_date"].dt.date >= m_start))]
    active_count = len(active)

    # Entrées et sorties
    hires = EMP[(EMP["hire_date"].dt.date >= m_start) & (EMP["hire_date"].dt.date <= m_end)].shape[0]
    leaves = EMP[(EMP["leave_date"].notna()) & (EMP["leave_date"].dt.date >= m_start) & (EMP["leave_date"].dt.date <= m_end)].shape[0]

    # Absentéisme
    abs_rows = ABS[(ABS["month"].dt.date >= m_start) & (ABS["month"].dt.date <= m_end)]
    total_absent_days = int(abs_rows["absent_days"].sum()) if not abs_rows.empty else 0
    working_days = 21 * active_count if active_count > 0 else 0
    absenteeism_rate = total_absent_days / working_days if working_days > 0 else 0

    # Recrutement
    rec_rows = REC[(REC["close_date"].dt.date >= m_start) & (REC["close_date"].dt.date <= m_end)]
    hires_rec = int((rec_rows["status"] == "Hired").sum())
    time_to_hire = rec_rows["time_to_hire"].mean() if not rec_rows.empty else np.nan
    cost_per_hire = rec_rows["cost"].sum() / hires_rec if hires_rec > 0 else np.nan

    rows.append({
        "month": m_start,
        "active_count": active_count,
        "hires": hires,
        "leaves": leaves,
        "turnover": (leaves / active_count) if active_count > 0 else 0,
        "total_absent_days": total_absent_days,
        "absenteeism_rate": absenteeism_rate,
        "recruit_hires": hires_rec,
        "time_to_hire": time_to_hire,
        "cost_per_hire": cost_per_hire
    })

df_metrics = pd.DataFrame(rows)

# Sauvegarde
df_metrics.to_csv(os.path.join(OUT_DIR, "metrics_monthly.csv"), index=False)
print("✔ KPIs mensuels sauvegardés dans data/metrics_monthly.csv")
