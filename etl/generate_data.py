import os
import random
from faker import Faker
import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

random.seed(42)
np.random.seed(42)
fake = Faker("fr_FR")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Paramètres
N_EMP = 220
MONTHS = 24
END = date.today().replace(day=1) - relativedelta(days=1)   # fin = dernier jour mois précédent
START = (END.replace(day=1) - relativedelta(months=MONTHS-1)).replace(day=1)
DEPTS = ["Finance","RH","Ventes","Marketing","IT","Ops"]
SITES = ["Paris","Lyon","Lille","Marseille","Nantes"]
CONTRACTS = ["CDI","CDD","Alternance","Stage"]

# 1) employés
employees = []
for i in range(1, N_EMP+1):
    gender = random.choice(["M", "F"])
    first = fake.first_name_male() if gender == "M" else fake.first_name_female()
    last = fake.last_name()
    dob = fake.date_of_birth(minimum_age=22, maximum_age=60)
    hire_date = fake.date_between(start_date=START - relativedelta(months=18), end_date=END)
    leave_date = None
    if random.random() < 0.18:  # environ 18% des employés quittent
        leave_candidate = hire_date + relativedelta(days=random.randint(120, 1400))
        if leave_candidate <= END:
            leave_date = leave_candidate
    dept = random.choices(DEPTS, weights=[0.12,0.08,0.30,0.15,0.25,0.10])[0]
    site = random.choice(SITES)
    contract = random.choices(CONTRACTS, weights=[0.7,0.15,0.1,0.05])[0]
    salary = int(np.random.normal(loc=3200, scale=700))
    salary = max(1200, salary)
    employees.append({
        "emp_id": i,
        "first_name": first,
        "last_name": last,
        "gender": gender,
        "dob": dob,
        "department": dept,
        "site": site,
        "contract": contract,
        "hire_date": hire_date,
        "leave_date": leave_date,
        "salary": salary
    })

df_emp = pd.DataFrame(employees)

# 2) absences (par mois)
def month_periods(start, end):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + relativedelta(months=1)

attendance_rows = []
for _, row in df_emp.iterrows():
    for m_start in month_periods(START, END):
        m_end = (m_start + relativedelta(months=1) - relativedelta(days=1))
        if row["hire_date"] <= m_end and (pd.isna(row["leave_date"]) or row["leave_date"] >= m_start):
            if random.random() < 0.35:
                days = int(max(1, np.random.poisson(1.2)))
                attendance_rows.append({
                    "emp_id": row["emp_id"],
                    "month": pd.to_datetime(m_start).date(),
                    "absent_days": days,
                    "absence_type": random.choices(["Maladie","Congé","Autre"], weights=[0.5,0.45,0.05])[0]
                })

df_abs = pd.DataFrame(attendance_rows)

# 3) recrutement
recruit_rows = []
for m_start in month_periods(START, END):
    n_open = random.randint(2,8)
    for j in range(n_open):
        open_date = pd.to_datetime(m_start) + pd.Timedelta(days=random.randint(0,15))
        time_to_hire = max(7, int(np.random.normal(30, 10)))
        close_date = open_date + pd.Timedelta(days=time_to_hire)
        status = random.choices(["Hired","Closed-no-hire"], weights=[0.7,0.3])[0]
        offers = random.randint(1,6)
        accepts = int(max(0, round(offers * (0.4 + random.random()*0.5))))
        recruit_rows.append({
            "req_id": f"{m_start}-{j}",
            "open_date": open_date.date(),
            "close_date": close_date.date(),
            "status": status,
            "offers_made": offers,
            "offers_accepted": min(accepts, offers),
            "time_to_hire": time_to_hire,
            "cost": random.randint(800, 7000)
        })

df_rec = pd.DataFrame(recruit_rows)

# Sauvegarde
df_emp.to_csv(os.path.join(OUT_DIR, "employees.csv"), index=False)
df_abs.to_csv(os.path.join(OUT_DIR, "absences.csv"), index=False)
df_rec.to_csv(os.path.join(OUT_DIR, "recruiting.csv"), index=False)

print("✔ Données générées dans /data")
print("Employés:", len(df_emp), "| Absences:", len(df_abs), "| Recrutement:", len(df_rec))
