import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from functii_online import nan_replace_df, salvare_matrice

# creeaza folderul de iesire daca nu exista deja
os.makedirs("data_out", exist_ok=True)

# ---------------------------------------------------------------
# Cerinta 1: incarcare
# Fisierul UCI foloseste ';' ca separator, nu ',' (de aceea aveai o singura coloana)
# ---------------------------------------------------------------
t = pd.read_csv("winequality-red.csv", sep=";")
print("shape:", t.shape)          # (numar randuri, numar coloane)
print(t.head())                   # primele 5 randuri
print(t.dtypes)                   # tipurile coloanelor

# ---------------------------------------------------------------
# Cerinta 2: valori lipsa
# ---------------------------------------------------------------
t = nan_replace_df(t)
# isna() da True/False pentru fiecare celula; sum() de doua ori le aduna pe toate
print("NaN ramase:", t.isna().sum().sum())   # trebuie sa fie 0

# ---------------------------------------------------------------
# Cerinta 3: filtrare quality >= 7, toate coloanele pastrate
# ---------------------------------------------------------------
t[t["quality"] >= 7].to_csv("data_out/vinuri_calitate_min7.csv", index=False)

# ---------------------------------------------------------------
# Cerinta 4: top 50 dupa alcool, descrescator
# ---------------------------------------------------------------
top50 = t.sort_values("alcohol", ascending=False).head(50)
top50.to_csv("data_out/top50_alcool.csv", index=False)

# ---------------------------------------------------------------
# Cerinta 5: media alcool si pH pentru fiecare nota de calitate
# groupby grupeaza randurile dupa quality; mean() face media pe grup.
# quality devine indexul, iar to_csv il scrie ca prima coloana.
# ---------------------------------------------------------------
medii = t.groupby("quality")[["alcohol", "pH"]].mean()
medii.to_csv("data_out/medii_pe_calitate.csv")

# ---------------------------------------------------------------
# Cerinta 6: coloana derivata aciditate_totala
# ---------------------------------------------------------------
t["aciditate_totala"] = t["fixed acidity"] + t["volatile acidity"] + t["citric acid"]
t[["quality", "alcohol", "aciditate_totala", "pH"]].to_csv(
    "data_out/vinuri_cu_aciditate.csv", index=False)

# ---------------------------------------------------------------
# Cerinta 7: nota de calitate cu media alcoolului maxima
# idxmax() returneaza INDEXUL (adica nota quality) unde media e maxima
# ---------------------------------------------------------------
quality_max = medii["alcohol"].idxmax()
print("Calitatea cu alcool mediu maxim:", quality_max)
pd.DataFrame({"quality": [quality_max]}).to_csv(
    "data_out/calitate_max_alcool.csv", index=False)

# ---------------------------------------------------------------
# Cerinta 8: matricea de corelatie (Pearson)
# ---------------------------------------------------------------
variabile = ["fixed acidity", "volatile acidity", "citric acid", "alcohol", "quality"]
R = t[variabile].corr()           # DataFrame cu nume pe linii si coloane
R.to_csv("data_out/R.csv")

# ---------------------------------------------------------------
# Cerinta 9: standardizare alcool
# Folosim deviatia standard de POPULATIE (ddof=0), ca la ACP,
# unde se imparte la n, nu la n-1. (pandas std() foloseste implicit ddof=1)
# ---------------------------------------------------------------
t["alcohol_std"] = (t["alcohol"] - t["alcohol"].mean()) / t["alcohol"].std(ddof=0)
t[["alcohol", "alcohol_std"]].to_csv("data_out/alcohol_standardizat.csv", index=False)

# ---------------------------------------------------------------
# Cerinta 10 + Bonus: matrice de distante euclidiene 20x20
# Standardizam fiecare coloana pe tot setul, apoi luam primele 20 de vinuri.
# ---------------------------------------------------------------
coloane_d = ["fixed acidity", "volatile acidity", "alcohol"]
std = (t[coloane_d] - t[coloane_d].mean()) / t[coloane_d].std(ddof=0)
primele20 = std.head(20).values   # array numpy 20 x 3

# cdist(A, A) calculeaza distanta dintre fiecare pereche de randuri
D = cdist(primele20, primele20, metric="euclidean")

etichete = [f"V{i}" for i in range(20)]
salvare_matrice(D, etichete, etichete, "data_out/d_euclid_20.csv")

print("Gata, fisierele sunt in data_out/")