import numpy as np
import pandas as pd


def nan_replace_df(tabel):
    """
    Cerinta 2: inlocuieste valorile lipsa (NaN) din tot tabelul.
      - coloane numerice  -> se inlocuiesc cu media coloanei
      - coloane nenumerice -> se inlocuiesc cu valoarea cea mai frecventa (modul)
    Functia este generala: merge pe orice DataFrame, nu doar pe cel cu vin.
    """
    for coloana in tabel.columns:
        # sarim peste coloanele fara valori lipsa (nu avem ce face)
        if tabel[coloana].isna().any():
            if pd.api.types.is_numeric_dtype(tabel[coloana]):
                # mean() ignora automat NaN cand calculeaza media
                tabel[coloana] = tabel[coloana].fillna(tabel[coloana].mean())
            else:
                # mode() returneaza o serie; luam prima valoare
                tabel[coloana] = tabel[coloana].fillna(tabel[coloana].mode()[0])
    return tabel


def salvare_matrice(x, index_linie, index_coloana, fisier):
    """
    Bonus: salveaza o matrice numpy intr-un CSV, cu nume pe linii si coloane.
      x             -> matricea (numpy array 2D)
      index_linie   -> lista cu numele liniilor
      index_coloana -> lista cu numele coloanelor
      fisier        -> calea fisierului de iesire
    """
    tabel = pd.DataFrame(x, index=index_linie, columns=index_coloana)
    tabel.to_csv(fisier)