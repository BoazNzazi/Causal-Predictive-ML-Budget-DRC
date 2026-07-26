# ============================================================
# EDA — ANALYSE EXPLORATOIRE DES DONNÉES
# DRC_FiscalPanel_1996_2023.xlsx
# ============================================================
# SORTIES :
#   ../data/processed/ResultatML/
#       eda_01_stats_descriptives.xlsx
#       eda_02_completude.png
#       eda_03_evolution_budget_total.png
#       eda_04_taux_exec_par_fonction.png
#       eda_05_heatmap_correlation.png
#       eda_06_distribution_taux_exec.png
#       eda_07_evolution_macro.png
#       eda_08_anomalies_temporelles.png
#       eda_09_top_fonctions.png
#       eda_10_boxplot_taux_par_fonction.png
#       eda_summary.json   ← pour dashboard React
# ============================================================

print("=" * 65)
print("EDA — ANALYSE EXPLORATOIRE")
print("DRC_FiscalPanel_1996_2023")
print("=" * 65)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# ── Chemins
DATA_DIR   = Path("../data/processed")
RESULT_DIR = DATA_DIR / "ResultatML"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

F_FINAL = DATA_DIR / "DRC_FiscalPanel_1996_2023.xlsx"

# ── Style global
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
})
PALETTE = "#1A3A5C"
ACCENT  = "#E84B3A"
SOFT    = "#4C9BE8"

# ── Lecture
print("\nChargement : " + str(F_FINAL))
df = pd.read_excel(F_FINAL, engine="openpyxl")
df.columns = [str(c).strip() for c in df.columns]

col_fct = next(c for c in df.columns if "fonction" in c.lower())
col_ann = next(c for c in df.columns if "annee"    in c.lower())
df[col_ann] = df[col_ann].astype(int)

print("  " + str(len(df)) + " lignes × " +
      str(len(df.columns)) + " colonnes")
print("  " + str(df[col_fct].nunique()) + " fonctions | " +
      str(df[col_ann].nunique()) + " années (" +
      str(int(df[col_ann].min())) + "–" +
      str(int(df[col_ann].max())) + ")")


# ============================================================
# 1. STATISTIQUES DESCRIPTIVES
# ============================================================

print("\n" + "─" * 65)
print("1. STATISTIQUES DESCRIPTIVES")
print("─" * 65)

VARS_NUM = [
    "Budget_Depense_Courante", "Execution_Depense_Courante",
    "Budget_Depense_Capital",  "Execution_Depense_Capital",
    "TAUX_EXEC_COURANT",       "TAUX_EXEC_CAPITAL",
    "PIB_PAR_HABITANT_USD",    "CROISSANCE_PIB_PCT",
    "INFLATION_CPI_PCT",       "TAUX_CHANGE_CDF_USD",
    "RECETTES_ETAT_PCT_PIB",   "RENTE_RESSOURCES_PCT_PIB",
    "AIDE_ODA_USD",            "WGI_EFFICACITE_GOUV",
    "WGI_CONTROLE_CORRUPTION", "SOLDE_BUDGETAIRE_PCT_PIB",
    "DETTE_PUBLIQUE_PCT_PIB",
]
VARS_NUM = [v for v in VARS_NUM if v in df.columns]

stats = df[VARS_NUM].describe().T
stats["missing_pct"] = round(
    df[VARS_NUM].isna().mean() * 100, 1
)
stats["cv"] = round(
    (df[VARS_NUM].std() / df[VARS_NUM].mean().abs()) * 100, 1
)

stats.to_excel(
    RESULT_DIR / "eda_01_stats_descriptives.xlsx",
    engine="openpyxl"
)
print("\n  Stats descriptives :")
print(stats[["count","mean","std","min","50%","max",
             "missing_pct","cv"]].to_string())
print("\n  💾 eda_01_stats_descriptives.xlsx")


# ============================================================
# 2. CARTE DE COMPLÉTUDE
# ============================================================

print("\n" + "─" * 65)
print("2. CARTE DE COMPLÉTUDE")
print("─" * 65)

completude = pd.DataFrame({
    "Variable":  VARS_NUM,
    "Pct_OK":    [round(df[v].notna().mean()*100,1) for v in VARS_NUM],
})
completude = completude.sort_values("Pct_OK", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = [ACCENT if p < 60 else (
          "#F5A623" if p < 80 else PALETTE)
          for p in completude["Pct_OK"]]
bars = ax.barh(completude["Variable"], completude["Pct_OK"],
               color=colors, height=0.65)
ax.axvline(80, color="gray", ls="--", lw=1, alpha=0.6,
           label="Seuil 80%")
ax.axvline(60, color=ACCENT, ls=":", lw=1, alpha=0.6,
           label="Seuil 60%")
for bar, val in zip(bars, completude["Pct_OK"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val}%", va="center", fontsize=8)
ax.set_xlabel("% de valeurs non-manquantes")
ax.set_title("Complétude des variables — DRC_FiscalPanel_1996_2023")
ax.set_xlim(0, 110)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RESULT_DIR / "eda_02_completude.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  💾 eda_02_completude.png")


# ============================================================
# 3. ÉVOLUTION DU BUDGET TOTAL PAR ANNÉE
# ============================================================

print("\n" + "─" * 65)
print("3. ÉVOLUTION BUDGET TOTAL")
print("─" * 65)

budget_annuel = df.groupby(col_ann).agg(
    Budget_Courant   = ("Budget_Depense_Courante",  "sum"),
    Exec_Courant     = ("Execution_Depense_Courante","sum"),
    Budget_Capital   = ("Budget_Depense_Capital",   "sum"),
    Exec_Capital     = ("Execution_Depense_Capital", "sum"),
).reset_index()
budget_annuel["Taux_Exec_Courant"] = (
    budget_annuel["Exec_Courant"] /
    budget_annuel["Budget_Courant"] * 100
).round(1)
budget_annuel["Taux_Exec_Capital"] = (
    budget_annuel["Exec_Capital"] /
    budget_annuel["Budget_Capital"] * 100
).round(1)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Sous-graphe 1 : Montants
ax = axes[0]
ax.fill_between(budget_annuel[col_ann],
                budget_annuel["Budget_Courant"] / 1e6,
                alpha=0.3, color=PALETTE, label="Budget Courant")
ax.plot(budget_annuel[col_ann],
        budget_annuel["Budget_Courant"] / 1e6,
        color=PALETTE, lw=2)
ax.fill_between(budget_annuel[col_ann],
                budget_annuel["Exec_Courant"] / 1e6,
                alpha=0.3, color=SOFT, label="Exécution Courante")
ax.plot(budget_annuel[col_ann],
        budget_annuel["Exec_Courant"] / 1e6,
        color=SOFT, lw=2)

# Années électorales
for yr in [2006, 2011, 2018, 2023]:
    ax.axvline(yr, color=ACCENT, ls="--", lw=1, alpha=0.5)

ax.set_ylabel("Millions USD")
ax.set_title("Évolution des dépenses courantes (total RDC, toutes fonctions)")
ax.legend()
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:,.0f}M")
)

# Sous-graphe 2 : Taux d'exécution global
ax2 = axes[1]
ax2.plot(budget_annuel[col_ann],
         budget_annuel["Taux_Exec_Courant"],
         color=PALETTE, lw=2.5, marker="o", ms=4,
         label="Taux Courant")
ax2.plot(budget_annuel[col_ann],
         budget_annuel["Taux_Exec_Capital"],
         color=ACCENT, lw=2.5, marker="s", ms=4,
         label="Taux Capital")
ax2.axhline(100, color="gray", ls="--", lw=1, alpha=0.6)
ax2.axhline(80,  color="#F5A623", ls=":", lw=1, alpha=0.5)
for yr in [2006, 2011, 2018, 2023]:
    ax2.axvline(yr, color=ACCENT, ls="--", lw=1, alpha=0.5)
ax2.set_ylabel("Taux d'exécution (%)")
ax2.set_xlabel("Année")
ax2.set_title("Taux d'exécution budgétaire global (courant vs capital)")
ax2.legend()

fig.suptitle("Finances publiques RDC — 1996–2023",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(RESULT_DIR / "eda_03_evolution_budget_total.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  💾 eda_03_evolution_budget_total.png")


# ============================================================
# 4. TAUX D'EXÉCUTION PAR FONCTION (moyenne 1996-2023)
# ============================================================

print("\n" + "─" * 65)
print("4. TAUX D'EXÉCUTION PAR FONCTION")
print("─" * 65)

taux_fct = df.groupby(col_fct).agg(
    Taux_Courant = ("TAUX_EXEC_COURANT", "mean"),
    Taux_Capital = ("TAUX_EXEC_CAPITAL", "mean"),
    N_obs        = (col_ann,            "count"),
).reset_index()
taux_fct = taux_fct.sort_values("Taux_Courant", ascending=True)

fig, ax = plt.subplots(figsize=(11, 9))
y = np.arange(len(taux_fct))
w = 0.35
ax.barh(y + w/2, taux_fct["Taux_Courant"] * 100,
        w, color=PALETTE, alpha=0.85, label="Courant")
ax.barh(y - w/2, taux_fct["Taux_Capital"] * 100,
        w, color=SOFT, alpha=0.85, label="Capital")
ax.axvline(100, color="gray", ls="--", lw=1, alpha=0.6)
ax.axvline(80,  color=ACCENT, ls=":",  lw=1, alpha=0.5)
ax.set_yticks(y)
ax.set_yticklabels(taux_fct[col_fct], fontsize=8)
ax.set_xlabel("Taux d'exécution moyen (%)")
ax.set_title("Taux d'exécution budgétaire moyen par fonction\n(1996–2023)")
ax.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "eda_04_taux_exec_par_fonction.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  💾 eda_04_taux_exec_par_fonction.png")


# ============================================================
# 5. HEATMAP CORRÉLATION
# ============================================================

print("\n" + "─" * 65)
print("5. MATRICE DE CORRÉLATION")
print("─" * 65)

VARS_CORR = [
    "TAUX_EXEC_COURANT", "TAUX_EXEC_CAPITAL",
    "PIB_PAR_HABITANT_USD", "CROISSANCE_PIB_PCT",
    "INFLATION_CPI_PCT", "TAUX_CHANGE_CDF_USD",
    "RECETTES_ETAT_PCT_PIB", "RENTE_RESSOURCES_PCT_PIB",
    "WGI_EFFICACITE_GOUV", "WGI_CONTROLE_CORRUPTION",
    "SOLDE_BUDGETAIRE_PCT_PIB", "DETTE_PUBLIQUE_PCT_PIB",
    "ANNEE_ELECTORALE",
]
VARS_CORR = [v for v in VARS_CORR if v in df.columns]

corr = df[VARS_CORR].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, mask=~mask, ax=ax,
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    linewidths=0.5, square=True,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Matrice de corrélation — Variables ML\n(DRC 1996–2023)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULT_DIR / "eda_05_heatmap_correlation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  💾 eda_05_heatmap_correlation.png")

# Top corrélations avec TAUX_EXEC_COURANT
if "TAUX_EXEC_COURANT" in corr.columns:
    top_corr = (corr["TAUX_EXEC_COURANT"]
                .drop("TAUX_EXEC_COURANT")
                .abs()
                .sort_values(ascending=False)
                .head(5))
    print("\n  Top 5 corrélations avec TAUX_EXEC_COURANT :")
    print(corr["TAUX_EXEC_COURANT"]
          .drop("TAUX_EXEC_COURANT")
          .reindex(top_corr.index)
          .to_string())


# ============================================================
# 6. DISTRIBUTION DU TAUX D'EXÉCUTION
# ============================================================

print("\n" + "─" * 65)
print("6. DISTRIBUTION TAUX D'EXÉCUTION")
print("─" * 65)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, var, titre, color in [
    (axes[0], "TAUX_EXEC_COURANT",
     "Taux d'exécution COURANT", PALETTE),
    (axes[1], "TAUX_EXEC_CAPITAL",
     "Taux d'exécution CAPITAL", SOFT),
]:
    if var not in df.columns:
        continue
    vals = df[var].dropna() * 100
    ax.hist(vals, bins=30, color=color, alpha=0.7,
            edgecolor="white")
    ax.axvline(vals.mean(),  color=ACCENT, ls="--",
               lw=2, label=f"Moy. {vals.mean():.0f}%")
    ax.axvline(vals.median(), color="gray", ls=":",
               lw=2, label=f"Med. {vals.median():.0f}%")
    ax.axvline(100, color="black", ls="-", lw=1, alpha=0.4)
    ax.set_xlabel("Taux d'exécution (%)")
    ax.set_ylabel("Fréquence")
    ax.set_title(titre)
    ax.legend(fontsize=9)

    # Classes
    sous  = (vals < 80).mean() * 100
    norm  = ((vals >= 80) & (vals <= 110)).mean() * 100
    sur   = (vals > 110).mean() * 100
    print(f"\n  {var} :")
    print(f"   Sous-exécution (<80%)   : {sous:.1f}%")
    print(f"   Normal (80–110%)        : {norm:.1f}%")
    print(f"   Sur-exécution (>110%)   : {sur:.1f}%")
    print(f"   Moyenne                 : {vals.mean():.1f}%")
    print(f"   Médiane                 : {vals.median():.1f}%")

plt.suptitle("Distribution du taux d'exécution budgétaire — RDC 1996–2023",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULT_DIR / "eda_06_distribution_taux_exec.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  💾 eda_06_distribution_taux_exec.png")


# ============================================================
# 7. ÉVOLUTION VARIABLES MACRO
# ============================================================

print("\n" + "─" * 65)
print("7. ÉVOLUTION VARIABLES MACRO")
print("─" * 65)

macro_ann = df.groupby(col_ann).first().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
MACRO_PLOTS = [
    ("CROISSANCE_PIB_PCT",    "Croissance PIB (%)",       PALETTE),
    ("INFLATION_CPI_PCT",     "Inflation CPI (%)",         ACCENT),
    ("RECETTES_ETAT_PCT_PIB", "Recettes État (% PIB)",    SOFT),
    ("WGI_EFFICACITE_GOUV",   "WGI Efficacité Gouv.",     "#2ECC71"),
]

for ax, (var, titre, color) in zip(axes.flat, MACRO_PLOTS):
    if var not in macro_ann.columns:
        continue
    data = macro_ann[[col_ann, var]].dropna()
    ax.plot(data[col_ann], data[var], color=color,
            lw=2.5, marker="o", ms=3)
    ax.fill_between(data[col_ann], data[var],
                    alpha=0.1, color=color)
    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    for yr in [2006, 2011, 2018, 2023]:
        ax.axvline(yr, color=ACCENT, ls="--",
                   lw=0.8, alpha=0.4)
    ax.set_title(titre, fontweight="bold")
    ax.set_xlabel("Année")

fig.suptitle("Variables macroéconomiques — RDC 1996–2023\n"
             "(lignes rouges = années électorales)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULT_DIR / "eda_07_evolution_macro.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  💾 eda_07_evolution_macro.png")


# ============================================================
# 8. TOP 10 FONCTIONS PAR BUDGET MOYEN
# ============================================================

print("\n" + "─" * 65)
print("8. TOP 10 FONCTIONS PAR BUDGET")
print("─" * 65)

top10 = (
    df.groupby(col_fct)["Budget_Depense_Courante"]
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

fig, ax = plt.subplots(figsize=(10, 6))
colors_top = [PALETTE if i < 3 else SOFT
              for i in range(len(top10))]
bars = ax.barh(range(len(top10)), top10.values / 1e6,
               color=colors_top, height=0.65)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10.index, fontsize=9)
ax.set_xlabel("Budget courant moyen (Millions USD)")
ax.set_title("Top 10 fonctions par budget courant moyen\n(1996–2023)")
for bar, val in zip(bars, top10.values / 1e6):
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{val:,.1f}M", va="center", fontsize=8)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(RESULT_DIR / "eda_09_top_fonctions.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  💾 eda_09_top_fonctions.png")
print("\n  Top 10 :")
for fct, val in top10.items():
    print(f"   {fct:<40} {val/1e6:>8.1f} M USD")


# ============================================================
# 9. BOXPLOT TAUX D'EXÉCUTION PAR FONCTION
# ============================================================

print("\n" + "─" * 65)
print("9. VARIABILITÉ PAR FONCTION")
print("─" * 65)

if "TAUX_EXEC_COURANT" in df.columns:
    ordre = (df.groupby(col_fct)["TAUX_EXEC_COURANT"]
               .median()
               .sort_values()
               .index)

    fig, ax = plt.subplots(figsize=(11, 9))
    data_box = [
        df.loc[df[col_fct] == f, "TAUX_EXEC_COURANT"].dropna() * 100
        for f in ordre
    ]
    bp = ax.boxplot(data_box, vert=False, patch_artist=True,
                    medianprops=dict(color=ACCENT, lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE)
        patch.set_alpha(0.5)
    ax.set_yticks(range(1, len(ordre)+1))
    ax.set_yticklabels(ordre, fontsize=8)
    ax.axvline(100, color="gray", ls="--", lw=1, alpha=0.6)
    ax.axvline(80,  color=ACCENT, ls=":",  lw=1, alpha=0.5)
    ax.set_xlabel("Taux d'exécution courant (%)")
    ax.set_title("Variabilité du taux d'exécution par fonction\n"
                 "(ligne rouge = médiane | seuil 80% et 100%)")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "eda_10_boxplot_taux_par_fonction.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  💾 eda_10_boxplot_taux_par_fonction.png")


# ============================================================
# 10. EXPORT JSON POUR DASHBOARD REACT
# ============================================================

print("\n" + "─" * 65)
print("10. EXPORT JSON POUR DASHBOARD")
print("─" * 65)

# Statistiques clés pour le dashboard
taux_c = df["TAUX_EXEC_COURANT"].dropna() * 100
taux_k = df["TAUX_EXEC_CAPITAL"].dropna() * 100

summary = {
    "dataset": {
        "n_obs":       len(df),
        "n_fonctions": int(df[col_fct].nunique()),
        "n_annees":    int(df[col_ann].nunique()),
        "annee_min":   int(df[col_ann].min()),
        "annee_max":   int(df[col_ann].max()),
        "n_colonnes":  len(df.columns),
    },
    "taux_exec_courant": {
        "moyenne":      round(float(taux_c.mean()), 1),
        "mediane":      round(float(taux_c.median()), 1),
        "std":          round(float(taux_c.std()), 1),
        "sous_exec_pct":round(float((taux_c < 80).mean()*100), 1),
        "normal_pct":   round(float(((taux_c>=80)&(taux_c<=110)).mean()*100), 1),
        "sur_exec_pct": round(float((taux_c > 110).mean()*100), 1),
    },
    "taux_exec_capital": {
        "moyenne":      round(float(taux_k.mean()), 1),
        "mediane":      round(float(taux_k.median()), 1),
        "std":          round(float(taux_k.std()), 1),
        "sous_exec_pct":round(float((taux_k < 80).mean()*100), 1),
        "normal_pct":   round(float(((taux_k>=80)&(taux_k<=110)).mean()*100), 1),
        "sur_exec_pct": round(float((taux_k > 110).mean()*100), 1),
    },
    "evolution_annuelle": budget_annuel[[
        col_ann,
        "Budget_Courant", "Exec_Courant",
        "Budget_Capital", "Exec_Capital",
        "Taux_Exec_Courant", "Taux_Exec_Capital"
    ]].rename(columns={col_ann: "annee"})
      .fillna(0)
      .to_dict(orient="records"),
    "top10_fonctions": [
        {"fonction": k, "budget_moyen_m": round(v/1e6, 2)}
        for k, v in top10.items()
    ],
    "taux_par_fonction": taux_fct[[col_fct,
        "Taux_Courant","Taux_Capital"
    ]].rename(columns={
        col_fct: "fonction",
        "Taux_Courant": "taux_courant",
        "Taux_Capital": "taux_capital"
    }).fillna(0).to_dict(orient="records"),
    "completude": completude.to_dict(orient="records"),
    "correlations_taux_courant": (
        corr["TAUX_EXEC_COURANT"]
        .drop("TAUX_EXEC_COURANT")
        .sort_values(key=abs, ascending=False)
        .head(10)
        .round(3)
        .to_dict()
        if "TAUX_EXEC_COURANT" in corr.columns else {}
    ),
    "macro_annuelle": macro_ann[[
        col_ann,
        *[v for v in [
            "CROISSANCE_PIB_PCT","INFLATION_CPI_PCT",
            "RECETTES_ETAT_PCT_PIB","WGI_EFFICACITE_GOUV",
            "PIB_PAR_HABITANT_USD","TAUX_CHANGE_CDF_USD",
        ] if v in macro_ann.columns]
    ]].rename(columns={col_ann:"annee"})
      .fillna("null")
      .to_dict(orient="records"),
}

with open(RESULT_DIR / "eda_summary.json", "w",
          encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("  💾 eda_summary.json")


# ============================================================
# RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 65)
print("RÉSUMÉ EDA")
print("=" * 65)

print(f"""
  Dataset        : {len(df)} obs × {len(df.columns)} variables
  Fonctions      : {df[col_fct].nunique()}
  Années         : {int(df[col_ann].min())} → {int(df[col_ann].max())}

  Taux exécution COURANT :
    Moyenne      : {taux_c.mean():.1f}%
    Médiane      : {taux_c.median():.1f}%
    Sous-exec    : {(taux_c<80).mean()*100:.1f}% des observations
    Normal       : {((taux_c>=80)&(taux_c<=110)).mean()*100:.1f}% des observations
    Sur-exec     : {(taux_c>110).mean()*100:.1f}% des observations

  Taux exécution CAPITAL :
    Moyenne      : {taux_k.mean():.1f}%
    Médiane      : {taux_k.median():.1f}%
    Sous-exec    : {(taux_k<80).mean()*100:.1f}% des observations

  Fichiers générés dans ResultatML/ :
""")

for f in sorted(RESULT_DIR.glob("eda_*")):
    kb = round(f.stat().st_size / 1024, 1)
    print(f"   ✅  {f.name:45} ({kb} KB)")

print("\n  Prochaine étape : python ml_01_regression.py")
print("=" * 65)
