# ============================================================
# ML 03 — CAUSAL ML : DOUBLE MACHINE LEARNING (DML)
# DRC_FiscalPanel_1996_2023
# ============================================================
print("=" * 65)
print("ML 03 — CAUSAL ML : DOUBLE MACHINE LEARNING (DML)")
print("DRC_FiscalPanel_1996_2023")
print("=" * 65)
print()
print("  Méthode : Chernozhukov et al. (2018)")
print("  Question : Effet causal des élections sur les dépenses ?")
print()

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import json
from scipy import stats

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

print("  OK Librairies OK")

DATA_DIR   = Path("../data/processed")
RESULT_DIR = DATA_DIR / "ResultatML"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
F_FINAL    = DATA_DIR / "DRC_FiscalPanel_1996_2023.xlsx"

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans",
    "font.size": 10,   "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})
NAVY, RED, BLUE, GREEN, GOLD = "#1A3A5C","#E84B3A","#4C9BE8","#2ECC71","#F5A623"

print("\n" + "-" * 65)
print("1. CHARGEMENT")
print("-" * 65)

df = pd.read_excel(F_FINAL, engine="openpyxl")
df.columns = [str(c).strip() for c in df.columns]
col_fct = next(c for c in df.columns if "fonction" in c.lower())
col_ann = next(c for c in df.columns if "annee"    in c.lower())
df[col_ann] = df[col_ann].astype(int)
df = df.sort_values([col_fct, col_ann]).reset_index(drop=True)

for col in ["TAUX_EXEC_COURANT","TAUX_EXEC_CAPITAL",
            "TAUX_EXEC_COURANT_LAG1","TAUX_EXEC_CAPITAL_LAG1"]:
    if col in df.columns and df[col].dropna().max() <= 3.5:
        df[col] = df[col] * 100

for col in ["TAUX_EXEC_COURANT","TAUX_EXEC_CAPITAL"]:
    if col in df.columns:
        df[col] = df[col].clip(upper=300)

if "Budget_Depense_Courante" in df.columns:
    df["LOG_BUDGET_COURANT"] = np.log1p(df["Budget_Depense_Courante"].clip(lower=0))

fct_map = {f: i for i, f in enumerate(sorted(df[col_fct].unique()))}
df["FONCTION_CODE"] = df[col_fct].map(fct_map)

print(f"  Dataset : {len(df)} obs x {len(df.columns)} colonnes")

CONFOUNDERS = [f for f in [
    "FONCTION_CODE","PIB_PAR_HABITANT_USD","CROISSANCE_PIB_PCT",
    "INFLATION_CPI_PCT","TAUX_CHANGE_CDF_USD","RECETTES_ETAT_PCT_PIB",
    "RENTE_RESSOURCES_PCT_PIB","WGI_EFFICACITE_GOUV",
    "WGI_CONTROLE_CORRUPTION","SOLDE_BUDGETAIRE_PCT_PIB",
    "DETTE_PUBLIQUE_PCT_PIB","TAUX_EXEC_COURANT_LAG1",
    "CROISSANCE_PIB_PCT_LAG1",
] if f in df.columns]

T_VAR = "ANNEE_ELECTORALE"
Y_VARS = {k: v for k, v in {
    "TAUX_EXEC_COURANT":  "Taux d'execution courant (%)",
    "LOG_BUDGET_COURANT": "Log Budget courant",
}.items() if k in df.columns}

print(f"  Traitement T  : {T_VAR}")
print(f"  Confounders X : {len(CONFOUNDERS)}")
print(f"  Resultats Y   : {list(Y_VARS.keys())}")

print("\n" + "-" * 65)
print("2. DOUBLE MACHINE LEARNING")
print("-" * 65)

def double_ml(df, Y_col, T_col, X_cols, n_folds=5, random_state=42):
    cols_needed = [Y_col, T_col] + X_cols
    df_c = df[cols_needed].dropna()
    n    = len(df_c)
    Y = df_c[Y_col].values
    T = df_c[T_col].values
    X = df_c[X_cols].values

    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(X)
    scl = StandardScaler()
    X   = scl.fit_transform(X)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    model_Y = RandomForestRegressor(n_estimators=200, max_depth=5,
                                    min_samples_leaf=5,
                                    random_state=random_state, n_jobs=-1)
    model_T = RandomForestClassifier(n_estimators=200, max_depth=5,
                                     min_samples_leaf=5,
                                     random_state=random_state, n_jobs=-1)

    Y_hat = cross_val_predict(model_Y, X, Y, cv=kf)
    T_hat = cross_val_predict(model_T, X, T, cv=kf, method="predict_proba")[:, 1]

    Y_tilde = Y - Y_hat
    T_tilde = T - T_hat

    r2_Y = r2_score(Y, Y_hat)
    r2_T = r2_score(T, T_hat)

    theta = (T_tilde @ Y_tilde) / (T_tilde @ T_tilde)
    psi   = T_tilde * (Y_tilde - theta * T_tilde)
    var_theta = (np.mean(psi**2) / (np.mean(T_tilde**2))**2) / n
    se    = np.sqrt(var_theta)

    t_stat  = theta / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    ci_low  = theta - 1.96 * se
    ci_high = theta + 1.96 * se

    return {
        "theta":   round(float(theta),   4),
        "se":      round(float(se),      4),
        "t_stat":  round(float(t_stat),  4),
        "p_value": round(float(p_value), 4),
        "ci_low":  round(float(ci_low),  4),
        "ci_high": round(float(ci_high), 4),
        "r2_Y":    round(float(r2_Y),    4),
        "r2_T":    round(float(r2_T),    4),
        "n_obs":   n,
        "Y_tilde": Y_tilde,
        "T_tilde": T_tilde,
        "T_hat":   T_hat,
        "T_orig":  T,
    }

resultats_dml = {}
for Y_col, Y_label in Y_VARS.items():
    print(f"\n  Cible Y = {Y_col}")
    res = double_ml(df, Y_col, T_VAR, CONFOUNDERS)
    resultats_dml[Y_col] = res
    sig = "*** (p<0.01)" if res["p_value"] < 0.01 else \
          "**  (p<0.05)" if res["p_value"] < 0.05 else \
          "*   (p<0.10)" if res["p_value"] < 0.10 else "(non-sig)"
    print(f"     theta   = {res['theta']:+.4f}  {sig}")
    print(f"     SE      = {res['se']:.4f}")
    print(f"     IC 95%  = [{res['ci_low']:+.4f} ; {res['ci_high']:+.4f}]")
    print(f"     p-value = {res['p_value']:.4f}")
    print(f"     R2_Y    = {res['r2_Y']:.4f}  |  R2_T = {res['r2_T']:.4f}")
    print(f"     N obs   = {res['n_obs']}")

print("\n" + "-" * 65)
print("3. HETEROGENEITE PAR GROUPE")
print("-" * 65)

GROUPES = {
    "Regime": ["DEFENSE_SECURITE","PRESIDENCE","PRIMATURE",
               "PARLEMENT","AFFAIRES_ETRANGERES","JUSTICE"],
    "Social": ["SANTE","EDUCATION_NATIONALE","AFFAIRES_SOCIALES",
               "CULTURE_LOISIRS","TRAVAIL_EMPLOI"],
    "Economique": ["ECONOMIE","AGRICULTURE_PECHE","MINES_ENERGIE",
                   "TRANSPORTS","TRAVAUX_PUBLICS"],
    "Administratif": ["SERVICES_GENERAUX","FINANCES_PUBLIQUES",
                      "ADMINISTRATION_TERRITOIRE","PLAN"],
}

Y_het = "TAUX_EXEC_COURANT"
resultats_het = {}
for groupe, fonctions in GROUPES.items():
    df_g = df[df[col_fct].isin(fonctions)].copy()
    if len(df_g.dropna(subset=[Y_het, T_VAR])) < 30:
        continue
    try:
        res_g = double_ml(df_g, Y_het, T_VAR, CONFOUNDERS)
        resultats_het[groupe] = res_g
        sig = "***" if res_g["p_value"]<0.01 else \
              "**"  if res_g["p_value"]<0.05 else \
              "*"   if res_g["p_value"]<0.10 else "ns"
        print(f"  {groupe:<15} theta={res_g['theta']:+.4f}  "
              f"p={res_g['p_value']:.3f} {sig}  N={res_g['n_obs']}")
    except Exception as e:
        print(f"  {groupe:<15} Erreur: {str(e)[:40]}")

print("\n" + "-" * 65)
print("4. GRAPHIQUES")
print("-" * 65)

# Forest plot
fig, ax = plt.subplots(figsize=(10, 5))
y_labels, thetas, ci_lows, ci_highs, colors_fp, p_vals = [], [], [], [], [], []
for Y_col, Y_label in Y_VARS.items():
    res = resultats_dml[Y_col]
    y_labels.append(Y_label)
    thetas.append(res["theta"]); ci_lows.append(res["ci_low"])
    ci_highs.append(res["ci_high"]); p_vals.append(res["p_value"])
    colors_fp.append(RED if res["p_value"] < 0.05 else NAVY)
for groupe, res in resultats_het.items():
    y_labels.append(f"  -> {groupe}")
    thetas.append(res["theta"]); ci_lows.append(res["ci_low"])
    ci_highs.append(res["ci_high"]); p_vals.append(res["p_value"])
    colors_fp.append(RED if res["p_value"]<0.05 else GOLD if res["p_value"]<0.10 else BLUE)

y_pos = np.arange(len(y_labels))
for i,(theta,cil,cih,col) in enumerate(zip(thetas,ci_lows,ci_highs,colors_fp)):
    ax.plot([cil,cih],[i,i],"-",color=col,lw=2.5,alpha=0.7)
    ax.plot(theta,i,"o",color=col,ms=9,zorder=5)
    sig_str = "***" if p_vals[i]<0.01 else "**" if p_vals[i]<0.05 else "*" if p_vals[i]<0.10 else ""
    ax.text(cih+0.2,i,f"theta={theta:+.2f}{sig_str}",va="center",fontsize=8.5,color=col)
ax.axvline(0,color="gray",ls="--",lw=1.5)
ax.set_yticks(y_pos); ax.set_yticklabels(y_labels,fontsize=9)
ax.set_xlabel("Effet causal theta (points de %)  |  IC 95%")
ax.set_title("ML 03 - Forest Plot : Effets causaux des annees electorales\nDouble Machine Learning (Chernozhukov et al., 2018)")
plt.tight_layout()
plt.savefig(RESULT_DIR/"ml_03_forest_plot.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Sauvegarde ml_03_forest_plot.png")

# Residus DML
res_main = resultats_dml["TAUX_EXEC_COURANT"]
T_tilde  = res_main["T_tilde"]; Y_tilde = res_main["Y_tilde"]
T_hat    = res_main["T_hat"];   T_orig  = res_main["T_orig"]
fig, axes = plt.subplots(1,2,figsize=(12,5))
mask0 = T_orig == 0; mask1 = T_orig == 1
axes[0].hist(T_hat[mask0],bins=25,alpha=0.6,color=NAVY,label="Non-electoral")
axes[0].hist(T_hat[mask1],bins=25,alpha=0.6,color=RED,label="Electoral")
axes[0].set_xlabel("P(Annee electorale | X) - Propensity Score")
axes[0].set_ylabel("Frequence")
axes[0].set_title("Distribution du Propensity Score\nSeparation = bon modele nuisance T")
axes[0].legend()
theta = res_main["theta"]
axes[1].scatter(T_tilde,Y_tilde,alpha=0.3,color=NAVY,s=20)
x_line = np.linspace(T_tilde.min(),T_tilde.max(),100)
axes[1].plot(x_line,theta*x_line,"-",color=RED,lw=2.5,label=f"theta={theta:+.3f}")
axes[1].axhline(0,color="gray",ls=":",lw=1); axes[1].axvline(0,color="gray",ls=":",lw=1)
axes[1].set_xlabel("Residu T (part electorale non-expliquee par X)")
axes[1].set_ylabel("Residu Y (part taux exec non-expliquee par X)")
axes[1].set_title("Relation causale nettoyee des confounders\nTAUX_EXEC_COURANT ~ ANNEE_ELECTORALE | X")
axes[1].legend()
fig.suptitle("ML 03 - Verification DML | Residus",fontsize=12,fontweight="bold")
plt.tight_layout()
plt.savefig(RESULT_DIR/"ml_03_residus_dml.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Sauvegarde ml_03_residus_dml.png")

# Evolution temporelle
fig, ax = plt.subplots(figsize=(13,5))
evo = df.groupby(col_ann)["TAUX_EXEC_COURANT"].mean().reset_index()
elections = sorted(df[df[T_VAR]==1][col_ann].unique())
ax.plot(evo[col_ann],evo["TAUX_EXEC_COURANT"],"o-",color=NAVY,lw=2.5,ms=5,label="Taux exec. courant moyen")
ax.axhline(evo["TAUX_EXEC_COURANT"].mean(),color="gray",ls="--",lw=1,alpha=0.6,label="Moyenne globale")
for el in elections:
    ax.axvspan(el-0.4,el+0.4,alpha=0.15,color=RED)
    ax.text(el,ax.get_ylim()[1] if hasattr(ax.get_ylim(),'__len__') else 200,
            f"Elec.\n{el}",ha="center",fontsize=7.5,color=RED)
theta_m = resultats_dml["TAUX_EXEC_COURANT"]["theta"]
p_m     = resultats_dml["TAUX_EXEC_COURANT"]["p_value"]
sig_m   = "***" if p_m<0.01 else "**" if p_m<0.05 else "*" if p_m<0.10 else "(ns)"
ax.text(0.02,0.95,f"Effet causal DML : theta = {theta_m:+.2f} pts% {sig_m}",
        transform=ax.transAxes,fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor=RED if p_m<0.05 else "gray",alpha=0.15),
        color=RED if p_m<0.05 else "gray")
ax.set_xlabel("Annee"); ax.set_ylabel("Taux d'execution moyen (%)")
ax.set_title("Evolution du taux d'execution et annees electorales\nZones rouges = annees electorales")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(RESULT_DIR/"ml_03_evolution_electorale.png",dpi=150,bbox_inches="tight")
plt.close()
print("  Sauvegarde ml_03_evolution_electorale.png")

# Heterogeneite
if resultats_het:
    groupes  = list(resultats_het.keys())
    thetas_g = [resultats_het[g]["theta"]   for g in groupes]
    ci_l_g   = [resultats_het[g]["ci_low"]  for g in groupes]
    ci_h_g   = [resultats_het[g]["ci_high"] for g in groupes]
    pvals_g  = [resultats_het[g]["p_value"] for g in groupes]
    cols_g   = [RED if p<0.05 else GOLD if p<0.10 else BLUE for p in pvals_g]
    fig, ax = plt.subplots(figsize=(9,5))
    x = np.arange(len(groupes))
    bars = ax.bar(x,thetas_g,color=cols_g,alpha=0.85,width=0.5,edgecolor="white")
    for i,(bar,cil,cih,p) in enumerate(zip(bars,ci_l_g,ci_h_g,pvals_g)):
        ax.errorbar(bar.get_x()+bar.get_width()/2,thetas_g[i],
                    yerr=[[thetas_g[i]-cil],[cih-thetas_g[i]]],
                    fmt="none",color="black",capsize=5,lw=2)
        sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else "ns"
        ax.text(i,max(thetas_g[i],0)+0.3,sig,ha="center",fontsize=11,fontweight="bold")
    ax.axhline(0,color="gray",ls="--",lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels(groupes,fontsize=10)
    ax.set_ylabel("Effet causal theta (points de %)")
    ax.set_title("Heterogeneite de l'effet electoral par groupe\nRouge=sig.5% | Orange=sig.10% | Bleu=ns")
    plt.tight_layout()
    plt.savefig(RESULT_DIR/"ml_03_heterogeneite.png",dpi=150,bbox_inches="tight")
    plt.close()
    print("  Sauvegarde ml_03_heterogeneite.png")

print("\n" + "-" * 65)
print("5. EXPORT RESULTATS")
print("-" * 65)

with pd.ExcelWriter(RESULT_DIR/"ml_03_resultats_dml.xlsx",engine="openpyxl") as writer:
    rows_main = []
    for Y_col,Y_label in Y_VARS.items():
        res = resultats_dml[Y_col]
        sig = "***" if res["p_value"]<0.01 else "**" if res["p_value"]<0.05 else "*" if res["p_value"]<0.10 else "ns"
        rows_main.append({"Variable_Y":Y_col,"Label":Y_label,"Traitement_T":T_VAR,
                          "Theta":res["theta"],"SE":res["se"],"t_stat":res["t_stat"],
                          "p_value":res["p_value"],"Sig":sig,
                          "IC95_bas":res["ci_low"],"IC95_haut":res["ci_high"],
                          "R2_Y":res["r2_Y"],"R2_T":res["r2_T"],"N":res["n_obs"]})
    pd.DataFrame(rows_main).to_excel(writer,sheet_name="Effets_Causaux",index=False)
    if resultats_het:
        rows_het = []
        for g,res in resultats_het.items():
            sig = "***" if res["p_value"]<0.01 else "**" if res["p_value"]<0.05 else "*" if res["p_value"]<0.10 else "ns"
            rows_het.append({"Groupe":g,"Theta":res["theta"],"SE":res["se"],
                             "p_value":res["p_value"],"Sig":sig,
                             "IC95_bas":res["ci_low"],"IC95_haut":res["ci_high"],"N":res["n_obs"]})
        pd.DataFrame(rows_het).to_excel(writer,sheet_name="Heterogeneite",index=False)
print("  Sauvegarde ml_03_resultats_dml.xlsx")

summary_dml = {}
for Y_col,res in resultats_dml.items():
    summary_dml[Y_col] = {k:v for k,v in res.items() if k not in ["Y_tilde","T_tilde","T_hat","T_orig"]}
summary_dml["heterogeneite"] = {g:{k:v for k,v in r.items() if k not in ["Y_tilde","T_tilde","T_hat","T_orig"]} for g,r in resultats_het.items()}
with open(RESULT_DIR/"ml_03_summary.json","w") as f:
    json.dump(summary_dml,f,indent=2)
print("  Sauvegarde ml_03_summary.json")

print("\n" + "=" * 65)
print("RESUME ML 03")
print("=" * 65)
for Y_col,res in resultats_dml.items():
    sig = "SIGNIFICATIF ***" if res["p_value"]<0.01 else "SIGNIFICATIF **" if res["p_value"]<0.05 else "MARGINALEMENT *" if res["p_value"]<0.10 else "NON-SIGNIFICATIF"
    print(f"\n  Y = {Y_col}")
    print(f"    theta   = {res['theta']:+.4f} | {sig}")
    print(f"    IC 95%  = [{res['ci_low']:+.4f} ; {res['ci_high']:+.4f}]")
    print(f"    p-value = {res['p_value']:.4f} | N = {res['n_obs']}")
if resultats_het:
    print("\n  Heterogeneite :")
    for g,res in resultats_het.items():
        sig = "***" if res["p_value"]<0.01 else "**" if res["p_value"]<0.05 else "*" if res["p_value"]<0.10 else "ns"
        print(f"    {g:<15} theta={res['theta']:+.4f}  p={res['p_value']:.3f} {sig}")
print("\n" + "-" * 65)
for f in sorted(RESULT_DIR.glob("ml_03_*")):
    kb = round(f.stat().st_size/1024,1)
    print(f"   OK  {f.name} ({kb} KB)")
print("\n  Prochaine etape : python ml_04_anomalies.py")
print("=" * 65)
