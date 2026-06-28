"""
VMI Intelligence API — Version déployable (Railway / Render)
Projet Motul — Ticket 38998
 
Accepte deux formats d'entrée :
  1. CSV synthétique (colonnes : MATNR;LIFNR;BUDAT;MENGE;WERTN)
  2. Export Excel SAP réel (SE16 → RKWA) — converti automatiquement
 
Endpoints :
  GET  /              → healthcheck
  POST /api/analyze   → upload CSV ou XLSX → prévisions + recommandations ZMRKO
  GET  /api/demo      → analyse sur données synthétiques intégrées
  GET  /api/zmrko/{matnr} → champs PA_ALLW + PA_EXPAM pour intégration ABAP
"""
 
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime, timedelta
 
app = FastAPI(
    title="VMI Intelligence API — Motul",
    description="Analyse VMI. Upload export SAP (XLSX ou CSV) → prévisions J+30 + recommandations ZMRKO.",
    version="3.0.0",
    docs_url="/docs",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Mapping colonnes SAP réel → colonnes internes ────────────────────────────
MAPPING_SAP = {
    "Article":                              "MATNR",
    "Fournisseur":                          "LIFNR",
    "Date comptable":                       "BUDAT",
    "Quantité prélevée pour liquidation":   "MENGE",
    "Montant":                              "WERTN",
    # Variantes possibles selon version SAP
    "MATNR": "MATNR",
    "LIFNR": "LIFNR",
    "BUDAT": "BUDAT",
    "MENGE": "MENGE",
    "WERTN": "WERTN",
}
 
DESCRIPTIONS_SAP = {}  # enrichi dynamiquement depuis les données réelles
 
def generer_donnees_demo() -> pd.DataFrame:
    np.random.seed(42)
    articles = [f"MAT-{i:03d}" for i in range(1, 11)]
    fournisseurs = {f"MAT-{i:03d}": f"FOURN-{((i-1)//2)+1:03d}" for i in range(1, 11)}
    prix = {f"MAT-{i:03d}": round(np.random.uniform(8, 65), 2) for i in range(1, 11)}
    rows = []
    date_debut = datetime(2022, 1, 3)
    for matnr in articles:
        base = np.random.uniform(20, 80)
        for semaine in range(130):
            date = date_debut + timedelta(weeks=semaine)
            saison = 1 + 0.3 * np.sin(2 * np.pi * semaine / 52)
            menge = max(0, base * saison + np.random.normal(0, base * 0.15))
            if np.random.rand() < 0.04:
                menge *= np.random.choice([2.5, 0.1])
            rows.append({
                "MATNR": matnr, "LIFNR": fournisseurs[matnr],
                "BUDAT": date, "MENGE": round(menge, 1),
                "WERTN": round(menge * prix[matnr], 2),
            })
    return pd.DataFrame(rows)
 
# ── Convertisseur universel SAP → format interne ─────────────────────────────
 
def convertir_export_sap(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accepte un DataFrame brut (export SE16 SAP ou CSV synthétique)
    et retourne un DataFrame normalisé avec colonnes MATNR/LIFNR/BUDAT/MENGE/WERTN.
    """
    # Renommer les colonnes SAP françaises vers noms internes
    df = df_raw.rename(columns=MAPPING_SAP)
 
    # Vérifier colonnes minimales requises
    colonnes_requises = {"MATNR", "BUDAT", "MENGE"}
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        raise ValueError(
            f"Colonnes introuvables : {manquantes}. "
            f"Colonnes présentes : {list(df_raw.columns)}"
        )
 
    # LIFNR optionnel
    if "LIFNR" not in df.columns:
        df["LIFNR"] = "INCONNU"
 
    # WERTN optionnel
    if "WERTN" not in df.columns:
        df["WERTN"] = 0.0
 
    # Conversion types
    df["BUDAT"] = pd.to_datetime(df["BUDAT"], infer_datetime_format=True, errors="coerce")
    df["MENGE"] = pd.to_numeric(df["MENGE"], errors="coerce").fillna(0).abs()
    df["WERTN"] = pd.to_numeric(df["WERTN"], errors="coerce").fillna(0).abs()
    df["MATNR"] = df["MATNR"].astype(str).str.strip()
    df["LIFNR"] = df["LIFNR"].astype(str).str.strip()
 
    df = df.dropna(subset=["BUDAT"])
    df = df[df["MENGE"] > 0]
 
    return df[["MATNR", "LIFNR", "BUDAT", "MENGE", "WERTN"]]
 
# ── Moteur d'analyse ─────────────────────────────────────────────────────────
 
def analyser_csv(df: pd.DataFrame, source: str = "") -> dict:
    articles = sorted(df["MATNR"].unique())
    resultats = {}
 
    for matnr in articles:
        sub = df[df["MATNR"] == matnr].copy()
 
        # Série hebdomadaire
        serie = (
            sub.set_index("BUDAT")["MENGE"]
            .resample("W-MON").sum()
            .reset_index()
        )
        serie.columns = ["ds", "y"]
 
        nb_semaines = len(serie)
        donnees_insuffisantes = nb_semaines < 4
 
        y = serie["y"].values
        n = len(y)
 
        if donnees_insuffisantes:
            # Prévision simple sur données courtes : moyenne des données disponibles
            moy = float(np.mean(y)) if n > 0 else 0
            derniere_date = serie["ds"].iloc[-1] if n > 0 else datetime.now()
            prevision_semaines = []
            for i in range(1, 5):
                prevision_semaines.append({
                    "ds": (derniere_date + timedelta(weeks=i)).strftime("%Y-%m-%d"),
                    "yhat": round(moy, 1),
                    "yhat_lower": round(moy * 0.7, 1),
                    "yhat_upper": round(moy * 1.3, 1),
                })
            total_j30   = round(moy * 4, 1)
            borne_basse = round(moy * 4 * 0.7, 1)
            borne_haute = round(moy * 4 * 1.3, 1)
            avertissement = f"Données limitées ({nb_semaines} semaine(s)) — prévision basée sur la moyenne"
        else:
            # Prévision moyenne mobile pondérée + tendance linéaire
            fenetre = min(8, n)
            poids = np.arange(1, fenetre + 1, dtype=float)
            poids /= poids.sum()
            base_prev = float(np.dot(y[-fenetre:], poids))
            n_trend = min(12, n)
            coeffs = np.polyfit(np.arange(n_trend), y[-n_trend:], 1)
            tendance_hebdo = float(coeffs[0])
            std = float(np.std(y[-fenetre:]))
            derniere_date = serie["ds"].iloc[-1]
            prevision_semaines = []
            for i in range(1, 5):
                yhat = max(0, base_prev + tendance_hebdo * i)
                prevision_semaines.append({
                    "ds": (derniere_date + timedelta(weeks=i)).strftime("%Y-%m-%d"),
                    "yhat": round(yhat, 1),
                    "yhat_lower": round(max(0, yhat - std), 1),
                    "yhat_upper": round(yhat + std, 1),
                })
            total_j30   = round(sum(s["yhat"] for s in prevision_semaines), 1)
            borne_basse = round(sum(s["yhat_lower"] for s in prevision_semaines), 1)
            borne_haute = round(sum(s["yhat_upper"] for s in prevision_semaines), 1)
            avertissement = None
 
        # Anomalies
        anomalies = []
        if not donnees_insuffisantes:
            rolling_mean = pd.Series(y).rolling(8, min_periods=3).mean()
            rolling_std  = pd.Series(y).rolling(8, min_periods=3).std().fillna(1)
            z_scores = (y - rolling_mean) / rolling_std
            for i, row in serie.iterrows():
                z = float(z_scores.iloc[i]) if not np.isnan(z_scores.iloc[i]) else 0
                if abs(z) > 1.8:
                    anomalies.append({
                        "ds": row["ds"].strftime("%Y-%m-%d"),
                        "y": round(float(row["y"]), 1),
                        "z_score": round(z, 2),
                        "type_anomalie": "pic" if z > 0 else "creux",
                    })
 
        # Stats
        moy_hebdo = round(float(np.mean(y)), 1) if len(y) > 0 else 0
        std_hebdo = round(float(np.std(y)), 1)  if len(y) > 0 else 0
        cv = std_hebdo / moy_hebdo if moy_hebdo > 0 else 0.1
 
        # PA_ALLW
        base_allw   = round(min(25.0, max(2.0, cv * 100 * 0.8)), 1)
        bonus_anom  = max(0, (len(anomalies) - 5) // 3) * 2.0
        pa_allw_val = round(min(25.0, max(2.0, base_allw + bonus_anom)), 1)
        interp_allw = (
            "Tolérance serrée — consommation stable"   if pa_allw_val <= 5  else
            "Tolérance modérée — légère variabilité"   if pa_allw_val <= 12 else
            "Tolérance large — consommation instable"
        )
 
        # PA_EXPAM
        total_menge = sub["MENGE"].sum()
        total_wertn = sub["WERTN"].sum()
        prix_moyen  = (total_wertn / total_menge) if total_menge > 0 else 0
        pa_expam_val = round(total_j30 * prix_moyen, 2)
 
        # Alerte stock
        stock_estime = float(serie["y"].iloc[-4:].sum()) * 0.6 if len(serie) >= 4 else moy_hebdo * 2
        ratio_stock  = stock_estime / total_j30 if total_j30 > 0 else 999
        jours_stock  = round(stock_estime / (total_j30 / 30), 1) if total_j30 > 0 else 999
        niveau_stock = "CRITIQUE" if ratio_stock < 0.2 else "ATTENTION" if ratio_stock < 0.5 else "OK"
 
        # Score risque
        s_stock = {"OK": 0, "ATTENTION": 20, "CRITIQUE": 40}[niveau_stock]
        s_anom  = min(20, len(anomalies) * 2)
        s_tol   = min(10, pa_allw_val / 2.5)
        score_risque  = round(s_stock + s_anom + s_tol, 1)
        niveau_risque = "FAIBLE" if score_risque < 25 else "MODÉRÉ" if score_risque < 55 else "ÉLEVÉ"
 
        # Historique pour graphique
        historique = [
            {
                "ds": row["ds"].strftime("%Y-%m-%d"),
                "y": round(float(row["y"]), 1),
                "anomalie": any(a["ds"] == row["ds"].strftime("%Y-%m-%d") for a in anomalies),
            }
            for _, row in serie.iterrows()
        ]
 
        lifnr = str(sub["LIFNR"].iloc[0]) if len(sub) > 0 else "INCONNU"
        desc  = DESCRIPTIONS_SAP.get(matnr, matnr)
 
        resultats[matnr] = {
            "matnr":       matnr,
            "description": desc,
            "fournisseur": lifnr,
            "horodatage":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "avertissement": avertissement,
            "stats": {
                "moy_hebdo_units": moy_hebdo,
                "std_hebdo_units": std_hebdo,
                "nb_semaines":     nb_semaines,
                "nb_anomalies":    len(anomalies),
                "periode_debut":   serie["ds"].iloc[0].strftime("%Y-%m-%d") if len(serie) > 0 else "",
                "periode_fin":     serie["ds"].iloc[-1].strftime("%Y-%m-%d") if len(serie) > 0 else "",
            },
            "prevision_j30": {
                "total_units": total_j30,
                "borne_basse": borne_basse,
                "borne_haute": borne_haute,
                "par_semaine": prevision_semaines,
            },
            "zmrko": {
                "PA_ALLW": {
                    "valeur":         pa_allw_val,
                    "unite":          "%",
                    "interpretation": interp_allw,
                },
                "PA_EXPAM": {
                    "valeur":      pa_expam_val,
                    "borne_basse": round(borne_basse * prix_moyen, 2),
                    "borne_haute": round(borne_haute * prix_moyen, 2),
                    "unite":       "EUR",
                    "prix_unitaire_moyen": round(prix_moyen, 4),
                },
            },
            "alertes": {
                "stock": {
                    "niveau":           niveau_stock,
                    "couleur":          {"OK": "vert", "ATTENTION": "orange", "CRITIQUE": "rouge"}[niveau_stock],
                    "jours_couverture": jours_stock,
                    "message": (
                        f"Stock insuffisant — rupture estimée dans {jours_stock} jours" if niveau_stock == "CRITIQUE" else
                        f"Stock bas — surveiller ({jours_stock} jours)"                 if niveau_stock == "ATTENTION" else
                        f"Stock suffisant ({jours_stock} jours de couverture)"
                    ),
                },
            },
            "risque_global": {
                "score":   score_risque,
                "niveau":  niveau_risque,
                "couleur": "rouge" if score_risque >= 55 else "orange" if score_risque >= 25 else "vert",
            },
            "historique": historique[-52:],
            "anomalies":  anomalies,
        }
 
    return resultats
 
# ── Routes ───────────────────────────────────────────────────────────────────
 
@app.get("/")
def healthcheck():
    return {"status": "ok", "service": "VMI Intelligence API — Motul", "version": "3.0.0",
            "timestamp": datetime.now().isoformat()}
 
@app.post("/api/analyze")
async def analyser(file: UploadFile = File(...)):
    """
    Upload export SAP (XLSX ou CSV).
    Détecte automatiquement le format et convertit les colonnes SAP françaises.
    """
    contenu = await file.read()
    nom = file.filename.lower()
 
    try:
        if nom.endswith(".xlsx") or nom.endswith(".xls"):
            df_raw = pd.read_excel(io.BytesIO(contenu))
        elif nom.endswith(".csv"):
            # Essai séparateur ; puis ,
            try:
                df_raw = pd.read_csv(io.BytesIO(contenu), sep=";", dtype=str)
                if len(df_raw.columns) < 3:
                    df_raw = pd.read_csv(io.BytesIO(contenu), sep=",", dtype=str)
            except Exception:
                df_raw = pd.read_csv(io.BytesIO(contenu), dtype=str)
        else:
            raise HTTPException(status_code=400, detail="Format accepté : .xlsx ou .csv")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture fichier : {e}")
 
    try:
        df = convertir_export_sap(df_raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
 
    resultats = analyser_csv(df, source=file.filename)
 
    nb_alertes = sum(1 for r in resultats.values() if r["alertes"]["stock"]["niveau"] != "OK")
    total_j30  = sum(r["prevision_j30"]["total_units"] for r in resultats.values())
    risque_max = max((r["risque_global"]["score"] for r in resultats.values()), default=0)
 
    # Avertissements données courtes
    avertissements = [
        f"{m} : {r['avertissement']}"
        for m, r in resultats.items() if r.get("avertissement")
    ]
 
    return {
        "meta": {
            "fichier":         file.filename,
            "nb_articles":     len(resultats),
            "nb_alertes":      nb_alertes,
            "total_j30_units": round(total_j30, 1),
            "risque_max":      risque_max,
            "horodatage":      datetime.now().isoformat(),
            "avertissements":  avertissements,
        },
        "articles": resultats,
    }
 
@app.get("/api/demo")
def demo():
    df = generer_donnees_demo()
    resultats = analyser_csv(df, source="DEMO")
    nb_alertes = sum(1 for r in resultats.values() if r["alertes"]["stock"]["niveau"] != "OK")
    total_j30  = sum(r["prevision_j30"]["total_units"] for r in resultats.values())
    risque_max = max((r["risque_global"]["score"] for r in resultats.values()), default=0)
    return {
        "meta": {
            "fichier": "DEMO — données synthétiques",
            "nb_articles": len(resultats), "nb_alertes": nb_alertes,
            "total_j30_units": round(total_j30, 1), "risque_max": risque_max,
            "horodatage": datetime.now().isoformat(), "avertissements": [],
        },
        "articles": resultats,
    }
 
@app.get("/api/zmrko/{matnr}")
def zmrko_article(matnr: str):
    df = generer_donnees_demo()
    resultats = analyser_csv(df)
    matnr = matnr.upper()
    if matnr not in resultats:
        raise HTTPException(status_code=404, detail=f"Article {matnr} introuvable")
    r = resultats[matnr]
    return {
        "matnr": matnr, "description": r["description"],
        "PA_ALLW": r["zmrko"]["PA_ALLW"],
        "PA_EXPAM": r["zmrko"]["PA_EXPAM"],
        "alertes": r["alertes"],
        "risque_global": r["risque_global"],
    }
 
