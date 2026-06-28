"""
VMI Intelligence API — Version déployable (Render.com)
Projet Motul — Ticket 38998

Différences vs version locale :
  - Pas de Prophet (trop lourd pour serveur gratuit)
  - Prévision via moyenne mobile + régression linéaire (numpy only)
  - Tout fonctionne en mémoire — aucun fichier local requis
  - Endpoint POST /api/analyze accepte un CSV uploadé directement

Endpoints :
  GET  /                        → healthcheck
  POST /api/analyze             → upload CSV RKWA → retourne prévisions + recommandations
  GET  /api/demo                → analyse sur données de démo intégrées
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime, timedelta
from typing import Optional

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VMI Intelligence API — Motul",
    description="Analyse de stocks consignation VMI. Upload CSV RKWA → prévisions J+30 + recommandations ZMRKO.",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Descriptions articles (en réel : table MAKT SAP) ─────────────────────────

DESCRIPTIONS = {
    "MAT-001": "Huile moteur 5W30 1L",
    "MAT-002": "Huile moteur 5W40 5L",
    "MAT-003": "Huile transmission 75W90",
    "MAT-004": "Liquide de frein DOT4",
    "MAT-005": "Antigel concentré 5L",
    "MAT-006": "Huile hydraulique HV46",
    "MAT-007": "Graisse multi-usages 1kg",
    "MAT-008": "Nettoyant injecteurs",
    "MAT-009": "Huile boîte automatique",
    "MAT-010": "Additif carburant diesel",
}

# ── Données de démo intégrées (si pas de CSV uploadé) ────────────────────────

def generer_donnees_demo() -> pd.DataFrame:
    """Génère un jeu de données réaliste sans fichier externe."""
    np.random.seed(42)
    articles = [f"MAT-{i:03d}" for i in range(1, 11)]
    fournisseurs = {
        "MAT-001": "FOURN-001", "MAT-002": "FOURN-001",
        "MAT-003": "FOURN-002", "MAT-004": "FOURN-002",
        "MAT-005": "FOURN-003", "MAT-006": "FOURN-003",
        "MAT-007": "FOURN-004", "MAT-008": "FOURN-004",
        "MAT-009": "FOURN-005", "MAT-010": "FOURN-005",
    }
    prix = {
        "MAT-001": 12.5, "MAT-002": 45.0, "MAT-003": 38.0,
        "MAT-004": 8.5,  "MAT-005": 22.0, "MAT-006": 55.0,
        "MAT-007": 15.0, "MAT-008": 9.5,  "MAT-009": 62.0, "MAT-010": 7.0,
    }
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
                "MATNR": matnr,
                "LIFNR": fournisseurs[matnr],
                "BUDAT": date.strftime("%Y%m%d"),
                "MENGE": round(menge, 1),
                "WERTN": round(menge * prix[matnr], 2),
            })
    return pd.DataFrame(rows)

# ── Moteur d'analyse ──────────────────────────────────────────────────────────

def analyser_csv(df: pd.DataFrame) -> dict:
    """
    Pipeline complet : prévision + anomalies + recommandations.
    Entrée  : DataFrame avec colonnes MATNR, LIFNR, BUDAT, MENGE, WERTN
    Sortie  : dict JSON prêt à envoyer au dashboard
    """
    # Validation colonnes
    colonnes_requises = {"MATNR", "BUDAT", "MENGE", "WERTN"}
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans le CSV : {manquantes}")

    df["BUDAT"] = pd.to_datetime(df["BUDAT"], format="%Y%m%d", errors="coerce")
    df["MENGE"] = pd.to_numeric(df["MENGE"], errors="coerce").fillna(0)
    df["WERTN"] = pd.to_numeric(df["WERTN"], errors="coerce").fillna(0)
    df = df.dropna(subset=["BUDAT"])

    articles = sorted(df["MATNR"].unique())
    resultats = {}

    for matnr in articles:
        sub = df[df["MATNR"] == matnr].copy()

        # Série hebdomadaire
        serie = (
            sub.groupby("BUDAT")["MENGE"]
            .sum()
            .resample("W-MON")
            .sum()
            .fillna(0)
            .reset_index()
        )
        serie.columns = ["ds", "y"]

        if len(serie) < 4:
            continue

        y = serie["y"].values
        n = len(y)

        # ── Prévision : moyenne mobile pondérée + tendance linéaire ──────────
        # Fenêtre glissante sur les 8 dernières semaines
        fenetre = min(8, n)
        poids = np.arange(1, fenetre + 1, dtype=float)
        poids /= poids.sum()
        base_prev = float(np.dot(y[-fenetre:], poids))

        # Tendance sur les 12 dernières semaines
        n_trend = min(12, n)
        x_trend = np.arange(n_trend)
        coeffs = np.polyfit(x_trend, y[-n_trend:], 1)
        tendance_hebdo = float(coeffs[0])

        # 4 semaines futures
        derniere_date = serie["ds"].iloc[-1]
        prevision_semaines = []
        for i in range(1, 5):
            date_prev = derniere_date + timedelta(weeks=i)
            yhat = max(0, base_prev + tendance_hebdo * i)
            # Intervalle simple : ±1 écart-type historique
            std = float(np.std(y[-fenetre:]))
            prevision_semaines.append({
                "ds":         date_prev.strftime("%Y-%m-%d"),
                "yhat":       round(yhat, 1),
                "yhat_lower": round(max(0, yhat - std), 1),
                "yhat_upper": round(yhat + std, 1),
            })

        total_j30 = round(sum(s["yhat"] for s in prevision_semaines), 1)
        borne_basse = round(sum(s["yhat_lower"] for s in prevision_semaines), 1)
        borne_haute = round(sum(s["yhat_upper"] for s in prevision_semaines), 1)

        # ── Anomalies : z-score glissant ────────────────────────────────────
        anomalies = []
        rolling_mean = pd.Series(y).rolling(8, min_periods=3).mean()
        rolling_std  = pd.Series(y).rolling(8, min_periods=3).std().fillna(1)
        z_scores = (y - rolling_mean) / rolling_std

        for i, (idx_row, row) in enumerate(serie.iterrows()):
            z = float(z_scores.iloc[i]) if not np.isnan(z_scores.iloc[i]) else 0
            if abs(z) > 1.8:
                anomalies.append({
                    "ds":           row["ds"].strftime("%Y-%m-%d"),
                    "y":            round(float(row["y"]), 1),
                    "z_score":      round(z, 2),
                    "type_anomalie": "pic" if z > 0 else "creux",
                })

        # ── Statistiques ────────────────────────────────────────────────────
        moy_hebdo = round(float(np.mean(y)), 1)
        std_hebdo = round(float(np.std(y)), 1)
        cv = std_hebdo / moy_hebdo if moy_hebdo > 0 else 0.1

        # ── Recommandation PA_ALLW ───────────────────────────────────────────
        base_allw = round(min(25.0, max(2.0, cv * 100 * 0.8)), 1)
        bonus_anom = max(0, (len(anomalies) - 5) // 3) * 2.0
        pa_allw = round(min(25.0, max(2.0, base_allw + bonus_anom)), 1)

        interp_allw = (
            "Tolérance serrée — consommation stable"   if pa_allw <= 5  else
            "Tolérance modérée — légère variabilité"   if pa_allw <= 12 else
            "Tolérance large — consommation instable"
        )

        # ── Recommandation PA_EXPAM ──────────────────────────────────────────
        total_menge = sub["MENGE"].sum()
        total_wertn = sub["WERTN"].sum()
        prix_moyen  = (total_wertn / total_menge) if total_menge > 0 else 0
        pa_expam    = round(total_j30 * prix_moyen, 2)

        # ── Score de risque ──────────────────────────────────────────────────
        # Stock synthétique : dernier mois de consommation
        stock_estime = float(serie["y"].iloc[-4:].sum()) * 0.6
        ratio_stock  = stock_estime / total_j30 if total_j30 > 0 else 999
        jours_stock  = round(stock_estime / (total_j30 / 30), 1) if total_j30 > 0 else 999

        if ratio_stock < 0.2:
            niveau_stock = "CRITIQUE"
            couleur_stock = "rouge"
        elif ratio_stock < 0.5:
            niveau_stock = "ATTENTION"
            couleur_stock = "orange"
        else:
            niveau_stock = "OK"
            couleur_stock = "vert"

        # Score global pondéré
        s_stock = {"OK": 0, "ATTENTION": 20, "CRITIQUE": 40}[niveau_stock]
        s_anom  = min(20, len(anomalies) * 2)
        s_tol   = min(10, pa_allw / 2.5)
        score_risque = round(s_stock + s_anom + s_tol, 1)
        niveau_risque = (
            "FAIBLE"  if score_risque < 25 else
            "MODÉRÉ"  if score_risque < 55 else
            "ÉLEVÉ"
        )

        # Historique formaté pour graphique
        historique = [
            {
                "ds":    row["ds"].strftime("%Y-%m-%d"),
                "y":     round(float(row["y"]), 1),
                "anomalie": any(a["ds"] == row["ds"].strftime("%Y-%m-%d") for a in anomalies),
            }
            for _, row in serie.iterrows()
        ]

        lifnr = sub["LIFNR"].iloc[0] if "LIFNR" in sub.columns else "INCONNU"

        resultats[matnr] = {
            "matnr":       matnr,
            "description": DESCRIPTIONS.get(matnr, matnr),
            "fournisseur": lifnr,
            "horodatage":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "stats": {
                "moy_hebdo_units": moy_hebdo,
                "std_hebdo_units": std_hebdo,
                "nb_semaines":     len(serie),
                "nb_anomalies":    len(anomalies),
                "periode_debut":   serie["ds"].iloc[0].strftime("%Y-%m-%d"),
                "periode_fin":     serie["ds"].iloc[-1].strftime("%Y-%m-%d"),
            },
            "prevision_j30": {
                "total_units":  total_j30,
                "borne_basse":  borne_basse,
                "borne_haute":  borne_haute,
                "par_semaine":  prevision_semaines,
            },
            "zmrko": {
                "PA_ALLW": {
                    "valeur":         pa_allw,
                    "unite":          "%",
                    "interpretation": interp_allw,
                },
                "PA_EXPAM": {
                    "valeur":      pa_expam,
                    "borne_basse": round(borne_basse * prix_moyen, 2),
                    "borne_haute": round(borne_haute * prix_moyen, 2),
                    "unite":       "EUR",
                    "prix_unitaire_moyen": round(prix_moyen, 4),
                },
            },
            "alertes": {
                "stock": {
                    "niveau":          niveau_stock,
                    "couleur":         couleur_stock,
                    "jours_couverture": jours_stock,
                    "message": (
                        f"Stock insuffisant — rupture estimée dans {jours_stock} jours" if niveau_stock == "CRITIQUE" else
                        f"Stock bas — surveiller ({jours_stock} jours)" if niveau_stock == "ATTENTION" else
                        f"Stock suffisant ({jours_stock} jours de couverture)"
                    ),
                },
            },
            "risque_global": {
                "score":   score_risque,
                "niveau":  niveau_risque,
                "couleur": "rouge" if score_risque >= 55 else "orange" if score_risque >= 25 else "vert",
            },
            "historique":  historique[-52:],  # dernière année seulement
            "anomalies":   anomalies,
        }

    return resultats

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Système"])
def healthcheck():
    return {
        "status":  "ok",
        "service": "VMI Intelligence API — Motul",
        "version": "2.0.0",
        "message": "POST /api/analyze avec un CSV RKWA pour lancer l'analyse",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/analyze", tags=["Analyse"])
async def analyser(file: UploadFile = File(...)):
    """
    Upload d'un fichier CSV RKWA exporté de SAP.
    Colonnes attendues : MATNR ; LIFNR ; BUDAT (YYYYMMDD) ; MENGE ; WERTN
    Séparateur : point-virgule (;)
    Retourne : prévisions J+30 + recommandations PA_ALLW / PA_EXPAM + alertes
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Fichier CSV requis (.csv)")

    contenu = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contenu), sep=";", dtype=str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture CSV : {e}")

    try:
        resultats = analyser_csv(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    nb_articles = len(resultats)
    nb_alertes  = sum(1 for r in resultats.values() if r["alertes"]["stock"]["niveau"] != "OK")
    total_j30   = sum(r["prevision_j30"]["total_units"] for r in resultats.values())
    risque_max  = max((r["risque_global"]["score"] for r in resultats.values()), default=0)

    return {
        "meta": {
            "fichier":         file.filename,
            "nb_articles":     nb_articles,
            "nb_alertes":      nb_alertes,
            "total_j30_units": round(total_j30, 1),
            "risque_max":      risque_max,
            "horodatage":      datetime.now().isoformat(),
        },
        "articles": resultats,
    }


@app.get("/api/demo", tags=["Analyse"])
def demo():
    """
    Analyse sur données de démonstration intégrées.
    Aucun upload requis — idéal pour tester ou présenter.
    """
    df = generer_donnees_demo()
    resultats = analyser_csv(df)

    nb_alertes = sum(1 for r in resultats.values() if r["alertes"]["stock"]["niveau"] != "OK")
    total_j30  = sum(r["prevision_j30"]["total_units"] for r in resultats.values())
    risque_max = max((r["risque_global"]["score"] for r in resultats.values()), default=0)

    return {
        "meta": {
            "fichier":         "DEMO — données synthétiques",
            "nb_articles":     len(resultats),
            "nb_alertes":      nb_alertes,
            "total_j30_units": round(total_j30, 1),
            "risque_max":      risque_max,
            "horodatage":      datetime.now().isoformat(),
        },
        "articles": resultats,
    }


@app.get("/api/zmrko/{matnr}", tags=["ZMRKO — Intégration ABAP"])
def zmrko_article(matnr: str):
    """
    Retourne PA_ALLW + PA_EXPAM pour un article via les données de démo.
    En production : appeler POST /api/analyze d'abord.
    Appelable depuis ZMRKO via CL_HTTP_CLIENT.
    """
    df = generer_donnees_demo()
    resultats = analyser_csv(df)
    matnr = matnr.upper()
    if matnr not in resultats:
        raise HTTPException(status_code=404, detail=f"Article {matnr} introuvable")
    r = resultats[matnr]
    return {
        "matnr":       matnr,
        "description": r["description"],
        "PA_ALLW":     r["zmrko"]["PA_ALLW"],
        "PA_EXPAM":    r["zmrko"]["PA_EXPAM"],
        "alertes":     r["alertes"],
        "risque_global": r["risque_global"],
    }
