"""Service web FastAPI qui sert les prédictions de churn.

Le service expose quatre endpoints HTTP :

    * ``GET  /health``           — état du service (utilisé par le
      monitoring d'infrastructure).
    * ``GET  /model-info``       — métadonnées sur le modèle chargé.
    * ``POST /predict``          — score d'**un** client.
    * ``POST /predict/batch``    — score d'**une liste** de clients.

Au démarrage du serveur, on charge **une seule fois** le modèle et le
builder de features, puis on les garde en mémoire jusqu'à l'arrêt.
Charger le modèle à chaque requête serait beaucoup trop lent.
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# C'est l'application web qui répond aux appels du CRM.
# Quand le serveur démarre, il charge le modèle gagnant
# (depuis MLflow) et son pipeline de features (depuis le
# disque). Ensuite, pour chaque appel à /predict, il :
#   - vérifie que la fiche client est correcte,
#   - lui applique les mêmes transformations qu'à
#     l'entraînement,
#   - demande au modèle sa probabilité de churn,
#   - renvoie cette probabilité avec un niveau de risque
#     et l'action recommandée.
# ───────────────────────────────────────────────────────

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    ChurnPrediction,
    PolicyFeatures,
)
from src.api.utils import (
    RETENTION_ACTIONS,
    classify_risk,
    estimate_clv,
    features_to_dataframe,
)
from src.features.actuarial_features import ActuarialFeatureBuilder
from src.features.build_features import NON_FEATURE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Configuration (via variables d'environnement) ───────────────────────────

# Adresse du serveur MLflow. Surcharger avec ``MLFLOW_TRACKING_URI``
# si MLflow tourne ailleurs que sur localhost.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Emplacement du builder de features sauvegardé pendant l'entraînement.
# On synchronise l'API et le trainer via ce fichier sur disque pour que
# l'API n'ait aucune dépendance MLflow pour la partie features.
FEATURE_BUILDER_PATH = Path(os.getenv("FEATURE_BUILDER_PATH", "models/feature_builder.pkl"))

# Quel modèle servir. ``"Production"`` est le tag standard MLflow pour
# la version actuellement validée pour le trafic réel.
MODEL_NAME = os.getenv("MODEL_NAME", "insurance_churn_xgb")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")


# ── État applicatif ─────────────────────────────────────────────────────────

# Ces valeurs sont remplies une fois pour toutes au démarrage par
# :func:`lifespan`, puis lues sans modification par les handlers.
state: dict[str, Any] = {
    "model": None,
    "feature_builder": None,
}


def _load_feature_builder() -> ActuarialFeatureBuilder:
    """Renvoie le builder de features, en privilégiant celui de l'entraînement.

    Si le fichier sauvegardé n'existe pas (ex. on lance l'API contre une
    arborescence sans entraînement préalable), on retombe sur un builder
    neuf. L'API continue à répondre mais la feature "prime vs marché"
    devient constante — un signal clair que l'étape d'entraînement a été
    sautée.

    Returns:
        Un :class:`ActuarialFeatureBuilder`, idéalement entraîné sur les
        données historiques.
    """
    if FEATURE_BUILDER_PATH.exists():
        logger.info("Chargement du builder de features depuis %s", FEATURE_BUILDER_PATH)
        return joblib.load(FEATURE_BUILDER_PATH)
    logger.warning(
        "Builder de features absent (%s) — fallback sur un builder vide. "
        "Lancer l'entrainement pour retrouver toute la fidelite des features.",
        FEATURE_BUILDER_PATH,
    )
    return ActuarialFeatureBuilder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et le builder une fois pour toutes au démarrage.

    Tout ce qui est créé ici reste en mémoire jusqu'à l'arrêt du serveur.
    Charger le modèle à chaque requête serait beaucoup trop lent
    (chargement de plusieurs centaines de mégaoctets).
    """
    logger.info(
        "Chargement du modele %s@%s depuis MLflow (%s) ...",
        MODEL_NAME, MODEL_STAGE, MLFLOW_TRACKING_URI,
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        state["feature_builder"] = _load_feature_builder()
        logger.info("Modele charge avec succes")
    except Exception as exc:  # noqa: BLE001 — on veut tout afficher en clair
        logger.error("Echec du chargement du modele : %s", exc)
        raise
    yield
    logger.info("Arret du service")


app = FastAPI(
    title="API de prediction de churn assurance",
    description=(
        "Predit la probabilite de depart de chaque assure. "
        "Concu par un consultant data au profil actuariel."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# On autorise les appels venant de n'importe quel domaine (CORS *).
# En production, restreindre à la liste exacte des origines de confiance.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Logique de prédiction ───────────────────────────────────────────────────


def _predict_single(policy: PolicyFeatures) -> ChurnPrediction:
    """Score un client unique et enrobe la sortie d'informations métier.

    Args:
        policy: La fiche client à scorer (déjà validée par Pydantic).

    Returns:
        Un :class:`ChurnPrediction` complet : probabilité, niveau de
        risque, action recommandée, valeur résiduelle estimée et version
        du modèle ayant fait la prédiction.
    """
    model = state["model"]
    builder: ActuarialFeatureBuilder = state["feature_builder"]

    df_raw = features_to_dataframe(policy)
    df_features = builder.transform(df_raw)

    # On retire les colonnes que le modèle n'attend pas (identifiants,
    # libellés textuels qui existent déjà sous forme encodée).
    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    df_model_input = df_features[feature_cols]

    probability = float(model.predict_proba(df_model_input)[:, 1][0])
    risk_tier = classify_risk(probability)

    return ChurnPrediction(
        policy_id=policy.policy_id,
        churn_probability=round(probability, 4),
        risk_tier=risk_tier,
        recommended_action=RETENTION_ACTIONS[risk_tier],
        estimated_clv=estimate_clv(policy),
        model_version=f"{MODEL_NAME}@{MODEL_STAGE}",
    )


# ── Endpoints HTTP ──────────────────────────────────────────────────────────


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Confirme que le service est démarré et indique si le modèle est chargé.

    Returns:
        Un petit JSON avec ``status: "ok"`` et un drapeau ``model_loaded``.
    """
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    """Renvoie les métadonnées identifiant le modèle actuellement servi.

    Returns:
        Dictionnaire avec ``model_name``, ``stage`` et ``tracking_uri``.
    """
    return {
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=ChurnPrediction)
def predict(policy: PolicyFeatures) -> ChurnPrediction:
    """Score un client unique et renvoie la prédiction enrichie.

    Args:
        policy: Fiche client envoyée dans le corps de la requête JSON.

    Returns:
        La prédiction complète (probabilité + niveau de risque + action).

    Raises:
        HTTPException: Code 503 si le modèle n'est pas chargé (ex. MLflow
            n'était pas joignable au démarrage).
    """
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    return _predict_single(policy)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest) -> BatchResponse:
    """Score une liste de clients et résume le risque global.

    Args:
        batch: Objet contenant la liste des fiches client à scorer.

    Returns:
        Une réponse contenant les prédictions individuelles plus un
        petit résumé : nombre de clients en risque ``high``/``critical``
        et somme de leurs primes annuelles.

    Raises:
        HTTPException: Code 503 si le modèle n'est pas chargé.
    """
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    predictions = [_predict_single(p) for p in batch.policies]

    at_risk = [p for p in predictions if p.risk_tier in ("high", "critical")]
    premium_at_risk = sum(
        next(pol.annual_premium for pol in batch.policies if pol.policy_id == p.policy_id)
        for p in at_risk
    )

    return BatchResponse(
        predictions=predictions,
        total_at_risk=len(at_risk),
        total_premium_at_risk=round(premium_at_risk, 2),
    )
