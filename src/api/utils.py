"""Fonctions utilitaires de l'API (logique pure, sans dépendances lourdes).

Ces helpers ne dépendent ni de MLflow, ni de FastAPI : ce sont des
fonctions pures et facilement testables. On y a regroupé :

    * la classification d'une probabilité en niveau de risque,
    * l'estimation de la valeur résiduelle (CLV) d'un client,
    * la conversion d'une fiche client (Pydantic) vers un DataFrame.
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# Il regroupe les petits calculs métier utilisés par l'API :
#   - traduire une probabilité (0-100 %) en mot ("low", "high"…)
#   - estimer la valeur restante d'un client en euros
#   - préparer la fiche client au format pandas pour le modèle
# ───────────────────────────────────────────────────────

import pandas as pd

from src.api.schemas import PolicyFeatures

# Plages de probabilité associées à chaque niveau de risque.
# ``low`` = risque faible de churn → ``critical`` = action urgente requise.
RISK_TIERS = {
    "low": (0.00, 0.20),
    "medium": (0.20, 0.45),
    "high": (0.45, 0.70),
    "critical": (0.70, 1.00),
}

# Action recommandée à l'équipe rétention pour chaque niveau de risque.
RETENTION_ACTIONS = {
    "low": "no_action",
    "medium": "soft_retention",       # email de fidélité, petite remise
    "high": "proactive_outreach",     # appel, remise 5–10 %
    "critical": "urgent_retention",   # conseiller dédié, remise maximale
}

# Valeur par défaut quand ``last_contact_days`` n'est pas fournie : 90 jours
# correspond grosso modo à la fréquence moyenne de contact lors d'un cycle
# de renouvellement annuel.
DEFAULT_LAST_CONTACT_DAYS = 90


def classify_risk(prob: float) -> str:
    """Convertit une probabilité de churn en niveau de risque lisible.

    Args:
        prob: Probabilité prédite par le modèle, entre 0 et 1.

    Returns:
        L'un des libellés ``"low"``, ``"medium"``, ``"high"`` ou
        ``"critical"``.
    """
    for tier, (low_bound, high_bound) in RISK_TIERS.items():
        if low_bound <= prob < high_bound:
            return tier
    return "critical"


def estimate_clv(features: PolicyFeatures) -> float:
    """Estime la valeur résiduelle d'un client (Customer Lifetime Value).

    Modèle simplifié : on calcule la valeur actualisée des primes futures
    sur la durée de vie estimée restante du contrat. C'est une heuristique,
    pas un calcul actuariel rigoureux — suffisante pour prioriser des
    actions de rétention.

    Args:
        features: Fiche client (la prime et l'ancienneté sont utilisées).

    Returns:
        Valeur résiduelle estimée en euros (toujours positive).
    """
    expected_remaining_years = max(1, (60 - features.tenure_months) / 12)
    discount_rate = 0.08
    clv = (
        features.annual_premium
        * (1 - (1 + discount_rate) ** -expected_remaining_years)
        / discount_rate
    )
    return round(clv, 2)


def features_to_dataframe(policy: PolicyFeatures) -> pd.DataFrame:
    """Convertit une fiche client validée en un DataFrame d'une ligne.

    Pydantic autorise ``last_contact_days`` à être ``None`` (champ
    optionnel). Or pandas crée alors une colonne de type ``object``, que
    XGBoost rejette à l'inférence. On remplace donc explicitement les
    ``None`` par la valeur par défaut :data:`DEFAULT_LAST_CONTACT_DAYS`
    pour que la colonne reste 100 % numérique.

    Args:
        policy: Fiche client validée par Pydantic.

    Returns:
        DataFrame d'une seule ligne au format attendu par le modèle.
    """
    record = policy.model_dump()
    if record.get("last_contact_days") is None:
        record["last_contact_days"] = DEFAULT_LAST_CONTACT_DAYS
    return pd.DataFrame([record])
