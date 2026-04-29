"""Schémas Pydantic décrivant les entrées et sorties de l'API.

Pydantic est une bibliothèque qui valide automatiquement le format des
données qu'on reçoit et qu'on renvoie. C'est l'équivalent d'un "contrat"
entre l'API et ses appelants : si la requête ne respecte pas le format
attendu (mauvais type, valeur hors plage…), Pydantic la rejette en
amont avec un message d'erreur clair, sans qu'on ait à écrire la
moindre validation à la main.

Ce module définit quatre schémas :

    * :class:`PolicyFeatures`  — les données d'**un client** en entrée.
    * :class:`ChurnPrediction` — la prédiction renvoyée pour un client.
    * :class:`BatchRequest`    — entrée du mode "score plusieurs clients".
    * :class:`BatchResponse`   — sortie du mode batch.
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# Il décrit "à quoi doit ressembler" une requête envoyée à
# l'API et "à quoi va ressembler" la réponse :
#   - quels champs sont obligatoires
#   - quels types ils doivent avoir (chiffre, texte…)
#   - quelles valeurs sont acceptées (âge entre 18 et 100, etc.)
# Si l'utilisateur envoie quelque chose d'incorrect, l'API
# refuse poliment au lieu de planter.
# ───────────────────────────────────────────────────────

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PolicyFeatures(BaseModel):
    """Données d'un client telles que le client (CRM, app web…) les envoie.

    Attributes:
        policy_id: Identifiant unique de la police (ex. ``"POL0012345"``).
        lob: Branche d'assurance — l'une parmi
            ``{"auto", "home", "liability", "health"}``.
        annual_premium: Prime annuelle payée par le client, en euros.
        tenure_months: Ancienneté du contrat, en mois.
        renewal_count: Nombre de renouvellements déjà effectués.
        claim_count_12m: Nombre de sinistres déclarés sur les 12 derniers mois.
        claim_count_all: Nombre total de sinistres déclarés depuis le début.
        claim_settled_pct: Fraction des sinistres déjà réglés (entre 0 et 1).
        days_to_settle_avg: Délai moyen de règlement des sinistres en jours.
        insured_age: Âge de l'assuré principal (entre 18 et 100).
        channel: Canal d'acquisition — l'un parmi
            ``{"Direct", "Broker", "Online", "Agent"}``.
        policy_count_active: Nombre total de polices actives détenues.
        premium_change_pct: Variation de la prime cette année (en pourcent).
        last_contact_days: Jours écoulés depuis le dernier contact, ou
            ``None`` si inconnu.
    """

    policy_id: str
    lob: str = Field(..., description="Branche : auto, home, liability, health")
    annual_premium: float = Field(..., gt=0, description="Prime annuelle en EUR")
    tenure_months: int = Field(..., ge=0, description="Ancienneté du contrat en mois")
    renewal_count: int = Field(0, ge=0)
    claim_count_12m: int = Field(0, ge=0)
    claim_count_all: int = Field(0, ge=0)
    claim_settled_pct: float = Field(1.0, ge=0, le=1)
    days_to_settle_avg: float = Field(30.0, ge=0)
    insured_age: int = Field(..., ge=18, le=100)
    channel: str = Field(..., description="Canal d'acquisition : Direct, Broker, Online, Agent")
    policy_count_active: int = Field(1, ge=1)
    premium_change_pct: float = Field(0.0)
    last_contact_days: Optional[int] = None

    @field_validator("lob")
    @classmethod
    def validate_lob(cls, v: str) -> str:
        """Vérifie la branche et la passe en minuscules.

        Args:
            v: Valeur reçue pour le champ ``lob``.

        Returns:
            La branche normalisée en minuscules.

        Raises:
            ValueError: Si la valeur ne fait pas partie des branches connues.
        """
        allowed = {"auto", "home", "liability", "health"}
        if v.lower() not in allowed:
            raise ValueError(f"lob doit etre l'un de {allowed}")
        return v.lower()

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        """Vérifie que le canal d'acquisition est connu.

        Args:
            v: Valeur reçue pour le champ ``channel``.

        Returns:
            La valeur inchangée si elle est valide.

        Raises:
            ValueError: Si la valeur ne fait pas partie des canaux connus.
        """
        allowed = {"Direct", "Broker", "Online", "Agent"}
        if v not in allowed:
            raise ValueError(f"channel doit etre l'un de {allowed}")
        return v


class ChurnPrediction(BaseModel):
    """Prédiction renvoyée pour un client.

    Attributes:
        policy_id: Identifiant de la police scorée (recopié de l'entrée).
        churn_probability: Probabilité de churn entre 0 et 1.
        risk_tier: Niveau de risque
            (``"low"``, ``"medium"``, ``"high"``, ``"critical"``).
        recommended_action: Action recommandée à l'équipe rétention.
        estimated_clv: Valeur résiduelle estimée du client, en euros.
        model_version: Version du modèle ayant fait la prédiction.
    """

    policy_id: str
    churn_probability: float
    risk_tier: str
    recommended_action: str
    estimated_clv: float
    model_version: str


class BatchRequest(BaseModel):
    """Requête de scoring pour plusieurs clients à la fois.

    Attributes:
        policies: Liste des fiches client à scorer.
    """

    policies: list[PolicyFeatures]


class BatchResponse(BaseModel):
    """Réponse du mode batch avec les prédictions et un résumé du risque.

    Attributes:
        predictions: Liste des prédictions, dans l'ordre des clients reçus.
        total_at_risk: Nombre de clients en risque ``high`` ou ``critical``.
        total_premium_at_risk: Cumul des primes annuelles des clients en
            risque ``high`` ou ``critical``, en euros.
    """

    predictions: list[ChurnPrediction]
    total_at_risk: int
    total_premium_at_risk: float
