"""Création d'indicateurs métier "actuariels" pour la prédiction de churn.

Le modèle ne peut pas apprendre directement à partir de la prime, de
l'âge ou du nombre de sinistres pris isolément : il a besoin d'indicateurs
plus parlants. Ce module en construit toute une famille, inspirée des
pratiques de tarification en assurance :

    * **Prix vs marché** — un client qui paie plus cher que la médiane de
      sa branche a une probabilité plus élevée de partir (il fait jouer la
      concurrence).
    * **Sinistralité** — un client qui a eu plusieurs sinistres récents,
      ou qui a attendu longtemps son indemnisation, est insatisfait.
    * **Fidélité** — l'ancienneté et le nombre de renouvellements protègent
      du churn ; à l'inverse, un client jamais renouvelé est très fragile.
    * **Multi-équipement** — un client qui détient plusieurs contrats chez
      nous est "verrouillé" par les remises de package.
    * **Cycle de vie** — l'âge et le canal d'acquisition expliquent une
      large part du comportement (les jeunes en ligne comparent davantage).

Ces indicateurs sont produits par la classe :class:`ActuarialFeatureBuilder`,
qui suit l'interface scikit-learn : ``fit`` apprend les paramètres (prix
médians par branche), ``transform`` ajoute les colonnes au DataFrame.
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# Il prend un tableau de clients tel qu'on le récupère du
# CRM (prime, âge, ancienneté, nombre de sinistres…) et y
# rajoute des colonnes calculées plus utiles : "ce client
# paie-t-il trop cher ?", "est-il nouveau ?", "a-t-il eu un
# sinistre récemment ?". Le modèle apprend sur ces colonnes
# enrichies plutôt que sur les colonnes brutes, ce qui
# améliore nettement sa précision.
# ───────────────────────────────────────────────────────

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ActuarialFeatureBuilder(BaseEstimator, TransformerMixin):
    """Pipeline scikit-learn qui crée les indicateurs métier d'assurance.

    Cette classe respecte l'interface standard de scikit-learn (``fit`` /
    ``transform`` / ``fit_transform``). Elle peut donc s'intégrer
    naturellement dans un :class:`sklearn.pipeline.Pipeline` ou être
    sérialisée avec joblib pour être rejouée à l'identique en production.

    Attributes:
        market_rate_col: Nom optionnel d'une colonne contenant les prix de
            marché externes. Si ``None``, les prix de marché sont estimés
            depuis les données elles-mêmes (médiane par branche).
        market_avg_premiums_: Dictionnaire ``{branche: prix médian}``
            appris à partir des données d'entraînement. Renseigné après
            l'appel à :meth:`fit`.
    """

    def __init__(self, market_rate_col: Optional[str] = None):
        """Initialise le builder.

        Args:
            market_rate_col: Nom d'une colonne externe contenant des prix
                de marché. Si ``None``, le builder déduira les prix de
                référence depuis les données d'entraînement.
        """
        self.market_rate_col = market_rate_col
        self.market_avg_premiums_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "ActuarialFeatureBuilder":
        """Apprend la prime médiane par branche d'assurance.

        Cette information sera ensuite utilisée par :meth:`transform` pour
        comparer la prime de chaque client à la médiane de sa branche et
        détecter les contrats potentiellement surfacturés.

        Args:
            X: Table des polices d'assurance utilisée pour apprendre.
            y: Ignoré. Présent uniquement pour la compatibilité avec
                l'API scikit-learn.

        Returns:
            ``self``, pour permettre le chaînage d'appels.
        """
        if "lob" in X.columns and "annual_premium" in X.columns:
            self.market_avg_premiums_ = (
                X.groupby("lob")["annual_premium"].median().to_dict()
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ajoute toutes les colonnes calculées au DataFrame fourni.

        Args:
            X: Table des polices d'assurance à enrichir.

        Returns:
            Une copie du DataFrame d'entrée avec les colonnes
            supplémentaires (prix vs marché, sinistralité, fidélité, etc.).
        """
        df = X.copy()
        df = self._premium_features(df)
        df = self._claims_features(df)
        df = self._loyalty_features(df)
        df = self._portfolio_features(df)
        df = self._lifecycle_features(df)
        return df

    def _premium_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les colonnes liées à la tarification.

        Crée notamment ``premium_to_market_ratio`` (prime / médiane de
        branche), ``is_overpriced`` (drapeau si > 120 % du marché) et
        ``log_premium`` (échelle logarithmique pour atténuer les outliers).

        Args:
            df: Table des polices.

        Returns:
            Le DataFrame d'entrée enrichi des indicateurs de tarification.
        """
        if "annual_premium" in df.columns and "lob" in df.columns:
            # Combien ce client paie-t-il par rapport au prix médian de sa branche ?
            df["premium_to_market_ratio"] = df.apply(
                lambda r: r["annual_premium"]
                / self.market_avg_premiums_.get(r["lob"], r["annual_premium"]),
                axis=1,
            )
            # Drapeau "client surfacturé" : il paie plus de 20 % au-dessus du marché.
            df["is_overpriced"] = (df["premium_to_market_ratio"] > 1.20).astype(int)
            # On compresse les très gros montants (échelle logarithmique).
            df["log_premium"] = np.log1p(df["annual_premium"])

        if "premium_change_pct" in df.columns:
            df["premium_increased"] = (df["premium_change_pct"] > 0).astype(int)
            # Une hausse de plus de 5 % est un déclencheur fort de churn.
            df["premium_increase_gt5pct"] = (df["premium_change_pct"] > 5).astype(int)

        return df

    def _claims_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les colonnes liées à l'historique de sinistres.

        Un client mécontent de sa gestion sinistre (sinistre récent,
        sinistre non encore réglé, délai de règlement > 45 jours) est
        beaucoup plus susceptible de partir.

        Args:
            df: Table des polices.

        Returns:
            Le DataFrame d'entrée enrichi des indicateurs de sinistralité.
        """
        if "claim_count_12m" in df.columns:
            df["has_recent_claim"] = (df["claim_count_12m"] > 0).astype(int)
            df["multi_claim"] = (df["claim_count_12m"] > 1).astype(int)

        if "claim_count_all" in df.columns and "tenure_months" in df.columns:
            # Fréquence annualisée moyenne sur l'ancienneté complète.
            df["claims_per_year"] = (
                df["claim_count_all"] / (df["tenure_months"] / 12).clip(lower=0.1)
            )

        if "claim_settled_pct" in df.columns:
            # Drapeau "sinistre encore en cours" — source de mécontentement.
            df["has_unsettled_claim"] = (df["claim_settled_pct"] < 1.0).astype(int)

        if "days_to_settle_avg" in df.columns:
            df["slow_settlement"] = (df["days_to_settle_avg"] > 45).astype(int)

        return df

    def _loyalty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les colonnes liées à l'ancienneté et à la fidélité.

        Args:
            df: Table des polices.

        Returns:
            Le DataFrame d'entrée enrichi des indicateurs de fidélité
            (``tenure_years``, ``is_new_customer``, ``is_loyal_customer``,
            etc.).
        """
        if "tenure_months" in df.columns:
            df["tenure_years"] = df["tenure_months"] / 12
            df["log_tenure"] = np.log1p(df["tenure_months"])
            # Nouveau client (< 1 an) : risque de churn élevé.
            df["is_new_customer"] = (df["tenure_months"] < 12).astype(int)
            # Client fidèle (≥ 5 ans) : risque de churn faible.
            df["is_loyal_customer"] = (df["tenure_months"] >= 60).astype(int)

        if "renewal_count" in df.columns:
            df["never_renewed"] = (df["renewal_count"] == 0).astype(int)
            df["log_renewals"] = np.log1p(df["renewal_count"])

        if "last_contact_days" in df.columns:
            # Plus de 6 mois sans contact : le client peut se sentir oublié.
            df["long_since_contact"] = (df["last_contact_days"] > 180).astype(int)

        return df

    def _portfolio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les colonnes liées au multi-équipement (plusieurs contrats).

        Args:
            df: Table des polices.

        Returns:
            Le DataFrame d'entrée enrichi des indicateurs de portefeuille.
        """
        if "policy_count_active" in df.columns:
            df["is_multi_line"] = (df["policy_count_active"] > 1).astype(int)
            # Éligibilité aux remises de package (auto + habitation, etc.).
            df["multi_line_discount_eligible"] = (df["policy_count_active"] >= 2).astype(int)

        return df

    def _lifecycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les colonnes liées à l'âge et au canal d'acquisition.

        Args:
            df: Table des polices.

        Returns:
            Le DataFrame d'entrée enrichi des indicateurs de cycle de vie.
        """
        if "insured_age" in df.columns:
            # Découpage en classes d'âge calquées sur les grilles tarifaires.
            df["age_segment_encoded"] = pd.cut(
                df["insured_age"],
                bins=[0, 25, 35, 50, 65, 100],
                labels=[0, 1, 2, 3, 4],
            ).astype(int)
            # Les 18–30 ans sont le segment le plus sensible au prix.
            df["is_young_adult"] = (df["insured_age"].between(18, 30)).astype(int)

        if "channel" in df.columns:
            df["is_online_customer"] = (df["channel"] == "Online").astype(int)
            df["is_broker_customer"] = (df["channel"] == "Broker").astype(int)

        return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Raccourci : crée la matrice de features prête pour l'entraînement.

    Cette fonction enchaîne ``ActuarialFeatureBuilder.fit_transform`` puis
    retire les colonnes que le modèle ne doit pas voir (identifiants,
    libellés textuels et la cible elle-même).

    Args:
        df: Données brutes au format "fiche client" (colonnes attendues :
            ``policy_id``, ``lob``, ``annual_premium``, ``tenure_months``,
            etc.).

    Returns:
        DataFrame numérique ne contenant que les colonnes utilisables comme
        entrée du modèle.
    """
    builder = ActuarialFeatureBuilder()
    df_features = builder.fit_transform(df)

    feature_cols = [
        c for c in df_features.columns
        if c not in [
            "policy_id", "customer_id", "inception_date", "expiry_date",
            "lob", "region", "channel",   # libellés bruts (versions encodées créées plus haut)
            "churn_label",                # la cible — le modèle ne doit pas la voir
        ]
    ]
    return df_features[feature_cols]


# Description lisible de chaque feature, pratique pour la documentation
# automatique et les "model cards" exigées par la réglementation.
FEATURE_DESCRIPTIONS = {
    "premium_to_market_ratio": "Ratio prime/médiane de la branche — signal majeur d'adéquation tarifaire",
    "is_overpriced": "1 si la prime dépasse 120 % du marché",
    "log_premium": "Prime annuelle en échelle logarithmique pour atténuer les outliers",
    "has_recent_claim": "1 si au moins un sinistre déclaré sur 12 mois",
    "multi_claim": "1 si 2 sinistres ou plus sur 12 mois (effet expérience négative)",
    "claims_per_year": "Fréquence annualisée des sinistres sur toute l'ancienneté",
    "has_unsettled_claim": "1 si un sinistre est encore en cours (proxy d'insatisfaction)",
    "slow_settlement": "1 si le délai moyen de règlement dépasse 45 jours",
    "tenure_years": "Ancienneté du contrat en années",
    "is_new_customer": "1 si l'ancienneté est < 12 mois",
    "is_loyal_customer": "1 si l'ancienneté est ≥ 5 ans",
    "never_renewed": "1 si c'est la première année du contrat",
    "is_multi_line": "1 si le client détient plusieurs contrats actifs",
    "is_young_adult": "1 si l'âge est compris entre 18 et 30 ans (forte sensibilité au prix)",
    "is_online_customer": "1 si le client a été acquis via un canal digital (churn plus élevé)",
}
