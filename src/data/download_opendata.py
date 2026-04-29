"""Téléchargement et préparation du dataset public d'assurance.

Ce script effectue deux opérations, dans l'ordre :

    1. **Téléchargement** des fichiers source COIL 2000 depuis l'archive
       publique de l'université UC Irvine. Ce dataset contient les données
       anonymisées d'environ 9 000 assurés hollandais (86 colonnes) plus
       une étiquette binaire "ce client a-t-il souscrit l'offre ?". On
       utilise cette étiquette comme proxy de "ce client va-t-il partir ?",
       car les datasets publics de churn pur sont très rares.

    2. **Traduction** de ces 86 codes obscurs en néerlandais (``MGEMLEEF``,
       ``PPERSAUT``…) vers les champs métier que le reste du projet attend
       (``insured_age``, ``annual_premium``, ``tenure_months``…). Tous les
       modules en aval — feature engineering, entraînement, API, monitoring
       — voient ainsi la **même structure** de données, ce qui évite les
       décalages de schéma.

Le mapping COIL → schéma assurance est documenté dans la fonction
:func:`_coil_to_insurance_schema`. Il est déterministe (seed fixe) afin
que les fichiers parquet produits soient reproductibles entre exécutions.

Source du dataset :
    https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000

Licence : Creative Commons Attribution 4.0 (CC BY 4.0)
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# Il télécharge un vrai dataset d'assurance depuis Internet,
# le convertit dans le format de "fiche client" que le reste
# du projet utilise, et écrit deux fichiers parquet sur disque
# (un pour entraîner le modèle, un pour le tester).
# ───────────────────────────────────────────────────────

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Emplacements sur disque : "raw" pour les fichiers téléchargés tels quels,
# "processed" pour les parquets de sortie déjà au bon format.
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "coil2000"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# URL des quatre fichiers source COIL 2000 sur le serveur UCI.
COIL_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt"
COIL_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt"
COIL_DICT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/TicDataDescr.txt"
COIL_TARGETS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt"

# Graine aléatoire : tous les choix "aléatoires" du script utilisent cette
# valeur, ce qui rend le résultat 100 % reproductible.
RANDOM_SEED = 42

# Noms des 86 colonnes brutes du fichier COIL, dans leur ordre d'origine.
# La dernière colonne (``CARAVAN``) est la cible : "ce client a-t-il souscrit
# une assurance caravane ?"
COIL_COLUMNS = [
    "MOSTYPE", "MAANTHUI", "MGEMOMV", "MGEMLEEF", "MOSHOOFD",
    "MGODRK", "MGODPR", "MGODOV", "MGODGE", "MRELGE", "MRELSA", "MRELOV",
    "MFALLEEN", "MFGEKIND", "MFWEKIND", "MOPLHOOG", "MOPLMIDD", "MOPLLAAG",
    "MBERHOOG", "MBERZELF", "MBERBOER", "MBERMIDD", "MBERARBG", "MBERARBO",
    "MSKA", "MSKB1", "MSKB2", "MSKC", "MSKD", "MHHUUR", "MHKOOP",
    "MAUT1", "MAUT2", "MAUT0", "MZFONDS", "MZPART", "MINKM30", "MINK3045",
    "MINK4575", "MINK7512", "MINK123M", "MINKGEM", "MKOOPKLA",
    "PWAPART", "PWABEDR", "PWALAND", "PPERSAUT", "PBESAUT", "PMOTSCO",
    "PVRAAUT", "PAANHANG", "PTRACTOR", "PWERKT", "PBROM", "PLEVEN",
    "PPERSONG", "PGEZONG", "PWAOREG", "PBRAND", "PZEILPL", "PPLEZIER",
    "PFIETS", "PINBOED", "PBYSTAND", "AWAPART", "AWABEDR", "AWALAND",
    "APERSAUT", "ABESAUT", "AMOTSCO", "AVRAAUT", "AAANHANG", "ATRACTOR",
    "AWERKT", "ABROM", "ALEVEN", "APERSONG", "AGEZONG", "AWAOREG",
    "ABRAND", "AZEILPL", "APLEZIER", "AFIETS", "AINBOED", "ABYSTAND",
    "CARAVAN",
]

# Pour chaque grande catégorie de produit, quelle est la colonne COIL qui
# représente le mieux son montant. On s'en sert pour décider quel produit
# "domine" le portefeuille du client.
LOB_DRIVERS = {
    "auto": "PPERSAUT",       # assurance auto particulier
    "home": "PBRAND",         # assurance habitation / incendie
    "liability": "PWAPART",   # responsabilité civile
    "health": "PLEVEN",       # assurance vie / santé
}

# Conversion des classes ordinales COIL (0–9) en euros annuels.
# Les coefficients ont été choisis pour que les primes finales tombent
# dans une fourchette réaliste de 200–2 500 €/an.
PREMIUM_EUR_PER_CLASS = {
    "PPERSAUT": 120,  # auto
    "PBRAND": 80,     # habitation
    "PWAPART": 60,    # responsabilité civile
    "PLEVEN": 90,     # vie
    "PMOTSCO": 50,    # cyclomoteur
}


# ── Téléchargement ──────────────────────────────────────────────────────────


def download_coil2000() -> dict[str, Path]:
    """Télécharge les quatre fichiers source COIL 2000 depuis le serveur UCI.

    Si un fichier est déjà présent sur disque, il n'est pas re-téléchargé.

    Returns:
        Dictionnaire qui associe chaque rôle de fichier (``train``, ``test``,
        ``targets``, ``dict``) à son chemin local.

    Raises:
        requests.HTTPError: Si l'un des téléchargements échoue.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    urls = {
        "train": COIL_TRAIN_URL,
        "test": COIL_TEST_URL,
        "targets": COIL_TARGETS_URL,
        "dict": COIL_DICT_URL,
    }

    for name, url in urls.items():
        dest = RAW_DIR / f"coil2000_{name}.txt"
        if dest.exists():
            log.info("  [keep] %s deja present", dest.name)
        else:
            log.info("  [get ] telechargement %s ...", url)
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            dest.write_bytes(response.content)
            log.info("  [ok  ] sauvegarde dans %s", dest)
        files[name] = dest

    return files


# ── Lecture des fichiers bruts ──────────────────────────────────────────────


def load_coil2000(files: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les fichiers bruts COIL en deux DataFrames pandas (train et test).

    Le fichier d'entraînement contient déjà les 86 colonnes (cible incluse),
    alors que le fichier de test n'en a que 85 — sa cible est dans un fichier
    séparé qu'on rattache ici.

    Args:
        files: Dictionnaire renvoyé par :func:`download_coil2000`.

    Returns:
        Une paire ``(df_train, df_test)``. Chaque DataFrame contient les 86
        colonnes COIL nommées dans leur ordre officiel ; la cible se trouve
        dans la colonne ``CARAVAN``.
    """
    df_train = pd.read_csv(files["train"], sep="\t", header=None, names=COIL_COLUMNS)
    log.info(
        "Train : %d lignes, %d colonnes | taux de positifs : %.1f%%",
        len(df_train), df_train.shape[1], df_train["CARAVAN"].mean() * 100,
    )

    df_test = pd.read_csv(files["test"], sep="\t", header=None, names=COIL_COLUMNS[:-1])
    targets = pd.read_csv(files["targets"], header=None, names=["CARAVAN"])
    df_test["CARAVAN"] = targets["CARAVAN"].values
    log.info(
        "Test  : %d lignes | taux de positifs : %.1f%%",
        len(df_test), df_test["CARAVAN"].mean() * 100,
    )

    return df_train, df_test


# ── Traduction COIL → schéma assurance ──────────────────────────────────────


def _derive_lob(df: pd.DataFrame) -> np.ndarray:
    """Choisit une seule "branche d'assurance" par client (auto, habitation, etc.).

    Le dataset COIL liste plusieurs produits côte à côte avec leurs montants.
    On choisit pour chaque client le produit pour lequel il paie le plus —
    c'est sa branche dominante. Si un client n'a aucun montant renseigné
    pour les quatre branches qu'on suit, on lui attribue ``"auto"`` par
    défaut (la branche la plus courante).

    Args:
        df: DataFrame brute COIL.

    Returns:
        Tableau de chaînes, un élément par ligne, parmi
        ``{"auto", "home", "liability", "health"}``.
    """
    lob_names = list(LOB_DRIVERS.keys())          # l'ordre sert d'index numérique
    driver_cols = [LOB_DRIVERS[n] for n in lob_names]
    contributions = df[driver_cols].to_numpy()    # shape (n_lignes, 4)

    # Pour chaque ligne, on garde la colonne avec le plus gros montant.
    dominant_lob_idx = np.argmax(contributions, axis=1)

    # Si toutes les contributions sont nulles, on retombe sur "auto" (index 0).
    no_signal = contributions.sum(axis=1) == 0
    dominant_lob_idx[no_signal] = 0

    return np.array(lob_names)[dominant_lob_idx]


def _derive_annual_premium(df: pd.DataFrame) -> pd.Series:
    """Convertit les classes ordinales COIL en une prime annuelle en euros.

    COIL stocke les contributions sous forme de petits entiers (0–9), chaque
    classe représentant une tranche de cotisation. On multiplie chaque classe
    par un coefficient en euros plausible par produit, puis on additionne le
    tout. Le minimum est borné à 50 € pour ne jamais produire une prime nulle
    ou négative (le schéma Pydantic les rejetterait).

    Args:
        df: DataFrame brute COIL.

    Returns:
        Série pandas des primes annuelles en euros.
    """
    premium = pd.Series(0.0, index=df.index)
    for col, eur_per_class in PREMIUM_EUR_PER_CLASS.items():
        if col in df.columns:
            premium = premium + df[col] * eur_per_class
    return premium.clip(lower=50.0).astype(float)


def _derive_tenure_months(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Déduit une ancienneté plausible (en mois) à partir d'un proxy COIL.

    Le champ ``MAANTHUI`` indique le nombre d'habitations occupées par le
    client (1–9). C'est un bon indicateur de stabilité de vie, donc on
    l'utilise comme proxy d'ancienneté. On y ajoute un petit bruit (0–5
    mois) pour éviter que toutes les anciennetés tombent sur des multiples
    d'année exacts.

    Args:
        df: DataFrame brute COIL.
        rng: Générateur aléatoire seedé pour le bruit reproductible.

    Returns:
        Tableau d'anciennetés en mois (toujours ≥ 0).
    """
    tenure_years = df["MAANTHUI"].clip(lower=1).astype(int)
    base_months = (tenure_years * 6).to_numpy()
    jitter = rng.integers(0, 6, size=len(df))
    return np.clip(base_months + jitter, 0, None).astype(int)


def _derive_insured_age(df: pd.DataFrame) -> np.ndarray:
    """Convertit la classe d'âge COIL (1–6) en un âge représentatif en années.

    COIL range les clients dans 6 tranches d'âge : 20–30, 30–40, …, 70+.
    On prend un âge médian par tranche puis on borne à l'intervalle accepté
    par l'API (18–100 ans).

    Args:
        df: DataFrame brute COIL.

    Returns:
        Tableau d'âges entiers.
    """
    age_class = df["MGEMLEEF"].clip(lower=1, upper=6).to_numpy()
    return np.clip(age_class * 10 + 15, 18, 100).astype(int)


def _derive_channel(df: pd.DataFrame) -> np.ndarray:
    """Répartit les clients sur les quatre canaux d'acquisition.

    COIL n'enregistre pas par quel canal le client a été acquis ; on utilise
    sa classe de pouvoir d'achat (``MKOOPKLA``, entier 1–8) comme clé de
    répartition déterministe : pour la même entrée, on obtient toujours le
    même canal.

    Args:
        df: DataFrame brute COIL.

    Returns:
        Tableau de chaînes parmi ``{"Direct", "Broker", "Online", "Agent"}``.
    """
    channels = np.array(["Direct", "Broker", "Online", "Agent"])
    spreading_key = df["MKOOPKLA"].fillna(0).astype(int).to_numpy() % len(channels)
    return channels[spreading_key]


def _coil_to_insurance_schema(
    df_coil: pd.DataFrame,
    *,
    seed: int = RANDOM_SEED,
    id_offset: int = 0,
) -> pd.DataFrame:
    """Traduit un DataFrame COIL brut vers le format "fiche client" du projet.

    Toutes les colonnes produites ici sont celles que le reste du projet
    attend : feature engineering, entraînement, API et monitoring lisent
    tous cette même structure. Voir le docstring du module pour le pourquoi
    de cette traduction.

    Args:
        df_coil: DataFrame brute COIL (avec ses 86 colonnes d'origine).
        seed: Graine aléatoire pour le bruit déterministe injecté afin
            d'éviter que tout le monde tombe sur des nombres ronds.
            Doit rester identique entre runs pour la reproductibilité.
        id_offset: Numéro de départ pour les ``policy_id`` synthétiques.
            Utile pour que les IDs train et test ne se chevauchent pas.

    Returns:
        DataFrame avec une ligne par client et les 15 champs standards du
        projet (``policy_id``, ``lob``, ``annual_premium``, …, plus la
        cible ``churn_label``).
    """
    rng = np.random.default_rng(seed)
    n = len(df_coil)

    # Identifiants et étiquettes
    policy_id = [f"POL{(id_offset + i):07d}" for i in range(n)]
    lob = _derive_lob(df_coil)
    churn_label = df_coil["CARAVAN"].astype(int).to_numpy()

    # Ancienneté et fidélité
    tenure_months = _derive_tenure_months(df_coil, rng)
    renewal_count = (tenure_months // 12).astype(int)

    # Tarification
    annual_premium = _derive_annual_premium(df_coil).to_numpy()
    # Évolution annuelle de la prime — COIL n'a pas d'historique donc on
    # tire une distribution plausible centrée sur une légère hausse de +2 %.
    premium_change_pct = rng.normal(loc=2.0, scale=5.0, size=n).round(1)

    # Sinistres
    recent_claims = df_coil[["AWAPART", "APERSAUT"]].sum(axis=1).clip(upper=10).astype(int).to_numpy()
    all_time_claims = recent_claims + (tenure_months // 24).astype(int)
    claim_settled_pct = np.clip(1.0 - 0.05 * recent_claims, 0.4, 1.0).astype(float)
    days_to_settle_avg = (25.0 + 5.0 * recent_claims).astype(float)

    # Profil client
    insured_age = _derive_insured_age(df_coil)
    channel = _derive_channel(df_coil)

    # Portefeuille
    contribution_cols = [
        c for c in df_coil.columns
        if c.startswith("P") and c not in ("PWERKT",)  # PWERKT est un marqueur, pas un produit
    ]
    policy_count_active = (
        (df_coil[contribution_cols] > 0).sum(axis=1).clip(lower=1).astype(int).to_numpy()
    )

    # Engagement : nombre de jours depuis le dernier contact. Pas d'historique
    # disponible donc on dérive depuis l'ancienneté avec un petit bruit.
    last_contact_days = ((tenure_months % 90) + rng.integers(10, 60, size=n)).astype(int)

    return pd.DataFrame({
        "policy_id": policy_id,
        "lob": lob,
        "annual_premium": annual_premium,
        "tenure_months": tenure_months,
        "renewal_count": renewal_count,
        "claim_count_12m": recent_claims,
        "claim_count_all": all_time_claims,
        "claim_settled_pct": claim_settled_pct,
        "days_to_settle_avg": days_to_settle_avg,
        "insured_age": insured_age,
        "channel": channel,
        "policy_count_active": policy_count_active,
        "premium_change_pct": premium_change_pct,
        "last_contact_days": last_contact_days,
        "churn_label": churn_label,
    })


# ── Sauvegarde ──────────────────────────────────────────────────────────────


def save_processed(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[Path, Path]:
    """Écrit les DataFrames train et test dans des fichiers parquet sur disque.

    Args:
        df_train: DataFrame d'entraînement au format "fiche client".
        df_test: DataFrame de test au format "fiche client".

    Returns:
        Une paire ``(chemin_train, chemin_test)`` des deux fichiers écrits.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "insurance_churn_train.parquet"
    test_path = PROCESSED_DIR / "insurance_churn_test.parquet"
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    log.info("[ok  ] ecrit %s (%d lignes)", train_path.name, len(df_train))
    log.info("[ok  ] ecrit %s (%d lignes)", test_path.name, len(df_test))
    return train_path, test_path


def main() -> None:
    """Pipeline complet : télécharge, traduit et sauvegarde les données.

    Après exécution, le projet dispose de deux parquets prêts à l'emploi
    dans ``data/processed/`` au format attendu par tous les autres modules.
    """
    files = download_coil2000()
    df_train_raw, df_test_raw = load_coil2000(files)

    # On utilise un offset différent pour le test set afin que les
    # ``policy_id`` train et test ne se chevauchent jamais.
    df_train = _coil_to_insurance_schema(df_train_raw, id_offset=0)
    df_test = _coil_to_insurance_schema(df_test_raw, id_offset=len(df_train_raw))

    train_path, test_path = save_processed(df_train, df_test)

    log.info(
        "Termine. Taux de churn train : %.1f%% | test : %.1f%%",
        df_train["churn_label"].mean() * 100,
        df_test["churn_label"].mean() * 100,
    )
    log.info("Fichier train : %s", train_path)
    log.info("Fichier test  : %s", test_path)


if __name__ == "__main__":
    main()
