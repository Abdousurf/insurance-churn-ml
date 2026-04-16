"""
Open Data Ingestion — UCI Insurance Company Benchmark (COIL 2000)
=================================================================
Source  : UCI Machine Learning Repository
URL     : https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000
Licence : Creative Commons Attribution 4.0 International (CC BY 4.0)

Dataset : 9 068 clients hollandais, 86 features socio-démographiques + historique assurances
Target  : CARAVAN — a-t-il souscrit une assurance caravane ? (binaire, déséquilibré ~6%)

Pourquoi ce dataset pour le churn/acquisition ML ?
  → Problème de classification binaire déséquilibré (réaliste en assurance)
  → Features mixtes : démographie + comportement + types de contrats détenus
  → Signal faible (6% de positifs) → travail sur threshold, calibration, SHAP
  → Riche en features corrélées → sélection de features réaliste

Mapping actuariel :
  - PPERSAUT / PBRAND → premium équivalent (proxy tarification)
  - AWAPART / AWALAND → sinistres déclarés (proxy claims history)
  - MINKGEM → revenu moyen du quartier (proxy risk factor)
  - APERSAUT → nb contrats auto (proxy multi-line discount)
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "coil2000"

# UCI ML Repository — COIL 2000 direct download URLs
COIL_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt"
COIL_TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt"
COIL_DICT_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/TicDataDescr.txt"
COIL_TARGET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt"

# Noms des 86 colonnes COIL 2000 (ordre officiel)
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
    "CARAVAN",  # target : 1 = a souscrit assurance caravane
]


def download_coil2000() -> dict[str, Path]:
    """Télécharge les fichiers COIL 2000 depuis UCI ML Repository."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = {}

    urls = {
        "train": COIL_TRAIN_URL,
        "test":  COIL_TEST_URL,
        "targets": COIL_TARGET_URL,
        "dict":  COIL_DICT_URL,
    }

    for name, url in urls.items():
        dest = DATA_DIR / f"coil2000_{name}.txt"
        if dest.exists():
            log.info("  ✓ Déjà présent : %s", dest.name)
        else:
            log.info("  ↓ Téléchargement %s…", url)
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                dest.write_bytes(r.content)
                log.info("  ✅ Sauvegardé : %s", dest)
            except Exception as e:
                log.error("  ✗ Échec : %s", e)
                continue
        files[name] = dest

    return files


def load_coil2000(files: dict[str, Path] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge et prépare les datasets train/test COIL 2000.
    Retourne (df_train, df_test).
    """
    if files is None:
        files = {
            "train":   DATA_DIR / "coil2000_train.txt",
            "test":    DATA_DIR / "coil2000_test.txt",
            "targets": DATA_DIR / "coil2000_targets.txt",
        }

    # Train (86 colonnes dont CARAVAN en dernière position)
    df_train = pd.read_csv(files["train"], sep="\t", header=None, names=COIL_COLUMNS)
    log.info("Train : %d lignes, %d colonnes | Taux positifs : %.1f%%",
             len(df_train), df_train.shape[1], df_train["CARAVAN"].mean() * 100)

    # Test (85 colonnes — targets séparées)
    df_test = pd.read_csv(files["test"], sep="\t", header=None, names=COIL_COLUMNS[:-1])
    if files.get("targets") and Path(files["targets"]).exists():
        targets = pd.read_csv(files["targets"], header=None, names=["CARAVAN"])
        df_test["CARAVAN"] = targets["CARAVAN"].values

    log.info("Test  : %d lignes | Taux positifs : %.1f%%",
             len(df_test), df_test["CARAVAN"].mean() * 100)

    return df_train, df_test


def engineer_actuarial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering actuariel à partir du dataset COIL 2000.
    Crée des features métier cohérentes avec le pipeline churn assurance.
    """
    df = df.copy()

    # --- Premium proxy (COIL encode les primes en classes 0-9) ---
    premium_cols = ["PPERSAUT", "PBRAND", "PLEVEN", "PGEZONG", "PINBOED"]
    df["premium_total_proxy"] = df[premium_cols].sum(axis=1)

    # --- Claims proxy (COIL encode les sinistres en nombre) ---
    claim_cols = ["AWAPART", "AWALAND", "APERSAUT", "ABRAND"]
    df["claim_count_proxy"] = df[claim_cols].sum(axis=1)

    # --- Multi-line indicator (clients avec plusieurs contrats) ---
    policy_cols = [c for c in df.columns if c.startswith("P") and c not in ["PPERSAUT"]]
    df["nb_product_types"] = (df[policy_cols] > 0).sum(axis=1)
    df["is_multiline"] = (df["nb_product_types"] > 2).astype(int)

    # --- Income segment ---
    df["income_segment"] = pd.cut(
        df["MINKGEM"],
        bins=[0, 2, 4, 6, 9],
        labels=["low", "medium", "high", "affluent"],
    )

    # --- Age group (MGEMLEEF = classe d'âge moyen du quartier) ---
    df["age_group"] = pd.cut(df["MGEMLEEF"], bins=[0, 2, 4, 6, 8], labels=["young", "middle", "senior", "elderly"])

    # --- Risk score actuariel simple ---
    df["actuarial_risk_score"] = (
        df["claim_count_proxy"] * 2
        + df["premium_total_proxy"] * 0.5
        - df["is_multiline"] * 1.5
    )

    log.info("✅ %d features actuarielles créées", 6)
    return df


def save_processed(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    out_dir = DATA_DIR.parent.parent / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(out_dir / "coil2000_train.parquet", index=False)
    df_test.to_parquet(out_dir / "coil2000_test.parquet", index=False)
    log.info("✅ Parquet sauvegardés dans %s", out_dir)


if __name__ == "__main__":
    files = download_coil2000()
    df_train, df_test = load_coil2000(files)
    df_train = engineer_actuarial_features(df_train)
    df_test  = engineer_actuarial_features(df_test)
    save_processed(df_train, df_test)

    print("\n📊 Dataset COIL 2000 — prêt pour entraînement")
    print(f"  Train : {len(df_train):,} lignes | {df_train.shape[1]} features")
    print(f"  Test  : {len(df_test):,} lignes")
    print(f"  Target positifs train : {df_train['CARAVAN'].mean():.1%}")
    print("\nFeatures actuarielles ajoutées :")
    new_cols = ["premium_total_proxy", "claim_count_proxy", "nb_product_types",
                "is_multiline", "income_segment", "actuarial_risk_score"]
    print(df_train[new_cols].describe())
