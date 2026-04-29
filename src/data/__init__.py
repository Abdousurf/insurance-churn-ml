"""Sous-package "données" : ingestion et préparation des fichiers source.

Ce module est responsable de transformer la matière brute (un dataset
public hollandais d'assurance) en deux fichiers parquet prêts à
l'emploi pour la suite du pipeline. Tout le reste du projet ne lit que
ces parquets — il n'a jamais besoin de connaître le format d'origine.

Symboles publics ré-exportés ici :

    * :func:`download_coil2000`   — télécharge les fichiers source UCI
    * :func:`load_coil2000`       — les charge en DataFrames pandas
    * :func:`save_processed`      — écrit les parquets de sortie
    * :func:`main`                — pipeline de bout en bout (utilisé par DVC et le Makefile)
"""

from src.data.download_opendata import (
    download_coil2000,
    load_coil2000,
    main,
    save_processed,
)

__all__ = [
    "download_coil2000",
    "load_coil2000",
    "main",
    "save_processed",
]
