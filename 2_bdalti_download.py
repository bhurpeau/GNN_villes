#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Télécharger automatiquement tous les départements de la BD ALTI depuis l'IGN.

Par défaut :
- va sur https://geoservices.ign.fr/telechargement-api/BDALTI
- récupère tous les liens .7z pointant vers data.geopf.fr/telechargement/download/BDALTI
- télécharge dans le répertoire indiqué.

Usage :
    python download_bdalti.py --outdir ./BDALTI_25m
    python download_bdalti.py --outdir ./BDALTI_25m --pattern 25M_ASC
    python download_bdalti.py --outdir ./BDALTI_75m --pattern 75M_ASC --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://geoservices.ign.fr/bdalti"
# User-Agent un peu plus propre que celui de requests par défaut
HEADERS = {
    "User-Agent": "BDALTI-downloader/1.0 (Python requests; contact: you@example.com)"
}


def list_bdalti_links(pattern=None):
    """
    Scrape la page BDALTI de l'IGN et renvoie la liste des URLs .7z.
    Optionnellement filtre les URLs contenant `pattern` (ex: "25M_ASC").
    """
    print(f"[INFO] Récupération de la page {BASE_URL}…")
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # On ne garde que les liens .7z BDALTI hébergés sur data.geopf.fr
        if (
            "data.geopf.fr/telechargement/download/BDALTI" in href
            and href.endswith(".7z")
        ):
            if pattern is None or pattern in href:
                links.add(href)

    links = sorted(links)
    print(f"[INFO] {len(links)} liens .7z trouvés (pattern={pattern!r})")
    return links


def download_file(url, outdir, sleep_between=0.5):
    """
    Télécharge un fichier en streaming dans outdir, avec reprise simple :
    - si le fichier existe déjà, on ne le retélécharge pas.
    """
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    out_path = Path(outdir) / filename

    if out_path.exists():
        print(f"[SKIP] {filename} existe déjà, on ne retélécharge pas.")
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    print(f"[DL]   {filename}")
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=2**20):  # 1 Mo
                if chunk:
                    f.write(chunk)

    tmp_path.rename(out_path)
    print(f"[OK]   {filename} téléchargé ({out_path})")

    if sleep_between > 0:
        time.sleep(sleep_between)


def main():
    parser = argparse.ArgumentParser(
        description="Téléchargement des départements BD ALTI depuis l'IGN."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Répertoire de sortie pour les .7z (sera créé si nécessaire).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "Filtre dans l'URL pour sélectionner un type de produit. "
            "Exemples : '25M_ASC', '75M_ASC'. Par défaut aucun filtre."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ne pas télécharger, seulement lister les URLs trouvées.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Pause (en secondes) entre deux téléchargements (défaut: 0.5).",
    )

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        urls = list_bdalti_links(pattern=args.pattern)
    except requests.RequestException as e:
        print(f"[ERREUR] Impossible de récupérer la page BDALTI : {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[DRY-RUN] URLs qui seraient téléchargées :")
        for u in urls:
            print(u)
        sys.exit(0)

    for url in urls:
        try:
            download_file(url, outdir=outdir, sleep_between=args.sleep)
        except requests.RequestException as e:
            print(f"[ERREUR] Téléchargement raté pour {url} : {e}", file=sys.stderr)
            # on continue avec les autres
            continue


if __name__ == "__main__":
    main()
