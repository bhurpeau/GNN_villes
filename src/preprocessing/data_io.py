# io.py
# Fonctions de lecture, écriture, et chargement de données CSV/Parquet

import geopandas as gpd
import pandas as pd
import requests
import subprocess
import time
import os
import rasterio
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse
from rasterio.errors import RasterioIOError

# URL de base pour le téléchargement BD ALTI (25m ou 75m)
BDALTI_BASE_URL = "https://geoservices.ign.fr/bdalti"
HEADERS = {"User-Agent": "BDALTI-downloader/1.0 (Python requests)"}


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_grid_shapefile(grid_path: str) -> gpd.GeoDataFrame:
    """Charge la grille des carreaux 1km depuis un fichier shapefile/GeoPackage."""
    return gpd.read_file(grid_path)


def load_communes_shapefile(communes_path: str) -> gpd.GeoDataFrame:
    """Charge le fichier des communes (polygones) depuis un fichier shapefile/GeoPackage."""
    return gpd.read_file(communes_path)


def load_csv_data(csv_path: str, sep: str = None) -> pd.DataFrame:
    """
    Charge des données tabulaires CSV. Essaie automatiquement de deviner le séparateur si non précisé.
    """
    if sep is not None:
        return pd.read_csv(csv_path, sep=sep)
    # Auto-détection du séparateur pour éviter les erreurs silencieuses
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if first_line.count(";") > first_line.count(","):
            return pd.read_csv(csv_path, sep=";")
        else:
            return pd.read_csv(csv_path, sep=",")


def load_parquet_data(parquet_path: str, cols: list = None) -> pd.DataFrame:
    """Charge des données tabulaires depuis un fichier Parquet."""
    if cols is not None:
        pd.read_parquet(parquet_path, columns=cols)
    return pd.read_parquet(parquet_path)


def load_geoparquet_data(parquet_path: str) -> gpd.GeoDataFrame:
    """Charge des données tabulaires depuis un fichier GeoParquet."""
    return gpd.read_parquet(parquet_path)


def save_parquet_data(df: pd.DataFrame, path: str):
    """Enregistre un DataFrame au format Parquet."""
    df.to_parquet(path)


def save_geoparquet_data(df: gpd.GeoDataFrame, path: str):
    """Enregistre un GeoDataFrame au format Parquet."""
    df.to_parquet(path)


def list_bdalti_links(pattern: str = None) -> list:
    """
    Récupère la liste des URL .7z pour BD ALTI depuis l'IGN.
    Filtre les URLs contenant `pattern` si fourni (ex: "25M_ASC" ou "75M_ASC").
    """
    resp = requests.get(BDALTI_BASE_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "data.geopf.fr/telechargement/download/BDALTI" in href and href.endswith(".7z"):
            if pattern is None or pattern in href:
                links.append(href)
    links = sorted(set(links))
    print(f"[INFO] {len(links)} fichiers trouvés pour BD ALTI (filtre={pattern})")
    return links


def download_file(url: str, outdir: Path, sleep_between: float = 0.5):
    """
    Télécharge un fichier depuis `url` vers le dossier `outdir`.
    Évite les re-téléchargements en cas de fichier déjà présent.
    """
    filename = Path(urlparse(url).path).name
    out_path = outdir / filename
    if out_path.exists():
        print(f"[SKIP] {filename} déjà présent.")
        return
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    print(f"[DOWNLOAD] {filename}...")
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        outdir.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=2**20):  # 1 Mo
                if chunk:
                    f.write(chunk)
    tmp_path.rename(out_path)
    print(f"[OK] {filename} téléchargé dans {out_path}")
    if sleep_between > 0:
        time.sleep(sleep_between)


def download_bdalti(outdir: str, pattern: str = None, dry_run: bool = False):
    """
    Télécharge tous les fichiers BD ALTI correspondant au `pattern` (ex: "25M_ASC").
    - `outdir` : dossier de destination des fichiers .7z
    - `pattern` : filtre pour sélectionner le MNT 25m ou 75m (par exemple)
    - `dry_run` : si True, ne fait qu'afficher les liens trouvés sans télécharger.
    """
    outdir_path = Path(outdir)
    links = list_bdalti_links(pattern=pattern)
    if dry_run:
        print("[DRY RUN] Fichiers BD ALTI disponibles :")
        for url in links:
            print(f" - {url}")
        return
    for url in links:
        try:
            download_file(url, outdir_path, sleep_between=0.5)
        except Exception as e:
            print(f"[ERREUR] Échec du téléchargement pour {url} : {e}")
            continue


def is_raster_valid(path):
    """
    Vérifie si un fichier raster existe et est lisible.
    Retourne False si le fichier est corrompu ou inexistant.
    """
    if not os.path.exists(path):
        return False

    try:
        with rasterio.open(path) as src:
            # On tente une lecture basique des métadonnées pour être sûr
            _ = src.profile
            return True
    except (RasterioIOError, Exception):
        return False
