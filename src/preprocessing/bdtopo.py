# bdtopo.py

import os
import re
import glob
import shutil
import subprocess
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BDTOPO_PAGE = "https://geoservices.ign.fr/bdtopo"
DOWNLOAD_DIR = "downloads"
UNZIP_DIR = "unzipped"
OUT_GPKG = "./data/bdtopo_routes.gpkg"
LAYER_OUT = "troncon_de_route"
DATE_FILTER = "2021-03"


def find_7z_links(date):
    resp = requests.get(BDTOPO_PAGE)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Motif général
    pattern = re.compile(r"BDTOPO_\d-\d_TOUSTHEMES_SHP_LAMB93_D\d{3}_\d{4}-\d{2}-\d{2}\.7z$")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # on filtre sur TOUSTHEMES + LAMB93 + date + extension
        if (
            "BDTOPO" in href
            and "TOUSTHEMES_SHP_LAMB93_D" in href
            and date in href
            and href.endswith(".7z")
        ):
            if pattern.search(href):
                if href.startswith("http"):
                    url = href
                else:
                    url = requests.compat.urljoin(BDTOPO_PAGE, href)
                links.add(url)

    links = sorted(links)
    return links


def download_file(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filename = url.split("/")[-1]
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        print(f"[SKIP] {filename} existe déjà")
        return out_path

    print(f"[DL] {filename}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(out_path, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=filename,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return out_path


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def unzip_one_archive(archive_path):
    """
    Décompresse une archive .7z dans UNZIP_DIR/<nom_sans_ext>/ et renvoie ce dossier.
    """
    os.makedirs(UNZIP_DIR, exist_ok=True)
    archive_path = Path(archive_path)
    dept_name = archive_path.stem  # ex: BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_D001_2021-03-15
    target_dir = Path(UNZIP_DIR) / dept_name

    if target_dir.exists():
        # On repart sur du propre
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["7z", "x", str(archive_path), f"-o{target_dir}"]
    run(cmd)

    return target_dir


def find_troncon_shapefiles_in_dir(root_dir):
    """
    Cherche les TRONCON_DE_ROUTE.shp sous root_dir.
    """
    pattern = os.path.join(str(root_dir), "**", "TRONCON_DE_ROUTE.shp")
    files = glob.glob(pattern, recursive=True)
    files = sorted(files)
    print(f"{len(files)} fichiers TRONCON_DE_ROUTE.shp trouvés dans {root_dir}.")
    return files


def merge_shapefiles_to_gpkg(shapefiles):
    """
    Fusionne les shapefiles TRONCON_DE_ROUTE dans OUT_GPKG
    en filtrant :
      FICTIF = 'Non'
      ETAT   = 'En service'
    et en imposant une géométrie 2D (LINESTRING).
    """
    if not shapefiles:
        return

    out_dir = os.path.dirname(OUT_GPKG)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    first_create = not os.path.exists(OUT_GPKG)

    # Requête OGR SQL pour filtrer les champs attributaires
    sql = "SELECT * FROM TRONCON_DE_ROUTE WHERE FICTIF = 'Non' AND ETAT = 'En service'"

    for shp in shapefiles:
        if first_create:
            # Création du GPKG
            cmd = [
                "ogr2ogr",
                "-f",
                "GPKG",
                OUT_GPKG,
                shp,
                "-nln",
                LAYER_OUT,
                "-sql",
                sql,
                "-t_srs",
                "EPSG:2154",
                "-nlt",
                "LINESTRING",  # aplati en 2D
                "-dim",
                "2",
                "-progress",
            ]
            first_create = False
        else:
            # Append
            cmd = [
                "ogr2ogr",
                "-f",
                "GPKG",
                "-append",
                OUT_GPKG,
                shp,
                "-nln",
                LAYER_OUT,
                "-sql",
                sql,
                "-t_srs",
                "EPSG:2154",
                "-nlt",
                "LINESTRING",
                "-dim",
                "2",
                "-progress",
            ]
        run(cmd)


def main():
    links = find_7z_links(DATE_FILTER)
    print(f"{len(links)} archives BDTOPO Tous Thèmes trouvées pour {DATE_FILTER}.")

    for url in links:
        # 1) Téléchargement
        archive_path = download_file(url, DOWNLOAD_DIR)

        # 2) Décompression de cette archive uniquement
        dept_dir = unzip_one_archive(archive_path)

        # 3) Recherche des TRONCON_DE_ROUTE.shp
        shp_files = find_troncon_shapefiles_in_dir(dept_dir)
        if not shp_files:
            print(f"[WARN] Aucun TRONCON_DE_ROUTE.shp trouvé dans {dept_dir}.")
        else:
            # 4) Fusion dans le GPKG national
            merge_shapefiles_to_gpkg(shp_files)

        # 5) Nettoyage : dossier décompressé + archive pour économiser le disque
        shutil.rmtree(dept_dir)
        os.remove(archive_path)
        print(f"[CLEAN] Supprimé : {dept_dir} et {archive_path}")
    cmd = ["rm", "-r", DOWNLOAD_DIR]
    run(cmd)
    cmd = ["rm", "-r", UNZIP_DIR]
    run(cmd)
    print(f"Terminé. GPKG final : {OUT_GPKG}")


if __name__ == "__main__":
    main()
