#!/usr/bin/env python3
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import pickle
from pathlib import Path

GRID_1KM = "./data/grille1km_metropole.gpkg"
ID_COL = "id_carr_1km"

DEM = "./BDALTI_25m/bdalti25m.tif"
TILE_ID_RASTER = "./data/id_carr_1km_25m.tif"
IDMAP_PKL = "./data/id_carr_1km_idmap.pkl"  # pour garder le mapping original

def main():
    # 1. Charger la grille
    g1km = gpd.read_file(GRID_1KM)

    # 2. Ouvrir le MNT pour récupérer la grille raster
    with rasterio.open(DEM) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        dem_height = src.height
        dem_width = src.width
        profile = src.profile

    # 3. Reprojeter la grille si besoin
    if g1km.crs != dem_crs:
        g1km = g1km.to_crs(dem_crs)

    # 4. Construire un id compact pour chaque carreau
    unique_ids = g1km[ID_COL].unique()
    unique_ids = np.sort(unique_ids.astype(str))  # tout en str

    id_map = {val: i+1 for i, val in enumerate(unique_ids)} 
    id_map_inv = {v: k for k, v in id_map.items()}          
    N = len(unique_ids)
    print(f"{N} carreaux 1km uniques.")

    # 5. Préparer les shapes pour rasterize
    shapes = [
        (geom, id_map[str(id_val)])
        for geom, id_val in zip(g1km.geometry, g1km[ID_COL])
    ]

    # 6. Rasterisation
    tile_id_array = rasterize(
        shapes=shapes,
        out_shape=(dem_height, dem_width),
        transform=dem_transform,
        fill=0,
        dtype="int32"
    )

    # 7. Écriture du raster d'IDs
    profile.update({
        "dtype": "int32",
        "count": 1,
        "nodata": 0,
        "compress": "LZW",
        "tiled": True,
        "BIGTIFF": "YES",
    })

    with rasterio.open(TILE_ID_RASTER, "w", **profile) as dst:
        dst.write(tile_id_array, 1)

    # 8. Sauvegarder le mapping pour usage ultérieur
    with open(IDMAP_PKL, "wb") as f:
        pickle.dump(id_map_inv, f)

    print(f"Raster ID écrit dans {TILE_ID_RASTER}")
    print(f"Mapping inverse ID sauvegardé dans {IDMAP_PKL}")

if __name__ == "__main__":
    main()
