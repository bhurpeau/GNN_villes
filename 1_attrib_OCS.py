import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from collections import defaultdict, Counter

GRID_1KM = "data/grille1km_metropole.gpkg"
RASTER_OCS = "data/OCS_2023.tif"
RASTER_TILE_ID = "data/grille1km_ID_10m.tif" 
ID_1KM_COL = "id_carr_1km"

def main():
    # 1. Charger la grille 1 km
    g1km = gpd.read_file(GRID_1KM)
    
    # 2. Ouvrir le raster OCS pour récupérer shape/transform/CRS
    with rasterio.open(RASTER_OCS) as src:
        ocs_crs = src.crs
        ocs_transform = src.transform
        ocs_height = src.height
        ocs_width = src.width
        ocs_profile = src.profile
    
    # 3. Reprojeter la grille si besoin
    if g1km.crs != ocs_crs:
        g1km = g1km.to_crs(ocs_crs)
    
    # 4. Pour rasteriser, on a besoin d'IDs entiers compacts
    #    (optionnel mais plus propre : on remappe ID_1KM -> [1..N])
    unique_ids = g1km[ID_1KM_COL].unique()
    id_map = {val: i+1 for i, val in enumerate(unique_ids)}  # 0 = "aucun carreau"
    id_map_inv = {v: k for k, v in id_map.items()}
    
    # 5. Préparer les tuples (geom, value) pour rasterize
    shapes = [
        (geom, id_map[id_val])
        for geom, id_val in zip(g1km.geometry, g1km[ID_1KM_COL])
    ]
    
    # 6. Rasterisation de la grille 1km à 10m
    tile_id_array = rasterize(
        shapes=shapes,
        out_shape=(ocs_height, ocs_width),
        transform=ocs_transform,
        fill=0,         # 0 = aucun carreau
        dtype="int32"
    )
    
    # 7. Sauvegarder ce raster d'ID
    tile_profile = ocs_profile.copy()
    tile_profile.update({
        "dtype": "int32",
        "count": 1,
        "nodata": 0,
        "driver": "GTiff",
        "BIGTIFF": "YES",      
        "TILED": True,         
        "BLOCKXSIZE": 512,     
        "BLOCKYSIZE": 512,
        "COMPRESS": "LZW",     
    })
    
    with rasterio.open(RASTER_TILE_ID, "w", **tile_profile) as dst:
        dst.write(tile_id_array, 1)
    # Dictionnaire : (tile_id_int, classe) -> count
    counts = defaultdict(int)
    
    with rasterio.open(RASTER_OCS) as src_ocs, rasterio.open(RASTER_TILE_ID) as src_tiles:
        assert src_ocs.width == src_tiles.width
        assert src_ocs.height == src_tiles.height
    
        ocs_nodata = src_ocs.nodata
    
        # On parcourt le raster par blocs internes
        for ji, window in src_ocs.block_windows(1):
            # Lire la fenêtre OCS et la fenêtre TileID
            ocs_block = src_ocs.read(1, window=window)
            tile_block = src_tiles.read(1, window=window)
    
            # Aplatir
            ocs_flat = ocs_block.ravel()
            tile_flat = tile_block.ravel()
    
            # Masque : garder les pixels avec un tile_id > 0 et non nodata
            if ocs_nodata is not None:
                mask = (tile_flat > 0) & (ocs_flat != ocs_nodata)
            else:
                mask = (tile_flat > 0)
    
            if not np.any(mask):
                continue
    
            tile_vals = tile_flat[mask]
            class_vals = ocs_flat[mask]
    
            # On utilise np.unique sur les couples (tile, classe) pour cette fenêtre
            pairs = np.stack([tile_vals, class_vals], axis=1)
            uniq_pairs, uniq_counts = np.unique(pairs, axis=0, return_counts=True)
    
            # Accumuler dans le dict global
            for (tile_id_int, classe), c in zip(uniq_pairs, uniq_counts):
                counts[(int(tile_id_int), int(classe))] += int(c)
    rows = []
    for (tile_id_int, classe), nb_pixels in counts.items():
        rows.append({
            "ID_1KM_INT": tile_id_int,
            "classe": classe,
            "nb_pixels": nb_pixels
        })
    
    df_counts = pd.DataFrame(rows)
    
    
    # Remapper ID_1KM_INT -> vrai ID_1KM via id_map_inv (construit plus haut)
    df_counts["ID_1KM"] = df_counts["ID_1KM_INT"].map(id_map_inv)
    
    # Optionnel : on peut se débarrasser de la colonne intermédiaire
    df_counts = df_counts[["ID_1KM", "classe", "nb_pixels"]]
    
    
    with rasterio.open(RASTER_OCS) as src:
        px_w, px_h = src.res
        px_area = abs(px_w * px_h)
    
    df_counts["surface"] = df_counts["nb_pixels"] * px_area
    
    # Calcul des parts par carreau 1 km
    df_counts["total_pixels"] = df_counts.groupby("ID_1KM")["nb_pixels"].transform("sum")
    df_counts["part_pixels"] = df_counts["nb_pixels"] / df_counts["total_pixels"]
    
    df_pivot = df_counts.pivot(
        index="ID_1KM",
        columns="classe",
        values="part_pixels"
    ).fillna(0)
    
    df_pivot.columns = [f"part_classe_{c}" for c in df_pivot.columns]
    
    df_counts.to_parquet("data/ocs_par_classe_par_carreau_1km_long.parquet")
    df_pivot.to_parquet("data/ocs_par_classe_par_carreau_1km_large.parquet")


if __name__ == "__main__":
    main()