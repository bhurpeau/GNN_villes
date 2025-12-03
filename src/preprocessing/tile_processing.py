# tile_processing.py
# Traitement des carreaux (statistiques raster, intersection...)

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import pandas as pd


def create_tile_id_raster(grid_gdf: gpd.GeoDataFrame, reference_raster_path: str, 
                          out_raster_path: str, id_col: str = "id_carr_1km") -> dict:
    """
    Rasterise la grille de carreaux 1km en utilisant un raster de référence pour l'emprise et la résolution.
    Chaque pixel du raster de sortie contient l'ID du carreau correspondant (0 si aucun).
    Retourne un mapping inverse {id_int: id_carreau_original}.
    """
    # Ouvrir le raster de référence pour obtenir dimensions, transform et CRS
    with rasterio.open(reference_raster_path) as src_ref:
        ref_crs = src_ref.crs
        ref_transform = src_ref.transform
        height, width = src_ref.height, src_ref.width
        profile = src_ref.profile

    # Reprojection de la grille si nécessaire
    if grid_gdf.crs != ref_crs:
        grid = grid_gdf.to_crs(ref_crs)
    else:
        grid = grid_gdf

    # Construire un ID entier compact pour chaque carreau (pour rasterize)
    unique_ids = sorted(grid[id_col].unique())
    id_map = {val: i + 1 for i, val in enumerate(unique_ids)}   # 0 sera utilisé pour "aucun carreau"
    id_map_inv = {v: k for k, v in id_map.items()}

    # Préparer les tuples (geometry, value) pour chaque carreau
    shapes = [(geom, id_map[val]) for geom, val in zip(grid.geometry, grid[id_col])]
    # Rasterisation de la grille
    tile_id_array = rasterize(shapes=shapes, out_shape=(height, width), transform=ref_transform,
                              fill=0, dtype="int32")
    # Mettre à jour le profil raster pour le fichier de sortie
    profile.update({
        "dtype": "int32",
        "count": 1,
        "nodata": 0,
        "compress": "LZW",
        "tiled": True,
        "BIGTIFF": "YES"
    })
    # Enregistrer le raster d'identifiants de carreaux
    with rasterio.open(out_raster_path, "w", **profile) as dst:
        dst.write(tile_id_array, 1)
    return id_map_inv


def compute_landcover_composition(tile_id_raster: str, landcover_raster: str, id_map_inv: dict) -> pd.DataFrame:
    """
    Calcule la composition d'occupation du sol de chaque carreau.
    Parcourt le raster d'occupation du sol et compte, pour chaque carreau (pixel du raster d'ID),
    la distribution des classes d'occupation du sol.
    Retourne un DataFrame avec une ligne par carreau et des colonnes part_classe_X pour chaque classe.
    """
    counts = {}  # (id_int, classe) -> nombre de pixels
    with rasterio.open(landcover_raster) as src_lc, rasterio.open(tile_id_raster) as src_tiles:
        assert src_lc.width == src_tiles.width and src_lc.height == src_tiles.height
        lc_nodata = src_lc.nodata
        for _, window in src_lc.block_windows(1):
            lc_block = src_lc.read(1, window=window)
            tile_block = src_tiles.read(1, window=window)
            # Aplatir les blocs en 1D
            lc_flat = lc_block.ravel()
            tile_flat = tile_block.ravel()
            # Masque: pixels appartenant à un carreau (id > 0) et ayant une classe valide
            mask = tile_flat > 0
            if lc_nodata is not None:
                mask &= (lc_flat != lc_nodata)
            if not np.any(mask):
                continue
            tile_vals = tile_flat[mask]
            class_vals = lc_flat[mask]
            # Comptage des paires (tile_id_int, classe) sur ce bloc
            pairs = np.stack([tile_vals, class_vals], axis=1)
            uniq_pairs, counts_pairs = np.unique(pairs, axis=0, return_counts=True)
            for (tile_int, classe), cnt in zip(uniq_pairs, counts_pairs):
                counts[(int(tile_int), int(classe))] = counts.get((int(tile_int), int(classe)), 0) + int(cnt)
    # Construire le DataFrame long
    data_rows = []
    for (tile_int, classe), nb in counts.items():
        data_rows.append({
            "id_carr_1km_int": tile_int,
            "classe": classe,
            "nb_pixels": nb
        })
    df_long = pd.DataFrame(data_rows)
    # Ajouter l'ID carreau d'origine via le mapping inverse
    df_long["id_carr_1km"] = df_long["id_carr_1km_int"].map(id_map_inv)
    # Calcul de la surface en m² couverte par chaque classe dans le carreau
    with rasterio.open(landcover_raster) as src_lc:
        px_width, px_height = src_lc.res  # résolution du pixel en unités de la projection
    df_long["surface_m2"] = df_long["nb_pixels"] * abs(px_width * px_height)
    # Calcul du pourcentage de pixels par classe dans le carreau
    df_long["total_pixels"] = df_long.groupby("id_carr_1km")["nb_pixels"].transform("sum")
    df_long["part_pixels"] = df_long["nb_pixels"] / df_long["total_pixels"]
    # Pivot en format large: une colonne par classe (part_classe_X)
    df_wide = df_long.pivot(index="id_carr_1km", columns="classe", values="part_pixels").fillna(0)
    df_wide.columns = [f"part_classe_{int(c)}" for c in df_wide.columns]
    df_wide.reset_index(inplace=True)
    return df_wide


def compute_altitude_stats(tile_id_raster: str, dem_raster: str, slope_raster: str, id_map_inv: dict) -> pd.DataFrame:
    """
    Calcule l'altitude moyenne, l'écart-type et la pente moyenne par carreau.
    Parcourt le MNT (DEM) et le raster de pente par blocs pour agréger les statistiques sur chaque carreau.
    Retourne un DataFrame avec les colonnes: id_carr_1km, z_mean, z_std, slope_mean.
    """
    # Charger le mapping inverse (id raster -> id carreau réel)
    with rasterio.open(tile_id_raster) as src_tile, \
         rasterio.open(dem_raster) as src_dem, \
         rasterio.open(slope_raster) as src_slope:
        assert src_dem.width == src_slope.width == src_tile.width
        assert src_dem.height == src_slope.height == src_tile.height
        dem_nodata = src_dem.nodata
        slope_nodata = src_slope.nodata

        # Préparation des accumulateurs (index 0 ignoré, indices 1..N pour carreaux)
        N = int(src_tile.read(1).max())  # nombre total de carreaux (valeur max dans le raster ID)
        count = np.zeros(N + 1, dtype=np.int64)
        sum_z = np.zeros(N + 1, dtype=np.float64)
        sum_z2 = np.zeros(N + 1, dtype=np.float64)
        sum_slope = np.zeros(N + 1, dtype=np.float64)

        # Parcours des rasters par fenêtre (pour gérer de grands rasters sans tout charger en RAM)
        for _, window in src_dem.block_windows(1):
            dem_block = src_dem.read(1, window=window)
            slope_block = src_slope.read(1, window=window)
            tile_block = src_tile.read(1, window=window)
            dem_flat = dem_block.ravel()
            slope_flat = slope_block.ravel()
            tile_flat = tile_block.ravel()
            # Masque: pixels appartenant à un carreau (id > 0) et valides pour DEM et pente
            mask = tile_flat > 0
            if dem_nodata is not None:
                mask &= (dem_flat != dem_nodata)
            if slope_nodata is not None:
                mask &= (slope_flat != slope_nodata)
            if not np.any(mask):
                continue
            t = tile_flat[mask].astype(int)
            z = dem_flat[mask].astype(np.float64)
            s = slope_flat[mask].astype(np.float64)
            # Agrégation par carreau via np.bincount
            count += np.bincount(t, minlength=N+1)
            sum_z += np.bincount(t, weights=z, minlength=N+1)
            sum_z2 += np.bincount(t, weights=z * z, minlength=N+1)
            sum_slope += np.bincount(t, weights=s, minlength=N+1)

    # Calcul des statistiques pour chaque carreau (en excluant l'index 0 qui correspond à "aucun carreau")
    idx = np.arange(1, N+1)
    valid = count[idx] > 0
    id_int = idx[valid]
    n = count[id_int]
    mean_z = sum_z[id_int] / n
    var_z = (sum_z2[id_int] / n) - (mean_z ** 2)
    var_z[var_z < 0] = 0  # évite de petites valeurs négatives dues aux arrondis
    std_z = np.sqrt(var_z)
    mean_slope = sum_slope[id_int] / n

    # Créer le DataFrame des stats altimétriques
    original_ids = [id_map_inv[i] for i in id_int]  # remapper indices internes vers ID carreau original
    df_alt = pd.DataFrame({
        "id_carr_1km": original_ids,
        "z_mean": mean_z,
        "z_std": std_z,
        "slope_mean": mean_slope
    })
    return df_alt


def assign_tiles_to_communes(grid_gdf: gpd.GeoDataFrame, communes_gdf: gpd.GeoDataFrame, 
                             id_col: str = "id_carr_1km", commune_code_col: str = "code",
                             nearest_distance: float = 2000.0) -> pd.Series:
    """
    Attribue chaque carreau 1km à une commune.
    - Utilise une intersection spatiale pour déterminer la commune couvrant la plus grande partie du carreau.
    - Pour les carreaux non assignés (orphelins), utilise la commune la plus proche (si distance < nearest_distance).
    Retourne une Series indexée par id_carr_1km donnant le code commune assigné.
    """
    # Intersection carreaux/communes pour trouver les chevauchements
    pieces = gpd.overlay(grid_gdf, communes_gdf[[commune_code_col, 'geometry']], how='intersection')
    pieces['area'] = pieces.geometry.area
    # Trier par aire décroissante pour chaque carreau
    pieces = pieces.sort_values(['id_carr_1km', 'area'], ascending=[True, False])
    # Garder la commune principale par carreau (la plus grande surface)
    main_assignments = pieces.drop_duplicates(subset=[id_col], keep='first')
    mapping = pd.Series(main_assignments[commune_code_col].values, index=main_assignments[id_col].values)
    # Identifier les carreaux sans commune (pas d'intersection)
    all_tiles = set(grid_gdf[id_col])
    assigned_tiles = set(mapping.index)
    orphans = all_tiles - assigned_tiles
    if orphans:
        orphans_gdf = grid_gdf[grid_gdf[id_col].isin(orphans)]
        # Trouver pour chaque carreau orphelin la commune la plus proche (jointure spatiale nearest)
        nearest = gpd.sjoin_nearest(orphans_gdf, communes_gdf[[commune_code_col, 'geometry']], how='left', distance_col='dist_to_commune')
        # Filtrer les attributions trop lointaines
        nearest_valid = nearest[nearest['dist_to_commune'] < nearest_distance]
        for _, row in nearest_valid.iterrows():
            mapping[row[id_col]] = row[commune_code_col]
    return mapping
