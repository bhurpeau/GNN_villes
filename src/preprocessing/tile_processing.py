# tile_processing.py
# Traitement des carreaux (statistiques raster, intersection...)

import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import pickle
from rasterio import features
from rasterio.windows import Window
from tqdm import tqdm


def create_tile_id_raster(
    grid_gdf: gpd.GeoDataFrame,
    reference_raster_path: str,
    out_raster_path: str,
    id_col: str = "id_carr_1km",
) -> dict:
    """
    Rasterisation optimisée avec Index Spatial (R-tree).
    N'utilise que peu de RAM et va beaucoup plus vite.
    """
    # 0. Préparation des données vectorielles
    grid = grid_gdf.reset_index(drop=True)

    print("   -> Construction de l'index spatial...")
    sindex = grid.sindex

    # Mapping ID -> Entier
    unique_ids = sorted(grid[id_col].unique())
    id_map = {val: i + 1 for i, val in enumerate(unique_ids)}
    id_map_inv = {v: k for k, v in id_map.items()}
    grid["raster_val"] = grid[id_col].map(id_map)
    values_array = grid["raster_val"].values
    geoms_array = grid.geometry.values

    # 1. Lecture de la référence
    with rasterio.open(reference_raster_path) as src_ref:
        profile = src_ref.profile.copy()
        height, width = src_ref.height, src_ref.width
        ref_transform = src_ref.transform

    # 2. Config sortie
    profile.update(
        dtype=rasterio.int32,
        count=1,
        nodata=0,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_NEEDED",
    )

    # 3. Écriture intelligente
    print(f"   -> Rasterisation ({width}x{height}) ...")

    with rasterio.open(out_raster_path, "w", **profile) as dst:
        # Liste des fenêtres
        windows_list = list(dst.block_windows(1))

        for _, window in tqdm(windows_list, desc="Rasterisation"):

            # A. Calcul de la boîte englobante de la fenêtre (en coordonnées géographiques)
            win_bounds = rasterio.windows.bounds(window, ref_transform)

            # B. Interrogation de l'index spatial
            candidate_idxs = list(sindex.intersection(win_bounds))
            if not candidate_idxs:
                continue

            # C. Préparation des shapes LOCALES uniquement
            local_geoms = geoms_array[candidate_idxs]
            local_vals = values_array[candidate_idxs]
            local_shapes = list(zip(local_geoms, local_vals))

            # D. Rasterisation locale
            window_transform = rasterio.windows.transform(window, ref_transform)

            img = features.rasterize(
                local_shapes,
                out_shape=(int(window.height), int(window.width)),
                transform=window_transform,
                fill=0,
                default_value=0,
                dtype=rasterio.int32,
            )

            # E. Écriture
            if img.max() > 0:
                dst.write(img, 1, window=window)

    print("✅ Rasterisation terminée.")
    return id_map_inv


def compute_landcover_composition(
    id_raster_path: str, ocs_raster_path: str, id_map_inv: dict
) -> pd.DataFrame:
    """
    Calcule la composition OCS pour chaque carreau.
    Version optimisée : Lecture par grands blocs + Comptage Numpy pur.
    """
    print("Calcul de la composition OCS...")

    # 1. Préparation de la matrice de comptage
    max_tile_id = max(id_map_inv.keys())
    max_class_code = 255

    # Matrice dense : Lignes = Carreaux, Colonnes = Classes OCS
    counts_matrix = np.zeros((max_tile_id + 1, max_class_code + 1), dtype=np.uint32)

    with (
        rasterio.open(id_raster_path) as src_id,
        rasterio.open(ocs_raster_path) as src_ocs,
    ):
        h, w = src_id.height, src_id.width
        # 2. Définition d'une "Grosse Fenêtre" de lecture (2048x2048)
        step = 2048
        windows = []
        for row_off in range(0, h, step):
            height_window = min(step, h - row_off)
            for col_off in range(0, w, step):
                width_window = min(step, w - col_off)
                windows.append(Window(col_off, row_off, width_window, height_window))

        # 3. Boucle de lecture
        for win in tqdm(windows, desc="Analyse OCS"):
            data_id = src_id.read(1, window=win)
            if data_id.max() == 0:
                continue

            data_ocs = src_ocs.read(1, window=win)
            # 4. Comptage vectorisé
            mask = data_id > 0
            valid_ids = data_id[mask]
            valid_classes = data_ocs[mask]
            valid_mask = valid_classes <= max_class_code
            valid_ids = valid_ids[valid_mask]
            valid_classes = valid_classes[valid_mask]
            if len(valid_ids) > 0:
                np.add.at(counts_matrix, (valid_ids, valid_classes), 1)

    # 5. Conversion finale en DataFrame
    print("   -> Conversion en tableau Pandas...")
    present_classes = np.where(counts_matrix.sum(axis=0) > 0)[0]

    # Construction du DF
    final_data = counts_matrix[1:, present_classes]

    # Noms de colonnes
    col_names = [f"part_classe_{c}" for c in present_classes]

    df = pd.DataFrame(final_data, columns=col_names)
    df["id_int"] = np.arange(1, max_tile_id + 1)

    df["id_carr_1km"] = df["id_int"].map(id_map_inv)

    total_pixels = df[col_names].sum(axis=1)
    for col in col_names:
        df[col] = df[col] / total_pixels.replace(0, 1)  # Évite div/0

    # Nettoyage
    return df.drop(columns=["id_int"])


def compute_altitude_stats(
    tile_id_raster: str, dem_raster: str, slope_raster: str, id_map_inv: dict
) -> pd.DataFrame:
    """
    Calcule l'altitude moyenne, l'écart-type et la pente moyenne par carreau.
    Parcourt le MNT (DEM) et le raster de pente par blocs pour agréger les statistiques sur chaque carreau.
    Retourne un DataFrame avec les colonnes: id_carr_1km, z_mean, z_std, slope_mean.
    """
    # Charger le mapping inverse (id raster -> id carreau réel)
    with (
        rasterio.open(tile_id_raster) as src_tile,
        rasterio.open(dem_raster) as src_dem,
        rasterio.open(slope_raster) as src_slope,
    ):
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
                mask &= dem_flat != dem_nodata
            if slope_nodata is not None:
                mask &= slope_flat != slope_nodata
            if not np.any(mask):
                continue
            t = tile_flat[mask].astype(int)
            z = dem_flat[mask].astype(np.float64)
            s = slope_flat[mask].astype(np.float64)
            # Agrégation par carreau via np.bincount
            count += np.bincount(t, minlength=N + 1)
            sum_z += np.bincount(t, weights=z, minlength=N + 1)
            sum_z2 += np.bincount(t, weights=z * z, minlength=N + 1)
            sum_slope += np.bincount(t, weights=s, minlength=N + 1)

    # Calcul des statistiques pour chaque carreau (en excluant l'index 0 qui correspond à "aucun carreau")
    idx = np.arange(1, N + 1)
    valid = count[idx] > 0
    id_int = idx[valid]
    n = count[id_int]
    mean_z = sum_z[id_int] / n
    var_z = (sum_z2[id_int] / n) - (mean_z**2)
    var_z[var_z < 0] = 0  # évite de petites valeurs négatives dues aux arrondis
    std_z = np.sqrt(var_z)
    mean_slope = sum_slope[id_int] / n

    # Créer le DataFrame des stats altimétriques
    original_ids = [
        id_map_inv[i] for i in id_int
    ]  # remapper indices internes vers ID carreau original
    df_alt = pd.DataFrame(
        {
            "id_carr_1km": original_ids,
            "z_mean": mean_z,
            "z_std": std_z,
            "slope_mean": mean_slope,
        }
    )
    return df_alt


def assign_tiles_to_communes(
    grid_gdf: gpd.GeoDataFrame,
    communes_gdf: gpd.GeoDataFrame,
    id_col: str = "id_carr_1km",
    commune_code_col: str = "code",
    nearest_distance: float = 2000.0,
) -> pd.Series:
    """
    Attribue chaque carreau 1km à une commune.
    - Utilise une intersection spatiale pour déterminer la commune couvrant la plus grande partie du carreau.
    - Pour les carreaux non assignés (orphelins), utilise la commune la plus proche (si distance < nearest_distance).
    Retourne une Series indexée par id_carr_1km donnant le code commune assigné.
    """
    # Intersection carreaux/communes pour trouver les chevauchements
    pieces = gpd.overlay(grid_gdf, communes_gdf[[commune_code_col, "geometry"]], how="intersection")
    pieces["area"] = pieces.geometry.area
    # Trier par aire décroissante pour chaque carreau
    pieces = pieces.sort_values(["id_carr_1km", "area"], ascending=[True, False])
    # Garder la commune principale par carreau (la plus grande surface)
    main_assignments = pieces.drop_duplicates(subset=[id_col], keep="first")
    mapping = pd.Series(
        main_assignments[commune_code_col].values, index=main_assignments[id_col].values
    )
    # Identifier les carreaux sans commune (pas d'intersection)
    all_tiles = set(grid_gdf[id_col])
    assigned_tiles = set(mapping.index)
    orphans = all_tiles - assigned_tiles
    if orphans:
        orphans_gdf = grid_gdf[grid_gdf[id_col].isin(orphans)]
        # Trouver pour chaque carreau orphelin la commune la plus proche (jointure spatiale nearest)
        nearest = gpd.sjoin_nearest(
            orphans_gdf,
            communes_gdf[[commune_code_col, "geometry"]],
            how="left",
            distance_col="dist_to_commune",
        )
        # Filtrer les attributions trop lointaines
        nearest_valid = nearest[nearest["dist_to_commune"] < nearest_distance]
        for _, row in nearest_valid.iterrows():
            mapping[row[id_col]] = row[commune_code_col]
    return mapping
