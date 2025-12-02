#!/usr/bin/env python3
import rasterio
import numpy as np
import pickle
import pandas as pd

DEM = "./BDALTI_25m/bdalti25m.tif"
SLOPE = "./BDALTI_25m/bdalti25m_slope_deg.tif"      # ou _slope_pct.tif si tu as mis -p
TILE_ID_RASTER = "./data/id_carr_1km_25m.tif"
IDMAP_PKL = "./data/id_carr_1km_idmap.pkl"

OUT_CSV = "./data/stats_altitude_slope_1km.csv"

def main():
    # 1. Charger le mapping inverse int -> id_carr_1km
    with open(IDMAP_PKL, "rb") as f:
        id_map_inv = pickle.load(f)

    N = max(id_map_inv.keys())  # IDs internes = 1..N
    print(f"N carreaux = {N}")

    # 2. Ouvrir les rasters
    with rasterio.open(DEM) as src_dem, \
         rasterio.open(SLOPE) as src_slope, \
         rasterio.open(TILE_ID_RASTER) as src_tile:

        assert src_dem.width == src_tile.width == src_slope.width
        assert src_dem.height == src_tile.height == src_slope.height

        dem_nodata = src_dem.nodata
        slope_nodata = src_slope.nodata

        # 3. Accumulateurs par carreau (index 0 = vide, 1..N = carreaux)
        count = np.zeros(N+1, dtype=np.int64)
        sum_z = np.zeros(N+1, dtype=np.float64)
        sum_z2 = np.zeros(N+1, dtype=np.float64)
        sum_slope = np.zeros(N+1, dtype=np.float64)

        # 4. Parcours par blocs
        for ji, window in src_dem.block_windows(1):
            dem_block = src_dem.read(1, window=window)
            slope_block = src_slope.read(1, window=window)
            tile_block = src_tile.read(1, window=window)

            dem_flat = dem_block.ravel()
            slope_flat = slope_block.ravel()
            tile_flat = tile_block.ravel()

            # Masque : pixels appartenant à un carreau (id > 0)
            # et valides pour DEM & pente
            mask = (tile_flat > 0)
            if dem_nodata is not None:
                mask &= (dem_flat != dem_nodata)
            if slope_nodata is not None:
                mask &= (slope_flat != slope_nodata)

            if not np.any(mask):
                continue

            t = tile_flat[mask]
            z = dem_flat[mask].astype(np.float64)
            s = slope_flat[mask].astype(np.float64)

            # np.bincount sur 1..N (index 0 = ignoré)
            # Nombre de pixels par carreau dans ce bloc
            bc_count = np.bincount(t, minlength=N+1)
            bc_sum_z = np.bincount(t, weights=z, minlength=N+1)
            bc_sum_z2 = np.bincount(t, weights=z*z, minlength=N+1)
            bc_sum_slope = np.bincount(t, weights=s, minlength=N+1)

            count += bc_count
            sum_z += bc_sum_z
            sum_z2 += bc_sum_z2
            sum_slope += bc_sum_slope

    # 5. Calcul des stats
    # On ignore l'index 0
    idx = np.arange(1, N+1)
    valid = count[idx] > 0

    id_int = idx[valid]
    n = count[id_int]
    mean_z = sum_z[id_int] / n

    var_z = sum_z2[id_int] / n - mean_z**2
    var_z = np.clip(var_z, 0, None)
    std_z = np.sqrt(var_z)

    mean_slope = sum_slope[id_int] / n

    # 6. Remapping vers id_carr_1km d'origine
    id_carr_1km = [id_map_inv[i] for i in id_int]

    df = pd.DataFrame({
        "id_carr_1km": id_carr_1km,
        "z_mean": mean_z,
        "z_std": std_z,
        "slope_mean": mean_slope,
        "n_pixels": n
    })

    df.to_csv(OUT_CSV, index=False)
    print(f"Stats écrites dans {OUT_CSV}")

if __name__ == "__main__":
    main()
