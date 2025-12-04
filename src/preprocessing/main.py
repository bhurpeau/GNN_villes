# main.py
# Pipeline principal CLI pour préparer les graphes

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import os
from .data_io import run, load_grid_shapefile, load_communes_shapefile, load_csv_data, load_parquet_data, save_geoparquet_data
from .tile_processing import create_tile_id_raster, compute_landcover_composition, compute_altitude_stats, assign_tiles_to_communes
from .graph_utils import build_micro_intra_edges, build_macro_physical_graph
from .cities_processing import process_macro_flows

# Définition des chemins de fichiers en entrée (à adapter si besoin)
GRID_PATH = "data/grille1km_metropole.gpkg"
COMMUNES_PATH = "data/commune_francemetro_2023.gpkg"
ROUTE_PATH = "data/bdtopo_routes.gpkg"
RASTER_OCS_PATH = "data/OCS_2023.tif"
RASTER_DEM_PATH = "BDALTI/bdalti25m.tif"
RASTER_SLOPE_PATH = "BDALTI/bdalti25m_slope_deg.tif"
MiGRATIONS = "./data/DAD21.parquet"
TRAVAIL = "./data/DT21.parquet"
POPULATION = "./data/pop21.csv"
ACTIVITE = './data/acti.csv'

# Chemins de sortie (parquet et numpy)
OUT_TILE_FEATURES = "data_GNN/statistiques_carreaux.parquet.gz"
OUT_EDGES_ALL = "data_GNN/edges_toutes_communes.npy"
raster_output = "data/grille1km_id_10m.tif"
raster_output_25 = "data/id_carr_1km_25m.tif"
# IDMAP_PKL = "data/grille1km_id_10m_idmap.pkl"
# IDMAP_PKL_25 = "data/grille1km_id_25m_idmap.pkl"


def main():
    # 1. Charger les données de base
    grid = load_grid_shapefile(GRID_PATH)            # Grille 1km (GeoDataFrame)
    communes = load_communes_shapefile(COMMUNES_PATH)  # Communes (GeoDataFrame)
    try:
        pop_df = load_parquet_data("data/grid_1km.parquet")  # population par carreau (INSEE grille 1km)
    except FileNotFoundError:
        pop_df = None
    try:
        socio_df = load_csv_data("data/filo_2019_carreaux_1km_met.csv")  # données Filosofi 2019 par carreau
    except FileNotFoundError:
        socio_df = None

    # 2. Rasterisation de la grille aux résolutions nécessaires et calcul des features par carreau
    # Raster ID aligné sur l'OCS (10m) pour l'occupation du sol
    if os.path.exists(raster_output):
        os.remove(raster_output)
    print(f"⚙️ Génération de {raster_output}...")
    id_map_inv_10m = create_tile_id_raster(grid, RASTER_OCS_PATH, raster_output, id_col="id_carr_1km")
    ocs_df = compute_landcover_composition(raster_output, RASTER_OCS_PATH, id_map_inv_10m)
    # Raster ID aligné sur le MNT (25m) pour altitude/pente
    if os.path.exists(raster_output_25):
        os.remove(raster_output_25)
    print(f"⚙️ Génération de {raster_output_25}...")
    id_map_inv_25m = create_tile_id_raster(grid, RASTER_DEM_PATH, raster_output_25, id_col="id_carr_1km")
    if os.path.exists(RASTER_SLOPE_PATH):
        os.remove(RASTER_SLOPE_PATH)
    cmd = [
            'gdaldem',
            'slope',
            RASTER_DEM_PATH,
            RASTER_SLOPE_PATH,
            '-compute_edges'
            ]
    run(cmd)
    alt_df = compute_altitude_stats(raster_output_25, RASTER_DEM_PATH, RASTER_SLOPE_PATH, id_map_inv_25m)
    # 3. Fusion des features dans un seul DataFrame
    stats_df = pd.merge(alt_df, ocs_df, on="id_carr_1km", how="left")
    # Ajouter population et socio-éco si disponibles
    if pop_df is not None:
        stats_df = pd.merge(stats_df, pop_df, left_on="id_carr_1km", right_on="GRD_ID", how="left")
    if socio_df is not None:
        stats_df = pd.merge(stats_df, socio_df, left_on="id_carr_1km", right_on="idcar_1km", how="left")
    # 4. Ajouter la géométrie des carreaux et assigner les communes
    # Merge avec la grille pour récupérer la géométrie
    stats_gdf = gpd.GeoDataFrame(pd.merge(stats_df, grid[['id_carr_1km', 'geometry']], on="id_carr_1km", how="left"), crs=grid.crs)
    # Attribution des communes
    mapping_series = assign_tiles_to_communes(grid, communes)
    stats_gdf['code'] = stats_gdf['id_carr_1km'].map(mapping_series.to_dict())
    # 5. Calcul de features additionnelles (structures OCS et socio-éco)
    # Regroupements de classes OCS (structure du territoire)
    stats_gdf['struct_bati'] = stats_gdf.get('part_classe_1', 0) + stats_gdf.get('part_classe_2', 0)
    stats_gdf['struct_eco'] = stats_gdf.get('part_classe_3', 0) + stats_gdf.get('part_classe_4', 0)
    # Somme des classes nature 11 à 19
    nature_cols = [f"part_classe_{c}" for c in range(11, 20)]
    stats_gdf['struct_nature'] = stats_gdf[nature_cols].sum(axis=1) if nature_cols[0] in stats_gdf else 0
    # Somme des classes agriculture 20 à 22
    agri_cols = [f"part_classe_{c}" for c in range(20, 23)]
    stats_gdf['struct_agri'] = stats_gdf[agri_cols].sum(axis=1) if agri_cols[0] in stats_gdf else 0
    # Eau et glacier
    stats_gdf['struct_eau'] = stats_gdf.get('part_classe_23', 0)
    stats_gdf['struct_glacier'] = stats_gdf.get('part_classe_24', 0)
    # Indicateurs socio-démographiques (si données disponibles)
    if 'TOT_P_2021' in stats_gdf:
        stats_gdf['densite_pop'] = stats_gdf['TOT_P_2021']  # par km² (carreaux de 1 km²)
        if 'TOT_P_2011' in stats_gdf:
            # Croissance démographique 2011-2021
            stats_gdf['croissance_pop'] = (stats_gdf['TOT_P_2021'] - stats_gdf['TOT_P_2011']) / stats_gdf['TOT_P_2011'].replace({0: np.nan})
            stats_gdf['croissance_pop'] = stats_gdf['croissance_pop'].fillna(0)
    if 'ind_snv' in stats_gdf and 'ind' in stats_gdf:
        stats_gdf['niveau_vie_moyen'] = (stats_gdf['ind_snv'] / stats_gdf['ind'].replace({0: np.nan})).fillna(0)
    if 'men_pauv' in stats_gdf and 'men' in stats_gdf:
        stats_gdf['taux_pauvrete'] = (stats_gdf['men_pauv'] / stats_gdf['men'].replace({0: np.nan})).fillna(0)
    if 'men_prop' in stats_gdf and 'men' in stats_gdf:
        stats_gdf['part_proprio'] = (stats_gdf['men_prop'] / stats_gdf['men'].replace({0: np.nan})).fillna(0)
    if 'men_mais' in stats_gdf and 'men_coll' in stats_gdf:
        stats_gdf['part_maison'] = (stats_gdf['men_mais'] / stats_gdf['men'].replace({0: np.nan})).fillna(0)
    if 'log_soc' in stats_gdf and 'men_coll' in stats_gdf and 'men_mais' in stats_gdf:
        stats_gdf['part_hlm'] = (stats_gdf['log_soc'] / (stats_gdf['men_coll'] + stats_gdf['men_mais']).replace({0: np.nan})).fillna(0)
    # Marquer les carreaux dont certaines données ont été imputées (si champ i_est_1km présent par ex.)
    if 'i_est_1km' in stats_gdf:
        stats_gdf['is_imputed'] = (stats_gdf['i_est_1km'] > 0).astype(int)
    # Remplacer les NaN restants par 0
    stats_gdf = stats_gdf.fillna(0)

    # Sauvegarder le tableau final des nœuds (features par carreau) pour une utilisation ultérieure
    save_geoparquet_data(stats_gdf, OUT_TILE_FEATURES)
    print(f"[OK] Données de {len(stats_gdf)} carreaux enregistrées dans {OUT_TILE_FEATURES}.")

    # 6. Construction du graphe de contiguïté des carreaux
    edges = build_micro_intra_edges(stats_gdf)
    np.save(OUT_EDGES_ALL, edges)
    print(f"[OK] Graphe complet (arêtes intra- + inter-communales) enregistré: {OUT_EDGES_ALL} ({edges.shape[1]} arêtes).")

    gdf_communes = gpd.read_file(COMMUNES_PATH)
    gdf_routes = gpd.read_file(ROUTE_PATH, columns=['ID', 'IMPORTANCE', 'geometry'])

    # 7. Construction des données macro
    edge_index, edge_attr, mapping_idx_code = build_macro_physical_graph(gdf_communes, gdf_routes)
    torch.save({
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'mapping': mapping_idx_code
    }, "data_GNN/graph_macro_physique.pt")
    # 8. Construction des données macro (Flux & Socio-éco) - MANQUANT
    print("--- Traitement des flux Macro ---")
    # Définir les chemins (idéalement via config.yaml, sinon en constantes)
    process_macro_flows(
        dad_path=MiGRATIONS,
        dt_path=TRAVAIL,
        pop_path=POPULATION,
        acti_path=ACTIVITE,
        output_nodes="data_GNN/nodes_macro_attributes.parquet",
        output_edges="data_GNN/edges_macro_flux.parquet"
    )


if __name__ == "__main__":
    main()
