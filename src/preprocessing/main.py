# main.py
# Pipeline principal CLI pour préparer les graphes

import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
from io import load_grid_shapefile, load_communes_shapefile, load_csv_data, load_parquet_data, save_parquet_data
from tile_processing import create_tile_id_raster, compute_landcover_composition, compute_altitude_stats, assign_tiles_to_communes
from graph_utils import build_contiguity_edges, build_commune_adjacency_graph

# Définition des chemins de fichiers en entrée (à adapter si besoin)
GRID_PATH = "data/grille1km_metropole.gpkg"
COMMUNES_PATH = "data/commune_francemetro_2023.gpkg"
RASTER_OCS_PATH = "data/OCS_2018.tif"
RASTER_DEM_PATH = "BDALTI/bdalti25m.tif"
RASTER_SLOPE_PATH = "BDALTI/bdalti25m_slope_deg.tif"

# Chemins de sortie (parquet et numpy)
OUT_TILE_FEATURES = "data_GNN/statistiques_carreaux.parquet.gz"
OUT_EDGES_INTRA = "data_GNN/edges_intra_communes.npy"
OUT_EDGES_ALL = "data_GNN/edges_toutes_communes.npy"


def main():
    parser = argparse.ArgumentParser(description="Prépare les données de graphe pour les communes (carreaux de 1km).")
    parser.add_argument("--commune", help="Code commune (INSEE) cible pour extraire le sous-graphe correspondant.")
    parser.add_argument("--no-inter", action="store_true", help="Si spécifié, exclut les arêtes inter-communales du graphe des carreaux.")
    args = parser.parse_args()

    # 1. Charger les données de base
    grid = load_grid_shapefile(GRID_PATH)            # Grille 1km (GeoDataFrame)
    communes = load_communes_shapefile(COMMUNES_PATH)  # Communes (GeoDataFrame)
    # (Optionnel: charger données population et socio-éco si disponibles)
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
    id_map_inv_10m = create_tile_id_raster(grid, RASTER_OCS_PATH, "data/grille1km_id_10m.tif", id_col="id_carr_1km")
    ocs_df = compute_landcover_composition("data/grille1km_id_10m.tif", RASTER_OCS_PATH, id_map_inv_10m)
    # Raster ID aligné sur le MNT (25m) pour altitude/pente
    id_map_inv_25m = create_tile_id_raster(grid, RASTER_DEM_PATH, "data/id_carr_1km_25m.tif", id_col="id_carr_1km")
    alt_df = compute_altitude_stats("data/id_carr_1km_25m.tif", RASTER_DEM_PATH, RASTER_SLOPE_PATH, id_map_inv_25m)
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
    save_parquet_data(stats_gdf, OUT_TILE_FEATURES)
    print(f"[OK] Données de {len(stats_gdf)} carreaux enregistrées dans {OUT_TILE_FEATURES}.")

    # 6. Construction du graphe de contiguïté des carreaux
    include_inter = not args.no_inter
    edges = build_contiguity_edges(stats_gdf, include_inter_communal=include_inter)
    if include_inter:
        np.save(OUT_EDGES_ALL, edges)
        print(f"[OK] Graphe complet (arêtes intra- + inter-communales) enregistré: {OUT_EDGES_ALL} ({edges.shape[1]} arêtes).")
    else:
        np.save(OUT_EDGES_INTRA, edges)
        print(f"[OK] Graphe intra-communal uniquement enregistré: {OUT_EDGES_INTRA} ({edges.shape[1]} arêtes).")

    # 7. Si une commune cible est spécifiée, extraire son sous-graphe
    if args.commune:
        code = args.commune
        # Filtrer les carreaux de cette commune
        commune_tiles = stats_gdf[stats_gdf['code'] == code].copy()
        if commune_tiles.empty:
            print(f"[ERREUR] Code commune {code} introuvable dans les données.")
            return
        # Inclure éventuellement les voisins immédiats de la commune pour garder les arêtes inter-communales
        if include_inter:
            # Sélectionner aussi les carreaux adjacents (voisins) appartenant aux autres communes
            all_edges = edges
            # Trouver les indices des nœuds correspondant à la commune cible
            node_indices = commune_tiles.index.to_list()
            # Trouver les arêtes où l'une des extrémités est dans ces indices
            mask_edges = np.isin(all_edges[0], node_indices) | np.isin(all_edges[1], node_indices)
            sub_edges = all_edges[:, mask_edges]
            # Récupérer l'ensemble des nœuds touchés par ces arêtes (commune + voisins)
            sub_node_indices = np.unique(sub_edges)
            subgraph_nodes = stats_gdf.iloc[sub_node_indices].copy()
        else:
            subgraph_nodes = commune_tiles
            sub_edges = build_contiguity_edges(subgraph_nodes, include_inter_communal=False)
        # Sauvegarder ou afficher le sous-graphe de la commune
        subgraph_nodes.to_file(f"data_GNN/commune_{code}_nodes.gpkg", driver="GPKG")
        np.save(f"data_GNN/commune_{code}_edges.npy", sub_edges)
        print(f"[OK] Sous-graphe de la commune {code} : {len(subgraph_nodes)} nœuds, {sub_edges.shape[1]} arêtes.")


if __name__ == "__main__":
    main()
