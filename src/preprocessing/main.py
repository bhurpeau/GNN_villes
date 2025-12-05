# main.py
# Pipeline principal CLI pour préparer les graphes

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import os
from .data_io import (
    run,
    load_grid_shapefile,
    load_communes_shapefile,
    load_csv_data,
    load_parquet_data,
    save_geoparquet_data,
)
from .tile_processing import (
    create_tile_id_raster,
    compute_landcover_composition,
    compute_altitude_stats,
    assign_tiles_to_communes,
)
from .graph_utils import build_micro_intra_edges, build_macro_physical_graph
from .cities_processing import process_macro_flows
from .amenities_processing import process_amenities

# Définition des chemins de fichiers en entrée (à adapter si besoin)
GRID_PATH = "data/grille1km_metropole.gpkg"
COMMUNES_PATH = "data/commune_francemetro_2023.gpkg"
ROUTE_PATH = "data/bdtopo_routes.gpkg"
RASTER_OCS_PATH = "data/OCS_2023.tif"
RASTER_DEM_PATH = "BDALTI/bdalti25m.tif"
RASTER_SLOPE_PATH = "BDALTI/bdalti25m_slope_deg.tif"
MiGRATIONS = "data/DAD21.parquet"
TRAVAIL = "data/DT21.parquet"
POPULATION = "data/pop21.csv"
ACTIVITE = "data/acti.csv"
BPE_PATH = "data/BPE24.parquet"
# Chemins de sortie (parquet et numpy)
OUT_TILE_FEATURES = "data_GNN/statistiques_carreaux.parquet.gz"
OUT_EDGES_ALL = "data_GNN/edges_toutes_communes.npy"
raster_output = "data/grille1km_id_10m.tif"
raster_output_25 = "data/id_carr_1km_25m.tif"
out_micro_bpe = "data_GNN/amenities_micro.parquet"
out_macro_bpe = "data_GNN/amenities_macro.parquet"


def main():
    # 1. Charger les données de base
    grid = load_grid_shapefile(GRID_PATH)  # Grille 1km (GeoDataFrame)
    communes = load_communes_shapefile(COMMUNES_PATH)  # Communes (GeoDataFrame)
    try:
        pop_df = load_parquet_data(
            "data/grid_1km.parquet"
        )  # population par carreau (INSEE grille 1km)
    except FileNotFoundError:
        pop_df = None
    try:
        socio_df = load_csv_data(
            "data/filo_2019_carreaux_1km_met.csv"
        )  # données Filosofi 2019 par carreau
    except FileNotFoundError:
        socio_df = None

    print("Traitement des équipements (BPE)...")
    try:
        df_micro_bpe, df_macro_bpe = process_amenities(BPE_PATH, grid, out_micro_bpe, out_macro_bpe)
    except FileNotFoundError:
        print(f"⚠️ Attention : Fichier BPE introuvable à {BPE_PATH}. Les aménités seront ignorées.")
        df_micro_bpe, df_macro_bpe = None, None
    # 2. Rasterisation de la grille aux résolutions nécessaires et calcul des features par carreau
    # Raster ID aligné sur l'OCS (10m) pour l'occupation du sol
    if os.path.exists(raster_output):
        os.remove(raster_output)
    print(f"⚙️ Génération de {raster_output}...")
    id_map_inv_10m = create_tile_id_raster(
        grid, RASTER_OCS_PATH, raster_output, id_col="id_carr_1km"
    )
    ocs_df = compute_landcover_composition(raster_output, RASTER_OCS_PATH, id_map_inv_10m)
    # Raster ID aligné sur le MNT (25m) pour altitude/pente
    if os.path.exists(raster_output_25):
        os.remove(raster_output_25)
    print(f"⚙️ Génération de {raster_output_25}...")
    id_map_inv_25m = create_tile_id_raster(
        grid, RASTER_DEM_PATH, raster_output_25, id_col="id_carr_1km"
    )
    if os.path.exists(RASTER_SLOPE_PATH):
        os.remove(RASTER_SLOPE_PATH)
    cmd = ["gdaldem", "slope", RASTER_DEM_PATH, RASTER_SLOPE_PATH, "-compute_edges"]
    run(cmd)
    alt_df = compute_altitude_stats(
        raster_output_25, RASTER_DEM_PATH, RASTER_SLOPE_PATH, id_map_inv_25m
    )
    # 3. Fusion des features dans un seul DataFrame
    stats_df = pd.merge(alt_df, ocs_df, on="id_carr_1km", how="left")
    # Ajouter population et socio-éco si disponibles
    if pop_df is not None:
        stats_df = pd.merge(stats_df, pop_df, left_on="id_carr_1km", right_on="GRD_ID", how="left")
        if "GRD_ID" in stats_df.columns:
            stats_df = stats_df.drop(columns=["GRD_ID"])
    if socio_df is not None:
        stats_df = pd.merge(
            stats_df, socio_df, left_on="id_carr_1km", right_on="idcar_1km", how="left"
        )
        if "idcar_1km_y" in stats_df.columns:
            stats_df = stats_df.drop(columns=["idcar_1km_y"])
        elif "idcar_1km" in stats_df.columns and "id_carr_1km" in stats_df.columns:
            stats_df = stats_df.drop(columns=["idcar_1km"])
    stats_df["id_carr_1km"] = stats_df["id_carr_1km"].astype(str)

    if df_micro_bpe is not None:
        print("Fusion des aménités (Micro)...")
        stats_df = pd.merge(stats_df, df_micro_bpe, on="id_carr_1km", how="left")
        # On remplit les trous par 0 (pas d'équipement)
        stats_df["nb_equipements"] = stats_df["nb_equipements"].fillna(0)
        stats_df["log_amenity_density"] = stats_df["log_amenity_density"].fillna(0)
    # 4. Ajouter la géométrie des carreaux et assigner les communes
    # Merge avec la grille pour récupérer la géométrie
    stats_gdf = gpd.GeoDataFrame(
        pd.merge(stats_df, grid[["id_carr_1km", "geometry"]], on="id_carr_1km", how="left"),
        crs=grid.crs,
    )
    # Attribution des communes
    mapping_series = assign_tiles_to_communes(grid, communes)
    stats_gdf["code"] = stats_gdf["id_carr_1km"].map(mapping_series.to_dict())
    # 5. Calcul de features additionnelles (structures OCS et socio-éco)
    # Regroupements de classes OCS (structure du territoire)
    stats_gdf["struct_bati"] = stats_gdf.get("part_classe_1", 0) + stats_gdf.get("part_classe_2", 0)
    stats_gdf["struct_eco"] = stats_gdf.get("part_classe_3", 0) + stats_gdf.get("part_classe_4", 0)
    # Somme des classes nature 11 à 19
    nature_cols = [f"part_classe_{c}" for c in range(11, 20)]
    stats_gdf["struct_nature"] = (
        stats_gdf[nature_cols].sum(axis=1) if nature_cols[0] in stats_gdf else 0
    )
    # Somme des classes agriculture 20 à 22
    agri_cols = [f"part_classe_{c}" for c in range(20, 23)]
    stats_gdf["struct_agri"] = stats_gdf[agri_cols].sum(axis=1) if agri_cols[0] in stats_gdf else 0
    # Eau et glacier
    stats_gdf["struct_eau"] = stats_gdf.get("part_classe_23", 0)
    stats_gdf["struct_glacier"] = stats_gdf.get("part_classe_24", 0)
    # Indicateurs socio-démographiques (si données disponibles)
    if "TOT_P_2021" in stats_gdf:
        stats_gdf["densite_pop"] = stats_gdf["TOT_P_2021"]  # par km² (carreaux de 1 km²)
        if "TOT_P_2011" in stats_gdf:
            # Croissance démographique 2011-2021
            stats_gdf["croissance_pop"] = (
                stats_gdf["TOT_P_2021"] - stats_gdf["TOT_P_2011"]
            ) / stats_gdf["TOT_P_2011"].replace({0: np.nan})
            stats_gdf["croissance_pop"] = stats_gdf["croissance_pop"].fillna(0)
    if "ind_snv" in stats_gdf and "ind" in stats_gdf:
        stats_gdf["niveau_vie_moyen"] = (
            stats_gdf["ind_snv"] / stats_gdf["ind"].replace({0: np.nan})
        ).fillna(0)
    if "men_pauv" in stats_gdf and "men" in stats_gdf:
        stats_gdf["taux_pauvrete"] = (
            stats_gdf["men_pauv"] / stats_gdf["men"].replace({0: np.nan})
        ).fillna(0)
    if "men_prop" in stats_gdf and "men" in stats_gdf:
        stats_gdf["part_proprio"] = (
            stats_gdf["men_prop"] / stats_gdf["men"].replace({0: np.nan})
        ).fillna(0)
    if "men_mais" in stats_gdf and "men_coll" in stats_gdf:
        stats_gdf["part_maison"] = (
            stats_gdf["men_mais"] / stats_gdf["men"].replace({0: np.nan})
        ).fillna(0)
    if "log_soc" in stats_gdf and "men_coll" in stats_gdf and "men_mais" in stats_gdf:
        stats_gdf["part_hlm"] = (
            stats_gdf["log_soc"]
            / (stats_gdf["men_coll"] + stats_gdf["men_mais"]).replace({0: np.nan})
        ).fillna(0)
    # Marquer les carreaux dont certaines données ont été imputées (si champ i_est_1km présent par ex.)
    if "i_est_1km" in stats_gdf:
        stats_gdf["is_imputed"] = (stats_gdf["i_est_1km"] > 0).astype(int)
    # Remplacer les NaN restants par 0
    stats_gdf = stats_gdf.fillna(0)
    # Sauvegarder le tableau final des nœuds (features par carreau) pour une utilisation ultérieure
    to_keep = [
        "id_carr_1km",
        "z_mean",
        "z_std",
        "slope_mean",
        "part_classe_0",
        "part_classe_1",
        "part_classe_2",
        "part_classe_3",
        "part_classe_4",
        "part_classe_5",
        "part_classe_6",
        "part_classe_7",
        "part_classe_8",
        "part_classe_9",
        "part_classe_10",
        "part_classe_11",
        "part_classe_12",
        "part_classe_13",
        "part_classe_14",
        "part_classe_15",
        "part_classe_16",
        "part_classe_17",
        "part_classe_18",
        "part_classe_19",
        "part_classe_20",
        "part_classe_21",
        "part_classe_22",
        "part_classe_23",
        "part_classe_24",
        "DIST_BORD",
        "TOT_P_2018",
        "TOT_P_2006",
        "TOT_P_2011",
        "TOT_P_2021",
        "i_est_1km",
        "lcog_geo",
        "ind",
        "men",
        "men_pauv",
        "men_1ind",
        "men_5ind",
        "men_prop",
        "men_fmp",
        "ind_snv",
        "men_surf",
        "men_coll",
        "men_mais",
        "log_av45",
        "log_45_70",
        "log_70_90",
        "log_ap90",
        "log_inc",
        "log_soc",
        "ind_0_3",
        "ind_4_5",
        "ind_6_10",
        "ind_11_17",
        "ind_18_24",
        "ind_25_39",
        "ind_40_54",
        "ind_55_64",
        "ind_65_79",
        "ind_80p",
        "ind_inc",
        "geometry",
        "code",
        "struct_bati",
        "struct_eco",
        "struct_nature",
        "struct_agri",
        "struct_eau",
        "struct_glacier",
        "densite_pop",
        "croissance_pop",
        "niveau_vie_moyen",
        "taux_pauvrete",
        "part_proprio",
        "part_maison",
        "part_hlm",
        "log_amenity_density",
        "nb_equipements",
        "is_imputed",
    ]

    cols_ids = ["id_carr_1km", "lcog_geo", "code"]
    for col in cols_ids:
        if col in stats_gdf.columns:
            # On remplit les NaNs par une valeur vide pour éviter le string "nan"
            stats_gdf[col] = stats_gdf[col].fillna("").astype(str)
    stats_gdf = stats_gdf[to_keep].copy()
    stats_gdf = stats_gdf.sort_values(["code", "id_carr_1km"]).reset_index(drop=True)

    save_geoparquet_data(stats_gdf, OUT_TILE_FEATURES)
    print(f"[OK] Données de {len(stats_gdf)} carreaux enregistrées dans {OUT_TILE_FEATURES}.")

    # 6. Construction du graphe de contiguïté des carreaux
    edges = build_micro_intra_edges(stats_gdf)
    np.save(OUT_EDGES_ALL, edges)
    print(
        f"[OK] Graphe complet (arêtes intra- + inter-communales) enregistré: {OUT_EDGES_ALL} ({edges.shape[1]} arêtes)."
    )

    gdf_communes = gpd.read_file(COMMUNES_PATH)
    gdf_routes = gpd.read_file(ROUTE_PATH, columns=["ID", "IMPORTANCE", "geometry"])

    # 7. Construction des données macro
    edge_index, edge_attr, mapping_idx_code = build_macro_physical_graph(gdf_communes, gdf_routes)
    torch.save(
        {"edge_index": edge_index, "edge_attr": edge_attr, "mapping": mapping_idx_code},
        "data_GNN/graph_macro_physique.pt",
    )
    # 8. Construction des données macro (Flux & Socio-éco) - MANQUANT
    print("--- Traitement des flux Macro ---")
    # Définir les chemins (idéalement via config.yaml, sinon en constantes)
    flux_internes, edges_final = process_macro_flows(
        dad_path=MiGRATIONS,
        dt_path=TRAVAIL,
        pop_path=POPULATION,
        acti_path=ACTIVITE,
        output_nodes="data_GNN/nodes_macro_attributes.parquet",
        output_edges="data_GNN/edges_macro_flux.parquet",
    )

    if df_macro_bpe is not None:
        print("Enrichissement Macro avec les équipements structurants...")
        # df_macro_bpe contient ["code", "nb_equip_structurants"]
        # flux_internes contient ["code", "macro_taux_retenue", ...]

        # Attention au typage du code commune pour la fusion
        flux_internes["code"] = flux_internes["code"].astype(str)
        df_macro_bpe["code"] = df_macro_bpe["code"].astype(str)

        nodes_enriched = flux_internes.merge(df_macro_bpe, on="code", how="left")
        nodes_enriched["nb_equip_structurants"] = nodes_enriched["nb_equip_structurants"].fillna(0)

        # On écrase le fichier précédent avec la version enrichie
        from .data_io import save_parquet_data

        save_parquet_data(nodes_enriched, "data_GNN/nodes_macro_attributes.parquet")
        print("✅ Nœuds Macro mis à jour avec BPE.")


if __name__ == "__main__":
    main()
