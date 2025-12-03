# graph_utils.py
# Construction du graphe de voisinage des carreaux et communes

import numpy as np
import geopandas as gpd
import torch
import pandas as pd
from libpysal import weights


def build_micro_intra_edges(df_tiles):
    """
    Construit les arêtes intra-communales (Micro) basées sur la contiguïté Reine.
    Filtre les liens qui traversent les frontières communales.
    """
    print("--- Construction du Graphe Micro (Intra-Commune) ---")
    df_tiles = df_tiles.reset_index(drop=True)

    # 1. Calcul de l'adjacence physique (Queen)
    # Silence_warnings évite les alertes sur les îles sans voisins
    w = weights.Queen.from_dataframe(df_tiles, use_index=False, silence_warnings=True)
    adj = w.to_adjlist(remove_symmetric=False)

    # 2. Filtrage : On ne garde que les liens au sein de la MÊME commune
    # On mappe les codes communes
    communes = df_tiles['code'].values
    adj['commune_src'] = communes[adj['focal']]
    adj['commune_dst'] = communes[adj['neighbor']]

    mask_internal = adj['commune_src'] == adj['commune_dst']
    internal_edges = adj[mask_internal]

    # 3. Conversion en Tenseur
    edge_index = torch.tensor([
        internal_edges['focal'].values,
        internal_edges['neighbor'].values
    ], dtype=torch.long)

    print(f"-> {edge_index.shape[1]} arêtes micro internes générées.")
    return edge_index


def build_macro_physical_graph(gdf_communes, gdf_routes):
    """
    Construit le graphe Macro complet (Squelette physique + Perméabilité routière).
    Intègre l'optimisation spatiale et les masques alignés.
    """
    print("--- Construction du Graphe Macro (Physique + Routes) ---")

    # 1. Préparation
    gdf = gdf_communes.reset_index(drop=True).copy()
    gdf['node_idx'] = gdf.index
    mapping_idx_code = gdf['code'].to_dict()

    # Buffer de sécurité pour la topologie
    gdf['geometry'] = gdf.geometry.buffer(0.1)

    # 2. Détection des voisins (Spatial Join)
    print("   -> Détection des frontières...")
    adj = gpd.sjoin(
        gdf[['geometry', 'node_idx']], 
        gdf[['geometry', 'node_idx']], 
        how='inner', 
        predicate='intersects'
    )
    # Retirer les auto-boucles
    adj = adj[adj['node_idx_left'] != adj['node_idx_right']]

    # 3. Calcul des longueurs (Vectorisé)
    print("   -> Calcul des longueurs...")
    geom_src = gdf.loc[adj['node_idx_left'], 'geometry'].values
    geom_dst = gdf.loc[adj['node_idx_right'], 'geometry'].values

    # Intersection vectorielle
    intersections = gpd.GeoSeries(geom_src).intersection(gpd.GeoSeries(geom_dst))
    lengths = intersections.length

    # 4. MASQUE STRICT (Alignement des index)
    # On ne garde que > 1m
    mask_valid = (lengths > 1.0).values

    final_src = adj['node_idx_left'].values[mask_valid]
    final_dst = adj['node_idx_right'].values[mask_valid]
    final_len = lengths.values[mask_valid]

    print(f"   -> {len(final_src)} frontières valides.")

    # 5. Calcul des Routes (Perméabilité)
    # On appelle la sous-fonction optimisée
    print("   -> Calcul de la perméabilité routière...")
    road_counts = _compute_road_crossings(
        final_src, final_dst, gdf, gdf_routes
    )

    # 6. Assemblage Final
    edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)

    # Attributs : [Log(Longueur), Log(Nb_Routes + 1)]
    attr_len = torch.tensor(final_len, dtype=torch.float).log1p().unsqueeze(1)
    attr_road = torch.tensor(road_counts, dtype=torch.float).log1p().unsqueeze(1)

    edge_attr = torch.cat([attr_len, attr_road], dim=1)

    return edge_index, edge_attr, mapping_idx_code


def _compute_road_crossings(src_idx, dst_idx, gdf_communes, gdf_routes):
    """
    Sous-fonction privée optimisée avec Spatial Index.
    """
    # Index spatial des routes
    routes_sindex = gdf_routes.sindex

    # Poids des routes (Feature Engineering)
    # Adaptez 'NATURE' selon votre BD TOPO
    if 'IMPORTANCE' in gdf_routes.columns:
        poids_route = {
            '1': 10.0,
            '2': 8.0,
            '3': 5.0,
            '4': 3.0,
            '5': 1.0,
            '6': 0.0,
        }
        # On map et on remplit les inconnus par 1.0 (Route standard)
        route_weights = gdf_routes['IMPORTANCE'].map(poids_route).fillna(1.0).values
    else:
        route_weights = np.ones(len(gdf_routes))

    gdf_routes['w_calc'] = route_weights

    geoms = gdf_communes.geometry.values
    results = []

    for i, (s, d) in enumerate(zip(src_idx, dst_idx)):
        # Intersection précise (Ligne frontière)
        boundary = geoms[s].intersection(geoms[d])

        if boundary.is_empty:
            results.append(0.0)
            continue

        # 1. Filtre Spatial Rapide (Bounding Box)
        # Renvoie les indices entiers des routes candidates
        candidate_ids = list(routes_sindex.query(boundary, predicate='intersects'))

        if not candidate_ids:
            results.append(0.0)
            continue

        # 2. Intersection Précise sur les candidats
        candidates = gdf_routes.iloc[candidate_ids]
        real_inter = candidates[candidates.intersects(boundary)]

        if real_inter.empty:
            results.append(0.0)
        else:
            results.append(real_inter['w_calc'].sum())

        if i % 10000 == 0:
            print(f"      ... {i}/{len(src_idx)} traités")

    return results
