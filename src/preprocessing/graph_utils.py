# graph_utils.py
# Construction du graphe de voisinage des carreaux et communes

import numpy as np
from libpysal import weights
import geopandas as gpd
import torch  # utilisé pour formater la sortie des arêtes des communes


def build_contiguity_edges(tiles_gdf: gpd.GeoDataFrame, include_inter_communal: bool = True) -> np.ndarray:
    """
    Construit les arêtes de contiguïté entre carreaux adjacents.
    Args:
        tiles_gdf (GeoDataFrame): GeoDataFrame des carreaux avec au minimum la géométrie et un code commune ('code').
        include_inter_communal (bool): Si False, ne garde que les arêtes entre carreaux de la même commune (intra-communales).
                                       Si True, inclut toutes les arêtes de voisinage (y compris entre communes différentes).
    Returns:
        edge_index (ndarray shape [2, M]): Tableau d'indices des paires de carreaux voisins (format [source_indices; target_indices]).
                                           Les indices de nœuds correspondent à l'index de tiles_gdf.reset_index(drop=True).
    """
    # S'assurer que l'index va de 0 à N-1 pour correspondre aux IDs de nœuds
    tiles = tiles_gdf.reset_index(drop=True).copy()
    # Calcul des voisins de Queen (partage d'une frontière ou d'un coin)
    w = weights.contiguity.Queen.from_dataframe(tiles, use_index=False)
    # Obtenir la liste des connexions (adjacency list)
    adj = w.to_adjlist(remove_symmetric=False)  # on garde les deux directions
    # Filtrer éventuellement les arêtes inter-communales
    if not include_inter_communal:
        # Récupérer le code commune de chaque index dans adj
        adj['comm_source'] = tiles.loc[adj['focal'], 'code'].values
        adj['comm_target'] = tiles.loc[adj['neighbor'], 'code'].values
        # Ne garder que les paires dont les deux carreaux sont de la même commune
        adj = adj[adj['comm_source'] == adj['comm_target']]
    # Construire le tableau d'arêtes
    edge_index = np.vstack([adj['focal'].to_numpy(), adj['neighbor'].to_numpy()])
    return edge_index


def build_commune_adjacency_graph(communes_gdf: gpd.GeoDataFrame):
    """
    Construit le graphe des communes adjacentes, pondéré par la longueur de frontière commune.
    Args:
        communes_gdf (GeoDataFrame): Doit contenir au moins 'code' pour identifier la commune et 'geometry' pour le polygone.
    Returns:
        edge_index (torch.LongTensor): tensor [2, E] des arêtes entre communes (indices 0..N-1 correspondant à l'index du GeoDataFrame fourni).
        edge_attr (torch.FloatTensor): tensor [E, 1] avec la longueur de frontière commune (normalisée par log1p).
        idx_to_code (dict): mapping de l'index numérique du nœud vers le code commune.
    """
    # Copier et réindexer le GeoDataFrame des communes
    gdf = communes_gdf.reset_index(drop=True).copy()
    gdf['node_idx'] = gdf.index  # index explicite du nœud
    idx_to_code = gdf['code'].to_dict()

    # Légère correction topologique : buffer 0 pour réparer d'éventuelles géométries invalides
    gdf['geometry'] = gdf.geometry.buffer(0.1)

    # Jointure spatiale du gdf communes avec lui-même pour trouver toutes les intersections (communes adjacentes ou se touchant)
    adj = gpd.sjoin(gdf[['geometry', 'node_idx']], gdf[['geometry', 'node_idx']], 
                    how='inner', predicate='intersects')
    # Enlever les auto-intersections (une commune avec elle-même)
    adj = adj[adj['node_idx_left'] != adj['node_idx_right']]

    # Calcul de la longueur des frontières communes pour chaque paire
    geom_left = gdf.loc[adj['node_idx_left'], 'geometry'].values
    geom_right = gdf.loc[adj['node_idx_right'], 'geometry'].values
    # Intersection des polygones adjacents -> résultat en LineString (ligne de frontière)
    border_lines = gpd.GeoSeries(geom_left).intersection(gpd.GeoSeries(geom_right))
    lengths = border_lines.length  # longueur en unités de la projection (mètres si projection métrique)

    # Filtrer les frontières insignifiantes (longueur <= 1 m, probablement juste un coin touchant)
    valid = lengths > 1.0
    src_nodes = adj['node_idx_left'].to_numpy()[valid]
    tgt_nodes = adj['node_idx_right'].to_numpy()[valid]
    border_lengths = lengths.to_numpy()[valid]

    # Construire les tenseurs PyTorch pour edge_index et edge_attr
    edge_index = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
    # On utilise une échelle logarithmique pour réduire la plage des valeurs de longueur
    edge_attr = torch.tensor(border_lengths, dtype=torch.float32).log1p().unsqueeze(1)
    return edge_index, edge_attr, idx_to_code
