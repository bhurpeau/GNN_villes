import geopandas as gpd
import pandas as pd
import torch
import numpy as np

def build_contact_opportunity_edges(gdf_communes):
    """
    Construit le graphe d'adjacence physique pondéré par la longueur de la frontière commune.
    
    Args:
        gdf_communes (GeoDataFrame): Doit contenir une colonne 'code' et 'geometry'.
                                     Les polygones doivent être valides.
    
    Returns:
        edge_index (LongTensor): [2, num_edges] Les connexions (Indices 0 à N)
        edge_attr (FloatTensor): [num_edges, 1] La longueur normalisée (Log)
        mapping_idx_code (dict): Pour retrouver le code commune à partir de l'index
    """
    print("--- Construction du Graphe d'Opportunité de Contact (Physique) ---")
    
    # 1. Préparation et Indexation
    # On travaille sur une copie pour ne pas casser l'original
    # On reset_index pour avoir des ID de nœuds propres (0, 1, 2... N) pour PyTorch
    gdf = gdf_communes.reset_index(drop=True).copy()
    
    # On crée une colonne explicite pour l'index du nœud
    gdf['node_idx'] = gdf.index
    
    # Création du mapping pour plus tard (Indispensable pour le niveau Macro)
    # Ex: {0: '01001', 1: '01002'...}
    mapping_idx_code = gdf['code'].to_dict()
    
    print("1. Nettoyage topologique léger...")
    # Buffer(0) ou très petit permet de réparer les géométries invalides 
    # et d'assurer que des polygones qui se touchent sont bien détectés
    gdf['geometry'] = gdf.geometry.buffer(0.1) 

    print("2. Détection des voisins (Spatial Join Vectorisé)...")
    # On joint le dataframe avec lui-même pour trouver les intersections
    # On garde les colonnes 'node_idx' pour savoir qui touche qui
    # suffixe _left = Source, suffixe _right = Cible
    adj = gpd.sjoin(
        gdf[['geometry', 'node_idx']], 
        gdf[['geometry', 'node_idx']], 
        how='inner', 
        predicate='intersects' # 'intersects' est plus robuste que 'touches' pour les frontières
    )
    
    # Filtre : On retire les auto-connexions (une commune se touche elle-même)
    adj = adj[adj['node_idx_left'] != adj['node_idx_right']]
    
    print(f"   -> {len(adj)} paires voisines potentielles détectées.")
    
    print("3. Calcul précis des longueurs de frontières...")
    # Optimisation : Au lieu de faire une boucle, on aligne les géométries
    # On récupère les géométries "Gauche" et "Droite" alignées sur les index du sjoin
    geom_source = gdf.loc[adj['node_idx_left'], 'geometry'].values
    geom_target = gdf.loc[adj['node_idx_right'], 'geometry'].values
    
    # Intersection vectorielle (GeoSeries vs GeoSeries)
    # Cela génère des LineString (la frontière commune)
    gs_source = gpd.GeoSeries(geom_source)
    gs_target = gpd.GeoSeries(geom_target)
    intersections = gs_source.intersection(gs_target)
    
    # Calcul de la longueur en mètres (si projection métrique type Lambert 93)
    lengths = intersections.length
    
    # 4. Nettoyage final
    # On ne garde que les frontières qui ont une longueur significative (> 1 mètre)
    # Cela élimine les points de contact uniques ou les erreurs de topologie
    mask_valid = lengths > 1.0
    
    final_src = adj['node_idx_left'][mask_valid].values
    final_dst = adj['node_idx_right'][mask_valid].values
    final_attr = lengths[mask_valid].values
    
    print(f"   -> {len(final_src)} arêtes physiques validées (Longueur > 1m).")

    # 5. Conversion en Tenseurs PyTorch
    # Format edge_index : [[Sources], [Cibles]]
    edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)
    
    # Format edge_attr : Normalisation Logarithmique
    # Pourquoi ? Une frontière peut faire 10m ou 50km. 
    # Le réseau apprend mieux avec des valeurs compressées (ex: entre 0 et 10) qu'avec 50000.
    # log1p calcule log(1 + x) pour gérer les petites valeurs proprement.
    edge_attr = torch.tensor(final_attr, dtype=torch.float).log1p().unsqueeze(1)
    
    return edge_index, edge_attr, mapping_idx_code

# Exemple d'utilisation :
# edge_index_phys, edge_attr_phys, map_code = build_contact_opportunity_edges(mon_gdf_communes)