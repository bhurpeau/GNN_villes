import geopandas as gpd
import pandas as pd
from libpysal import weights
import networkx as nx
import numpy as np
DATA_IN = "./data_GNN/statistiques_carreaux.parquet.gz"
DATA_OUT = "./data_GNN/edges_micro_intra.npy"


def main():
    # 1. Charger votre GeoDataFrame propre (avec géométrie et id_commune)
    # Supposons qu'il s'appelle 'df_final'
    # Assurez-vous que l'index est bien réinitialisé pour avoir des ID de 0 à N
    df_final = gpd.read_parquet(DATA_IN)
    df_final = df_final.reset_index(drop=True)
    
    print("Construction du graphe de contiguïté... (Cela peut prendre 1-2 minutes)")
    
    # 2. Calculer les poids de contiguïté (Queen) sur toute la région
    # Cela crée une matrice géante où chaque carreau est connecté à ses voisins physiques
    w = weights.Queen.from_dataframe(df_final, use_index=False)
    
    # 3. Transformer en liste d'arêtes (Format "Adjacency List")
    # adj_list est une liste de tuples (index_source, index_cible)
    adj_list = w.to_adjlist(remove_symmetric=False)
    
    # 4. FILTRAGE : La règle du Sous-Graphe Disjoint
    # On ajoute les infos de commune pour la source et la cible
    adj_list['commune_source'] = df_final.loc[adj_list['focal'], 'code'].values
    adj_list['commune_cible'] = df_final.loc[adj_list['neighbor'], 'code'].values
    
    # On ne garde le lien QUE si les deux carreaux sont dans la même commune
    # (On coupe les frontières entre villes)
    intra_edges = adj_list[adj_list['commune_source'] == adj_list['commune_cible']]
    
    # 5. Nettoyage final pour le GNN
    # On ne garde que les indices (Source -> Cible)
    edge_index = intra_edges[['focal', 'neighbor']].values.T 
    # .T transpose pour avoir le format [[S1, S2...], [T1, T2...]] typique de PyTorch Geometric
    
    print(f"Graphe construit !")
    print(f"Nombre total de connexions internes : {edge_index.shape[1]}")
    print(f"Exemple : Le carreau {edge_index[0][0]} est connecté au carreau {edge_index[1][0]}")
    
    np.save(DATA_OUT, edge_index)

if __name__ == "__main__":
    main()