import geopandas as gpd
import pandas as pd
import torch
import numpy as np

def main():
    grille = gpd.read_file("./data/grille1km_metropole.gpkg")
    communes = gpd.read_file("./data/commune_francemetro_2023.gpkg")
    # 1. Intersection complète (Overlay)
    # Cela découpe les carreaux aux frontières exactes des communes
    pieces = gpd.overlay(grille, communes, how='intersection')
    
    # 2. Calculer la surface de chaque morceau
    pieces['area_piece'] = pieces.geometry.area
    
    # 3. Trier pour mettre les plus gros morceaux en premier
    pieces = pieces.sort_values('area_piece', ascending=False)
    
    # 4. Pour chaque ID_Carreau, ne garder que le premier (le plus gros)
    
    attribution_finale = pieces.drop_duplicates(subset=['id_carr_1km'], keep='first')
    
    # 5. Nettoyage : on ne garde que l'ID carreau et l'ID commune 
    mapping_carreau_commune = attribution_finale[['id_carr_1km', 'code']]
    mapping_carreau_commune.to_parquet("./data/mapping_carreau_commune.parquet")
    
    grid = pd.read_parquet("./data/grid_1km.parquet")
    grid = grid[['GRD_ID', 'TOT_P_2018', 'TOT_P_2006',  'TOT_P_2011',
           'TOT_P_2021']]
    grille = gpd.read_file("./data/grille1km_metropole.gpkg")
    
    grille = grille.merge(grid, left_on='id_carr_1km', right_on='GRD_ID', how='left').drop(['idINSPIRE', 'GRD_ID'], axis=1)
    
    filo19 = pd.read_csv("./data/filo_2019_carreaux_1km_met.csv")
    
    stats = pd.read_parquet("./data/stats_1km.parquet")
    
    stats = stats.merge(grille, how='left', on="id_carr_1km")
    stats = stats.merge(filo19, how='left', left_on = 'id_carr_1km', right_on='idcar_1km')
    
    # 1. Tissu Urbain (La "Masse" bâtie)
    stats['struct_bati'] = stats['part_classe_1'] + stats['part_classe_2']
    
    # 2. Zones d'Activité (Emploi / ZI / Commercial / Routes)
    stats['struct_eco'] = stats['part_classe_3'] + stats['part_classe_4']
    
    # 3. Nature (Forêts et milieux semi-naturels)
    # Classes 11 à 19 (Forêts, Landes, Fourrés...)
    # Astuce : on utilise une liste pour être sûr de tout prendre
    cols_nature = [f'part_classe_{i}' for i in range(11, 20)] 
    stats['struct_nature'] = stats[cols_nature].sum(axis=1)
    
    # 4. Agriculture (Champs - Réserve foncière)
    # Classes 20 à 22 (Cultures, Prairies, Vergers)
    cols_agri = [f'part_classe_{i}' for i in range(20, 23)]
    stats['struct_agri'] = stats[cols_agri].sum(axis=1)
    
    # 5. Eau & Glaciers (Hydrosphère)
    # On peut grouper Eau (23) et Glaciers (24) ou les séparer.
    # Pour la morpho pure, les séparer est intéressant (le glacier est une contrainte "montagne", l'eau une "aménité")
    stats['struct_eau'] = stats['part_classe_23']
    stats['struct_glacier'] = stats['part_classe_24'] # Spécifique Alpes
    
    # Verification : La somme devrait faire (1 - part_classe_0)
    # Cela vous permet de vérifier l'intégrité des données
    
    
    
    # 1. Densité Humaine (Indicateur de charge)
    # On divise par la surface du carreau (qui est de 1km², donc c'est déjà une densité)
    stats['densite_pop'] = stats['TOT_P_2021'] # Ou 'ind' de Filosofi
    
    # 2. Richesse Relative
    # Attention : ind_snv est une SOMME. Il faut diviser par le nombre d'individus.
    # Remplacer les 0 par NaN temporairement pour éviter l'erreur, puis fillna
    stats['niveau_vie_moyen'] = (stats['ind_snv'] / stats['ind'].replace(0, np.nan)).fillna(0)
    
    # 3. Précarité Structurelle
    stats['taux_pauvrete'] = (stats['men_pauv'] / stats['men'].replace(0, np.nan)).fillna(0)
    
    # 4. Morphologie Sociale (Propriétaire vs Locataire)
    stats['part_proprio'] = (stats['men_prop'] / stats['men'].replace(0, np.nan)).fillna(0)
    
    # 5. Type d'Habitat (Morphologie Bâtie vue par l'INSEE)
    stats['part_maison'] = (stats['men_mais'] / stats['men'].replace(0, np.nan)).fillna(0)
    stats['part_hlm'] = (stats['log_soc'] / (stats['men_coll'] + stats['men_mais']).replace(0, np.nan)).fillna(0)
    
    # 6. Dynamique Démographique (Votre touche temporelle !)
    # Croissance sur 10 ans (2011-2021)
    stats['croissance_pop'] = (stats['TOT_P_2021'] - stats['TOT_P_2011']) / stats['TOT_P_2011'].replace(0, np.nan)
    stats['croissance_pop'] = stats['croissance_pop'].fillna(0)
    stats['is_imputed'] = (stats['i_est_1km'] > 0).astype(int)
    
    df_final = stats[['id_carr_1km','z_mean', 'z_std', 'slope_mean','struct_bati',
           'struct_eco', 'struct_nature', 'struct_agri', 'struct_eau',
           'struct_glacier','densite_pop', 'niveau_vie_moyen', 'taux_pauvrete',
           'part_proprio', 'part_maison', 'part_hlm', 'croissance_pop','is_imputed','geometry']]
    
    df_final = df_final.fillna(0)
    
    mapping_carreau_commune = pd.read_parquet("./data/mapping_carreau_commune.parquet")
    
    df_final = df_final.merge(mapping_carreau_commune, how='left', on='id_carr_1km')
    
    df_final = gpd.GeoDataFrame(df_final, geometry=df_final['geometry'], crs="EPSG:2154")
    
    # 1. Isoler les orphelins et les bien classés
    # On suppose que votre DF s'appelle 'df_final' et est un GeoDataFrame
    orphelins = df_final[df_final['code'].isna()].copy()
    bien_classes = df_final[~df_final['code'].isna()].copy()
    
    print(f"Nombre d'orphelins à sauver : {len(orphelins)}")
    
    # 2. Charger le fond de carte des communes (Contours officiels IGN ou Admin Express)
    # Il faut absolument les polygones des communes pour calculer la distance
    communes = communes[['code', 'geometry']] 
    
    # 3. Spatial Join "Nearest" (Le Sauvetage)
    # On cherche pour chaque orphelin la commune la plus proche
    # Attention : assurez-vous d'être dans le même CRS (ex: EPSG:2154 pour la France)
    sauvetage = gpd.sjoin_nearest(
        orphelins.drop(columns=['code']), # On enlève la colonne vide pour la remplacer
        communes,
        how='left',
        distance_col='dist_to_commune' # Utile pour vérifier qu'on ne va pas chercher trop loin
    )
    
    # 4. Nettoyage post-sauvetage
    # Si la distance est trop grande (ex: > 2km), c'est que c'est un vrai bug ou un carreau
    # perdu au milieu de la Suisse. On peut décider de les jeter.
    seuil_dist = 2000 # mètres
    sauvetage_valide = sauvetage[sauvetage['dist_to_commune'] < seuil_dist].copy()
    
    # Renommer la colonne récupérée (INSEE_COM) en lcog_geo pour matcher l'autre table
    # sauvetage_valide = sauvetage_valide.rename(columns={'INSEE_COM': 'lcog_geo'})
    
    # 5. Re-fusionner tout le monde
    cols_to_keep = bien_classes.columns # On garde la même structure
    df_complet = pd.concat([bien_classes, sauvetage_valide[cols_to_keep]])
    
    print(f"Total final : {len(df_complet)} carreaux.")
    
    df_complet.to_parquet("./data_GNN/statistiques_carreaux.parquet.gz", compression='gzip')

if __name__ == "__main__":
    main()