import geopandas as gpd


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


if __name__ == "__main__":
    main()