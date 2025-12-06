import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from .data_io import load_parquet_data, save_parquet_data


def process_amenities(bpe_path, grid_gdf, out_micro_path, out_macro_path):
    """
    Traite la BPE (Format Parquet) pour enrichir :
    - Micro : Densité d'équipements (Vitalité)
    - Macro : Équipements structurants (Centralité)
    """
    print("--- Traitement des Aménités (BPE Parquet) ---")

    # 1. Chargement optimisé
    print(f"   -> Chargement de {bpe_path}...")

    # Lecture directe du Parquet
    # On ne charge que les colonnes utiles pour économiser la RAM
    cols_utiles = ["TYPEQU", "LAMBERT_X", "LAMBERT_Y", "DEPCOM"]
    df_bpe = load_parquet_data(bpe_path, cols=cols_utiles)

    # Conversion en GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_bpe.LAMBERT_X, df_bpe.LAMBERT_Y)]
    gdf_bpe = gpd.GeoDataFrame(df_bpe, geometry=geometry, crs="EPSG:2154")

    # --- NIVEAU MICRO : Vitalité Locale ---
    print("   -> Calcul de la Vitalité Micro...")

    # Jointure spatiale (Points équipements -> Carreaux)
    joined = gpd.sjoin(
        gdf_bpe, grid_gdf[["id_carr_1km", "geometry"]], how="inner", predicate="within"
    )

    # Comptage par carreau
    micro_counts = (
        joined.groupby("id_carr_1km").size().reset_index(name="nb_equipements")
    )

    # Log-densité (Feature Engineering)
    micro_counts["log_amenity_density"] = np.log1p(micro_counts["nb_equipements"])

    save_parquet_data(micro_counts, out_micro_path)

    # --- NIVEAU MACRO : Centralité ---
    print("   -> Calcul de la Centralité Macro...")

    # Liste des équipements structurants (Santé, Éducation, Commerce, Loisirs...)
    # À adapter selon vos codes BPE cibles
    codes_structurants = [
        "D101",
        "D102",
        "D103",  # CHU, Hôpitaux
        "C101",
        "C102",
        "C104",  # Lycées, Enseignement Sup
        "G801",
        "G802",  # Hypermarchés, Supermarchés
        "A501",
        "A504",  # Administration, Justice
        "F303",  # Cinéma
    ]

    bpe_struct = df_bpe[df_bpe["TYPEQU"].isin(codes_structurants)]

    # Comptage par commune
    macro_counts = (
        bpe_struct.groupby("DEPCOM").size().reset_index(name="nb_equip_structurants")
    )
    macro_counts = macro_counts.rename(columns={"DEPCOM": "code"})

    save_parquet_data(macro_counts, out_macro_path)

    return micro_counts, macro_counts
