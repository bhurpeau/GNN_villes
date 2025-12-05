import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_GNN = os.path.join(PROJECT_ROOT, "data_GNN")
OUT_DIR = os.path.join(PROJECT_ROOT, "out")

# Fichiers d'entrée
EMBEDDINGS_FILE = os.path.join(OUT_DIR, "resultats_embeddings.parquet")
STATS_MICRO = os.path.join(DATA_GNN, "statistiques_carreaux.parquet.gz")
STATS_MACRO = os.path.join(DATA_GNN, "nodes_macro_attributes.parquet")


# Paramètres Clustering
def parse_args():
    p = argparse.ArgumentParser(description="Nombre de clusters")
    p.add_argument("--clusters", type=int, default=6)
    return p.parse_args()


def analyze():
    print("--- ANALYSE ET TYPOLOGIE DES VILLES ---")
    args = parse_args()
    N_CLUSTERS = args.clusters
    # 1. Chargement des Embeddings (La "Forme + Fonction" vue par le GNN)
    print("Chargement des embeddings...")
    if not os.path.exists(EMBEDDINGS_FILE):
        print(
            f"❌ Erreur : Fichier {EMBEDDINGS_FILE} introuvable. Lancez inference.py d'abord."
        )
        return

    df_emb = pd.read_parquet(EMBEDDINGS_FILE)
    # Conversion des listes en matrice numpy
    X = np.stack(df_emb["z_final"].values)
    print(f"   -> {X.shape[0]} villes, {X.shape[1]} dimensions latentes.")

    # 2. Clustering (Faire émerger les types)
    print(f"Classification en {N_CLUSTERS} classes...")

    # On normalise avant le clustering (bonnes pratiques)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choix : CAH (Ward) pour des clusters compacts et équilibrés
    cah = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
    clusters = cah.fit_predict(X_scaled)

    df_emb["cluster"] = clusters
    df_emb["cluster"] = df_emb["cluster"].astype(str)  # Pour le categorical plotting

    # 3. Visualisation (Projection 2D)
    print("Génération de la projection PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", s=2, alpha=0.6
    )
    plt.colorbar(scatter, label="Type de Ville")
    plt.title(
        f"Projection de l'Espace Latent Morpho-Dynamique (PCA)\n{N_CLUSTERS} Types identifiés"
    )
    plt.xlabel("Dimension Principale 1 (Structure ?)")
    plt.ylabel("Dimension Principale 2 (Flux ?)")

    plot_path = os.path.join(OUT_DIR, f"clusters_pca_{N_CLUSTERS}.png")
    plt.savefig(plot_path)
    print(f"   -> Graphique sauvegardé : {plot_path}")

    # 4. Interprétation (Qui sont ces clusters ?)
    print("Croisement avec les données réelles pour interprétation...")

    # Chargement des données brutes pour calculer les moyennes par classe
    # A. Micro (On agrège les carreaux par commune)
    df_micro = pd.read_parquet(STATS_MICRO)
    # On calcule la moyenne pondérée ou simple des stats par commune
    # Variables clés : Densité, Revenu, Bâti
    cols_to_agg = [
        "densite_pop",
        "niveau_vie_moyen",
        "struct_bati",
        "struct_nature",
        "struct_agri",
    ]
    df_stats_micro = df_micro.groupby("code")[cols_to_agg].mean().reset_index()

    # B. Macro
    df_macro = pd.read_parquet(STATS_MACRO)

    # Jointure Générale
    df_final = df_emb[["code_insee", "cluster"]].merge(
        df_stats_micro, left_on="code_insee", right_on="code", how="left"
    )
    df_final = df_final.merge(
        df_macro, left_on="code_insee", right_on="code", how="left"
    )

    # Calcul du profil moyen par cluster
    profile = (
        df_final.groupby("cluster")
        .mean(numeric_only=True)
        .drop(columns=["code_x", "code_y"], errors="ignore")
    )

    print("\n--- PROFILS MOYENS DES CLUSTERS ---")
    print(profile.T)

    profile_path = os.path.join(OUT_DIR, f"profils_clusters_{N_CLUSTERS}.csv")
    profile.to_csv(profile_path)
    print(f"\n✅ Profils sauvegardés dans : {profile_path}")

    # 5. Export Final pour Cartographie (QGIS)
    # On garde Code, Cluster, et quelques stats clés
    export_cols = [
        "code_insee",
        "cluster",
        "densite_pop",
        "macro_taux_retenue",
        "niveau_vie_moyen",
    ]
    carto_path = os.path.join(OUT_DIR, f"resultats_carto_{N_CLUSTERS}.csv")
    df_final[export_cols].to_csv(carto_path, index=False)
    print(f"✅ Données pour QGIS sauvegardées dans : {carto_path}")


if __name__ == "__main__":
    analyze()
