import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_GNN = os.path.join(PROJECT_ROOT, "data_GNN")
DATA_RAW = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "out")

# Fichiers
EMBEDDINGS_FILE = os.path.join(OUT_DIR, "resultats_embeddings.parquet")
STATS_MICRO = os.path.join(DATA_GNN, "statistiques_carreaux.parquet.gz")
STATS_MACRO = os.path.join(DATA_GNN, "nodes_macro_attributes.parquet")
COMMUNES_GPKG = os.path.join(DATA_RAW, "commune_francemetro_2023.gpkg")

# Définition des zones pour les cartes
ZONES_ZOOM = {
    "France": None,  # None = France entière
    "Paris_PC": ["75", "92", "93", "94"],
    "Lyon_Rhone": ["69"],
    "Marseille_BDR": ["13"],
}


def weighted_nan_average(values, weights):
    """Calcule une moyenne pondérée en ignorant les NaNs"""
    v = np.array(values)
    w = np.array(weights)
    mask = ~np.isnan(v) & ~np.isnan(w)
    if mask.sum() == 0 or w[mask].sum() == 0:
        return 0.0
    return np.average(v[mask], weights=w[mask])


def analyze():
    print("--- ANALYSE COMPLÈTE : STATS + CARTO (FRANCE & ZOOMS) ---")

    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"❌ Erreur : {EMBEDDINGS_FILE} manquant.")
        return

    # 1. Chargement et Fusion des Données
    print("Chargement des données...")
    df_emb = pd.read_parquet(EMBEDDINGS_FILE)
    df_micro = pd.read_parquet(STATS_MICRO)
    df_macro = pd.read_parquet(STATS_MACRO)

    # Chargement Géométrie une seule fois
    if os.path.exists(COMMUNES_GPKG):
        print("Chargement du fond de carte...")
        gdf_communes = gpd.read_file(COMMUNES_GPKG)[["code", "geometry"]]
        gdf_communes["dept"] = gdf_communes["code"].str[:2]  # Pour les filtres
    else:
        gdf_communes = None
        print("⚠️ Pas de fond de carte. Cartographie désactivée.")

    # Agrégation Micro -> Commune (Stats pondérées)
    def agg_density_func(x):
        w = df_micro.loc[x.index, "TOT_P_2021"]
        return weighted_nan_average(x, w)

    commune_stats = (
        df_micro.groupby("code")
        .agg(
            {
                "ind": "sum",
                "ind_snv": "sum",
                "TOT_P_2021": "sum",
                "struct_bati": "mean",
                "struct_nature": "mean",
                "struct_agri": "mean",
                "densite_pop": agg_density_func,
            }
        )
        .reset_index()
    )

    commune_stats["niveau_vie_moyen"] = commune_stats["ind_snv"] / commune_stats[
        "ind"
    ].replace(0, np.nan)

    # Fusion Totale
    df_final = df_emb.merge(
        commune_stats, left_on="code_insee", right_on="code", how="left"
    )
    df_final = df_final.merge(
        df_macro, left_on="code_insee", right_on="code", how="left"
    )

    # Remplissage NaNs Macro
    for col in ["macro_taux_retenue", "macro_taux_stabilite"]:
        df_final[col] = df_final[col].fillna(df_final[col].mean())

    # Préparation Clustering
    X = np.stack(df_final["z_final"].values)
    X_scaled = StandardScaler().fit_transform(X)

    # 2. Boucle d'analyse (6, 12, 18 clusters)
    for n_clusters in [6, 12, 18]:
        print(f"\n--- Traitement k={n_clusters} ---")

        # A. Clustering
        cah = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        clusters = cah.fit_predict(X_scaled)
        col_cluster = f"cluster_{n_clusters}"
        df_final[col_cluster] = clusters

        # B. Profils Statistiques
        def get_profile(grp):
            pop = grp["TOT_P_2021"]
            ind = grp["ind"]
            w_pop = pop.sum()
            w_ind = ind.sum()
            return pd.Series(
                {
                    "niveau_vie_moyen": (
                        np.average(grp["niveau_vie_moyen"], weights=ind)
                        if w_ind > 0
                        else 0
                    ),
                    "densite_pop": (
                        np.average(grp["densite_pop"], weights=pop) if w_pop > 0 else 0
                    ),
                    "struct_bati": grp["struct_bati"].mean(),
                    "struct_nature": grp["struct_nature"].mean(),
                    "struct_agri": grp["struct_agri"].mean(),
                    "macro_taux_retenue": (
                        np.average(grp["macro_taux_retenue"], weights=pop)
                        if w_pop > 0
                        else 0
                    ),
                    "macro_taux_stabilite": (
                        np.average(grp["macro_taux_stabilite"], weights=pop)
                        if w_pop > 0
                        else 0
                    ),
                }
            )

        profile = df_final.groupby(col_cluster).apply(get_profile)
        profile_path = os.path.join(OUT_DIR, f"profils_clusters_{n_clusters}.csv")
        profile.T.to_csv(profile_path)
        print(f"   ✅ Profils sauvegardés : {profile_path}")

        # C. Cartographie Multi-Echelles (Sémiologie Unifiée)
        if gdf_communes is not None:
            # Préparation de la palette de couleurs fixe
            cmap = plt.get_cmap("tab20")
            # On associe chaque cluster ID à une couleur unique
            cluster_colors = {i: cmap(i % 20) for i in range(n_clusters)}

            # Légende manuelle
            legend_patches = [
                mpatches.Patch(color=cluster_colors[i], label=f"C{i}")
                for i in range(n_clusters)
            ]

            # Jointure géométrique globale
            gdf_global = gdf_communes.merge(
                df_final[["code_insee", col_cluster]],
                left_on="code",
                right_on="code_insee",
                how="left",
            )

            # Attribution de la couleur
            gdf_global["color"] = (
                gdf_global[col_cluster].map(cluster_colors).fillna("#e0e0e0")
            )

            # Boucle sur les zones (France + Zooms)
            for zone_name, depts in ZONES_ZOOM.items():
                print(f"   -> Génération carte {zone_name}...")

                if depts is None:
                    # France Entière
                    gdf_plot = gdf_global
                    linewidth = 0.05  # Très fin pour la France
                    figsize = (16, 14)
                else:
                    # Zoom Départemental
                    gdf_plot = gdf_global[gdf_global["dept"].isin(depts)]
                    linewidth = 0.3  # Plus épais pour les zooms
                    figsize = (12, 10)

                if len(gdf_plot) == 0:
                    print(f"      ⚠️ Zone vide : {zone_name}")
                    continue

                fig, ax = plt.subplots(figsize=figsize)
                gdf_plot.plot(
                    color=gdf_plot["color"],
                    linewidth=linewidth,
                    edgecolor="white",
                    ax=ax,
                )

                # Légende (uniquement pour la France ou si besoin)
                if zone_name == "France" or zone_name == "Paris_PC":
                    ax.legend(
                        handles=legend_patches,
                        loc="upper left",
                        bbox_to_anchor=(1, 1),
                        title=f"Clusters (k={n_clusters})",
                        frameon=False,
                        ncol=2,
                    )

                ax.set_axis_off()
                ax.set_title(f"Typologie {n_clusters} classes - {zone_name}")

                out_png = os.path.join(OUT_DIR, f"carte_{zone_name}_k{n_clusters}.png")
                plt.savefig(out_png, dpi=200, bbox_inches="tight")
                plt.close()

    # 3. Export Final
    df_final.to_csv(os.path.join(OUT_DIR, "resultats_complets.csv"), index=False)
    print("\n✅ Terminé. Toutes les cartes sont dans 'out/'.")


if __name__ == "__main__":
    analyze()
