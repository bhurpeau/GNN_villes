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

# Zones de zoom
ZONES_ZOOM = {
    "France": None,
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
    print("--- ANALYSE STATISTIQUE (MÉTHODE MIXTE) & CARTOGRAPHIE ---")

    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"❌ Erreur : {EMBEDDINGS_FILE} manquant.")
        return

    # 1. Chargement
    print("Chargement des données...")
    df_emb = pd.read_parquet(EMBEDDINGS_FILE)
    df_micro = pd.read_parquet(STATS_MICRO)
    df_macro = pd.read_parquet(STATS_MACRO)

    if os.path.exists(COMMUNES_GPKG):
        gdf_communes = gpd.read_file(COMMUNES_GPKG)[["code", "geometry"]]
        gdf_communes["dept"] = gdf_communes["code"].str[:2]
    else:
        gdf_communes = None

    # Agrégation Micro -> Commune (Pondérée)
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

    # Fusion
    df_final = df_emb.merge(
        commune_stats, left_on="code_insee", right_on="code", how="left"
    )
    df_final = df_final.merge(
        df_macro, left_on="code_insee", right_on="code", how="left"
    )

    for col in ["macro_taux_retenue", "macro_taux_stabilite"]:
        df_final[col] = df_final[col].fillna(df_final[col].mean())

    # Clustering
    X = np.stack(df_final["z_final"].values)
    X_scaled = StandardScaler().fit_transform(X)

    # 2. Boucle d'analyse
    for n_clusters in [6, 12, 18]:
        print(f"\n--- Traitement k={n_clusters} ---")

        cah = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        clusters = cah.fit_predict(X_scaled)
        col_cluster = f"cluster_{n_clusters}"
        df_final[col_cluster] = clusters

        # --- CALCUL DES PROFILS (Correction Scientifique) ---
        def get_profile(grp):
            pop = grp["TOT_P_2021"]
            ind = grp["ind"]
            w_pop = pop.sum()
            w_ind = ind.sum()

            return pd.Series(
                {
                    # Revenu : PONDÉRÉ (Pour refléter la richesse réelle, on évite les biais des micro-villages riches)
                    "niveau_vie_moyen": (
                        np.average(grp["niveau_vie_moyen"], weights=ind)
                        if w_ind > 0
                        else 0
                    ),
                    # Densité : PONDÉRÉ (Densité vécue : la densité ressentie par l'habitant moyen)
                    "densite_pop": (
                        np.average(grp["densite_pop"], weights=pop) if w_pop > 0 else 0
                    ),
                    # Structure : MOYENNE SIMPLE (Portrait robot du territoire physique)
                    "struct_bati": grp["struct_bati"].mean(),
                    "struct_nature": grp["struct_nature"].mean(),
                    "struct_agri": grp["struct_agri"].mean(),
                    # Flux (Macro) : MOYENNE SIMPLE (Portrait robot du rôle fonctionnel de la commune)
                    # C'est ici que le changement impacte : on donne le même poids à chaque ville
                    # pour identifier le "rôle type" (ex: pôle relais) indépendamment de sa taille.
                    "macro_taux_retenue": grp["macro_taux_retenue"].mean(),
                    "macro_taux_stabilite": grp["macro_taux_stabilite"].mean(),
                    "nb_communes": len(grp),
                    "pop_totale": w_pop,
                }
            )

        profile = df_final.groupby(col_cluster).apply(get_profile)

        # Sauvegarde
        profile_path = os.path.join(OUT_DIR, f"profils_clusters_{n_clusters}.csv")
        profile.T.to_csv(profile_path)
        print(f"   ✅ Profils (Méthode Mixte) sauvegardés : {profile_path}")

        # Cartographie (Code identique)
        if gdf_communes is not None:
            cmap = plt.get_cmap("tab20")
            cluster_colors = {i: cmap(i % 20) for i in range(n_clusters)}
            legend_patches = [
                mpatches.Patch(color=cluster_colors[i], label=f"C{i}")
                for i in range(n_clusters)
            ]

            gdf_global = gdf_communes.merge(
                df_final[["code_insee", col_cluster]],
                left_on="code",
                right_on="code_insee",
                how="left",
            )
            gdf_global["color"] = (
                gdf_global[col_cluster].map(cluster_colors).fillna("#e0e0e0")
            )

            for zone_name, depts in ZONES_ZOOM.items():
                if depts is None:
                    gdf_plot = gdf_global
                    lw = 0.05
                    figsize = (16, 14)
                else:
                    gdf_plot = gdf_global[gdf_global["dept"].isin(depts)]
                    lw = 0.3
                    figsize = (12, 10)

                if len(gdf_plot) > 0:
                    fig, ax = plt.subplots(figsize=figsize)
                    gdf_plot.plot(
                        color=gdf_plot["color"], linewidth=lw, edgecolor="white", ax=ax
                    )
                    if zone_name == "France" or zone_name == "Paris_PC":
                        ax.legend(
                            handles=legend_patches,
                            loc="upper left",
                            bbox_to_anchor=(1, 1),
                            frameon=False,
                            ncol=2,
                        )
                    ax.set_axis_off()
                    ax.set_title(f"Typologie {n_clusters} classes - {zone_name}")
                    plt.savefig(
                        os.path.join(OUT_DIR, f"carte_{zone_name}_k{n_clusters}.png"),
                        dpi=200,
                        bbox_inches="tight",
                    )
                    plt.close()

    df_final.to_csv(os.path.join(OUT_DIR, "resultats_complets.csv"), index=False)
    print("\n✅ Terminé.")


if __name__ == "__main__":
    analyze()
