import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_RAW = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "out")

# Fichiers d'entrée (On reprend les résultats existants)
RESULTS_CSV = os.path.join(OUT_DIR, "resultats_complets.csv")
COMMUNES_GPKG = os.path.join(DATA_RAW, "commune_francemetro_2023.gpkg")

# Zone à zoomer : Paris (75) + Petite Couronne (92, 93, 94)
DEPTS_CIBLES = ["75", "92", "93", "94"]


def make_zoom():
    print("--- GÉNÉRATION CARTE ZOOM PARIS & PETITE COURONNE ---")

    if not os.path.exists(RESULTS_CSV):
        print("❌ Lancez d'abord analysis.py pour générer resultats_complets.csv")
        return

    # 1. Chargement des données
    print("Chargement des résultats...")
    df_res = pd.read_csv(
        RESULTS_CSV, dtype={"code_insee": str}
    )  # Force str pour le code

    print("Chargement du fond de carte...")
    gdf = gpd.read_file(COMMUNES_GPKG)

    # 2. Filtrage Géographique
    # On extrait le numéro de département (2 premiers chiffres du code INSEE)
    gdf["dept"] = gdf["code"].str[:2]
    gdf_zoom = gdf[gdf["dept"].isin(DEPTS_CIBLES)].copy()

    print(f"   -> {len(gdf_zoom)} communes sélectionnées.")

    # 3. Jointure
    gdf_map = gdf_zoom.merge(df_res, left_on="code", right_on="code_insee", how="left")

    # 4. Cartographie (Loop sur 6 et 12 clusters)
    for k in [6, 12, 18]:
        col_cluster = f"cluster_{k}"
        if col_cluster not in gdf_map.columns:
            continue

        print(f"Génération carte k={k}...")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Fond gris clair pour les communes manquantes (si filtre >50 flux)
        gdf_zoom.plot(ax=ax, color="#f0f0f0", edgecolor="white")

        # Carte des clusters
        plot = gdf_map.plot(
            column=col_cluster,
            categorical=True,
            cmap="tab20",  # Couleurs distinctes
            linewidth=0.5,
            edgecolor="grey",
            legend=True,
            ax=ax,
            legend_kwds={
                "loc": "upper left",
                "bbox_to_anchor": (1, 1),
                "fmt": "{:.0f}",
            },
        )

        ax.set_axis_off()
        ax.set_title(f"Typologie Morpho-Dynamique - Zoom Paris ({k} classes)")

        # Sauvegarde
        out_path = os.path.join(OUT_DIR, f"zoom_paris_k{k}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"   ✅ Sauvegardé : {out_path}")
        plt.close()


if __name__ == "__main__":
    make_zoom()
