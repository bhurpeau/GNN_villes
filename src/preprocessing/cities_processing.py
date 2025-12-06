import pandas as pd
import numpy as np
from .data_io import load_parquet_data, load_csv_data, save_parquet_data


def map_plm(code):
    """Aggrège les arrondissements vers la commune mère"""
    # Conversion en string au cas où
    code = str(code)
    if code.startswith("751") and len(code) == 5:
        return "75056"  # Paris
    if code.startswith("6938") and len(code) == 5:
        return "69123"  # Lyon
    if code.startswith("132") and len(code) == 5:
        return "13055"  # Marseille
    return code


def process_macro_flows(
    dad_path, dt_path, pop_path, acti_path, output_nodes, output_edges
):
    print("--- PRÉ-TRAITEMENT DES FLUX (PLM + FILTRE RELATIF) ---")

    # 1. Chargement
    dad = load_parquet_data(dad_path)  # Migrations
    dt = load_parquet_data(dt_path)  # Navettes
    pop = load_csv_data(pop_path)  # Stock Pop
    acti = load_csv_data(acti_path)  # Stock Actifs
    dad.columns = ["pop21", "code_a", "code"]
    dt.columns = ["pop21", "code", "code_a"]

    print("Conversion des types numériques...")
    dad["pop21"] = dad["pop21"].astype(float)
    dt["pop21"] = dt["pop21"].astype(float)
    # ------------------------------------------

    # 2. Correction PLM sur les Flux
    print("Aggregation PLM des flux...")
    for df in [dad, dt]:
        df["code"] = df["code"].astype(str).apply(map_plm)
        df["code_a"] = df["code_a"].astype(str).apply(map_plm)

    # Somme après mapping PLM
    dad_edges = dad.groupby(["code", "code_a"]).pop21.sum().reset_index()
    dt_edges = dt.groupby(["code", "code_a"]).pop21.sum().reset_index()

    # 3. Préparation des Stocks (Dénominateurs) avec correction PLM
    print("Préparation des stocks de population...")

    # Population (pour Migrations)
    pop = pop[["COM", "PMUN"]].copy()
    pop.columns = ["code", "stock_pop"]

    # Nettoyage et conversion Stocks
    pop["stock_pop"] = (
        pd.to_numeric(pop["stock_pop"], errors="coerce").fillna(0).astype(float)
    )
    pop["code"] = pop["code"].astype(str).apply(map_plm)

    pop = pop.groupby("code")["stock_pop"].sum().reset_index()

    # Actifs (pour Navettes)
    acti = acti[["CODGEO", "P21_POP1564"]].copy()
    acti.columns = ["code", "stock_acti"]

    # Nettoyage et conversion Stocks
    acti["stock_acti"] = (
        pd.to_numeric(acti["stock_acti"], errors="coerce").fillna(0).astype(float)
    )
    acti["code"] = acti["code"].astype(str).apply(map_plm)

    acti = acti.groupby("code")["stock_acti"].sum().reset_index()

    # 4. Jointure et Filtrage "Intelligent"
    print("Filtrage Hybride (Absolu + Relatif)...")

    # A. Traitement Migrations (DAD)
    dad_edges = dad_edges.merge(pop, on="code", how="left")

    # Sécurité division par zéro
    dad_edges["stock_pop"] = dad_edges["stock_pop"].fillna(0)
    dad_edges["share"] = 0.0
    mask_pop = dad_edges["stock_pop"] > 0
    dad_edges.loc[mask_pop, "share"] = (
        dad_edges.loc[mask_pop, "pop21"] / dad_edges.loc[mask_pop, "stock_pop"]
    )

    # RÈGLE DE FILTRE : > 50 personnes OU > .5% de la pop (avec plancher de 3)
    mask_dad = (dad_edges["pop21"] > 50) | (
        (dad_edges["share"] > 0.005) & (dad_edges["pop21"] > 3)
    )
    dad_final = dad_edges[mask_dad].copy()

    dad_final["migra"] = dad_final["share"]
    dad_final = dad_final[["code", "code_a", "migra"]]

    # B. Traitement Navettes (DT)
    dt_edges = dt_edges.merge(acti, on="code", how="left")

    # Sécurité division par zéro
    dt_edges["stock_acti"] = dt_edges["stock_acti"].fillna(0)
    dt_edges["share"] = 0.0
    mask_acti = dt_edges["stock_acti"] > 0
    dt_edges.loc[mask_acti, "share"] = (
        dt_edges.loc[mask_acti, "pop21"] / dt_edges.loc[mask_acti, "stock_acti"]
    )

    # RÈGLE DE FILTRE : > 50 personnes OU > 5% des actifs (lien fort)
    mask_dt = (dt_edges["pop21"] > 50) | (
        (dt_edges["share"] > 0.05) & (dt_edges["pop21"] > 3)
    )
    dt_final = dt_edges[mask_dt].copy()

    dt_final["d_t"] = dt_final["share"]
    dt_final = dt_final[["code", "code_a", "d_t"]]

    # 5. Fusion Finale
    invalid = ["YYYYY", "99999", "ZZZZZ"]
    dt_final = dt_final[~dt_final["code_a"].isin(invalid)]
    dad_final = dad_final[~dad_final["code_a"].isin(invalid)]

    edges_final = dt_final.merge(dad_final, on=["code", "code_a"], how="outer").fillna(
        0
    )

    # 6. Extraction Nœuds (Diagonale)
    flux_internes = edges_final[edges_final["code"] == edges_final["code_a"]].copy()
    flux_internes = flux_internes.rename(
        columns={"d_t": "macro_taux_retenue", "migra": "macro_taux_stabilite"}
    )

    print(f"Sauvegarde : {len(flux_internes)} communes avec attributs de flux.")
    save_parquet_data(
        flux_internes[["code", "macro_taux_retenue", "macro_taux_stabilite"]],
        output_nodes,
    )

    # 7. Nettoyage Arêtes (Exclure diagonale)
    edges_final = edges_final[edges_final["code"] != edges_final["code_a"]]

    print(f"Nombre d'arêtes conservées : {len(edges_final)}")

    save_parquet_data(edges_final, output_edges)
    return flux_internes, edges_final
