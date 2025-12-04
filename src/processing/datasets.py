import torch
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


def merge_macro_edges(phys_index, phys_attr, flux_index, flux_attr):
    """
    Fusionne les arêtes physiques (non-dirigées) et fonctionnelles (dirigées).
    Sortie : Graphe dirigé unifié avec attributs de dim 4 : [Len, Perm, Navette, Migra]
    """
    # 1. Physique (Symétrique) : On duplique pour rendre bidirectionnel (A->B et B->A)
    phys_index_bi = torch.cat([phys_index, phys_index.flip(0)], dim=1)
    phys_attr_bi = torch.cat([phys_attr, phys_attr], dim=0)  # [N_phys*2, 2]

    # 2. Flux (Asymétrique) : On garde tel quel (Source -> Cible)
    # flux_attr est déjà [N_flux, 2]

    # 3. Padding (Remplissage des dimensions manquantes)
    # Pour les arêtes physiques : Flux = 0
    zeros_flux = torch.zeros((phys_attr_bi.size(0), 2))
    attr_phys_final = torch.cat([phys_attr_bi, zeros_flux], dim=1)

    # Pour les arêtes de flux : Physique = 0
    zeros_phys = torch.zeros((flux_attr.size(0), 2))
    attr_flux_final = torch.cat([zeros_phys, flux_attr], dim=1)

    # 4. Concaténation Finale
    final_edge_index = torch.cat([phys_index_bi, flux_index], dim=1)
    final_edge_attr = torch.cat([attr_phys_final, attr_flux_final], dim=0)

    return final_edge_index, final_edge_attr


class FranceHierarchicalDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "statistiques_carreaux.parquet.gz",
            "edges_toutes_communes.npy",
            "nodes_macro_attributes.parquet",
            "graph_macro_physique.pt",
            "edges_macro_flux.parquet",
        ]

    @property
    def processed_file_names(self):
        return ["data_hierarchique_france.pt"]

    def process(self):
        print("--- DÉBUT DU TRAITEMENT ---")

        # 1. CHARGEMENT MICRO (Carreaux)
        print("Chargement Micro...")
        df_micro = pd.read_parquet(self.raw_paths[0])

        # [cite_start]Features Invariantes (Structure) [cite: 69-71]
        cols_invariant = [
            "z_mean",
            "z_std",
            "slope_mean",
            "struct_bati",
            "struct_eco",
            "struct_nature",
            "struct_agri",
            "struct_eau",
            "struct_glacier",
        ]

        # [cite_start]Features Variantes (Social) + QUALITÉ (is_imputed) [cite: 72-73]
        cols_variant = [
            "densite_pop",
            "croissance_pop",
            "niveau_vie_moyen",
            "taux_pauvrete",
            "part_proprio",
            "part_maison",
            "part_hlm",
            "is_imputed",
        ]

        x_invariant = torch.tensor(df_micro[cols_invariant].values, dtype=torch.float)
        x_variant = torch.tensor(df_micro[cols_variant].values, dtype=torch.float)

        # 2. STANDARDISATION (Calcul sur tout le dataset)

        # A. Invariants
        mean_inv = x_invariant.mean(dim=0)
        std_inv = x_invariant.std(dim=0)
        std_inv[std_inv == 0] = 1.0
        x_inv_scaled = (x_invariant - mean_inv) / std_inv

        # B. Variants (Attention au flag is_imputed)
        x_var_continuous = x_variant[:, :-1]
        x_var_flag = x_variant[:, -1:]  # colonne is_imputed

        mean_var = x_var_continuous.mean(dim=0)
        std_var = x_var_continuous.std(dim=0)
        std_var[std_var == 0] = 1.0

        x_var_scaled = (x_var_continuous - mean_var) / std_var
        x_variant_final = torch.cat([x_var_scaled, x_var_flag], dim=1)

        x_micro = torch.cat([x_inv_scaled, x_variant_final], dim=1)

        # 3. SAUVEGARDE DES STATS
        stats = {
            "mean_inv": mean_inv,
            "std_inv": std_inv,
            "mean_var": mean_var,
            "std_var": std_var,
        }
        torch.save(stats, self.processed_paths[0] + ".stats")

        # Découpage par commune
        edges_global = np.load(self.raw_paths[1])
        edge_index_global = torch.tensor(edges_global, dtype=torch.long)

        commune_groups = df_micro.groupby("code")
        sorted_codes = sorted(list(commune_groups.groups.keys()))

        data_list_micro = []

        print(f"Découpage des {len(sorted_codes)} communes...")
        for code in tqdm(sorted_codes):
            indices = commune_groups.get_group(code).index.values
            min_idx, max_idx = indices.min(), indices.max()

            # Extraction locale
            local_x = x_micro[min_idx : max_idx + 1]
            mask = (edge_index_global[0] >= min_idx) & (edge_index_global[0] <= max_idx)
            local_edges = edge_index_global[:, mask] - min_idx

            data_commune = Data(x=local_x, edge_index=local_edges)
            data_list_micro.append(data_commune)

        # 2. CHARGEMENT MACRO (Communes)
        print("Chargement Macro...")

        # Attributs Nodaux
        df_macro = pd.read_parquet(self.raw_paths[2])
        df_macro = df_macro.set_index("code").loc[sorted_codes].reset_index()
        x_macro = torch.tensor(
            df_macro[["macro_taux_retenue", "macro_taux_stabilite"]].values,
            dtype=torch.float,
        )

        # [cite_start]Gestion des Arêtes (Physique + Flux) [cite: 80-84]
        # Physique
        graph_phys = torch.load(self.raw_paths[3])

        # Flux
        df_flux = pd.read_parquet(self.raw_paths[4])
        map_commune = {code: i for i, code in enumerate(sorted_codes)}
        src = torch.tensor(
            df_flux["source_code"].map(map_commune).values, dtype=torch.long
        )
        dst = torch.tensor(
            df_flux["target_code"].map(map_commune).values, dtype=torch.long
        )
        edge_index_flux = torch.stack([src, dst], dim=0)
        edge_attr_flux = torch.tensor(
            df_flux[["d_t", "migra"]].values, dtype=torch.float
        )

        # FUSION
        final_index, final_attr = merge_macro_edges(
            graph_phys["edge_index"],
            graph_phys["edge_attr"],
            edge_index_flux,
            edge_attr_flux,
        )

        data_macro = Data(x=x_macro, edge_index=final_index, edge_attr=final_attr)

        # Sauvegarde : Macro en premier, puis liste Micro
        torch.save((data_macro, data_list_micro), self.processed_paths[0])
        print("Terminé.")
