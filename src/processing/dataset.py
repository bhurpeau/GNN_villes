import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


def merge_macro_edges(phys_index, phys_attr, flux_index, flux_attr):
    """Fusionne les arêtes physiques et fonctionnelles."""
    # 1. Physique (Symétrique)
    phys_index_bi = torch.cat([phys_index, phys_index.flip(0)], dim=1)
    phys_attr_bi = torch.cat([phys_attr, phys_attr], dim=0)

    # 2. Padding
    zeros_flux = torch.zeros((phys_attr_bi.size(0), 2))
    attr_phys_final = torch.cat([phys_attr_bi, zeros_flux], dim=1)

    zeros_phys = torch.zeros((flux_attr.size(0), 2))
    attr_flux_final = torch.cat([zeros_phys, flux_attr], dim=1)

    # 3. Concaténation
    final_edge_index = torch.cat([phys_index_bi, flux_index], dim=1)
    final_edge_attr = torch.cat([attr_phys_final, attr_flux_final], dim=0)

    return final_edge_index, final_edge_attr


class FranceHierarchicalDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

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

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    def process(self):
        print("--- DÉBUT DU TRAITEMENT (VERSION ROBUSTE) ---")

        # 1. CHARGEMENT MICRO & TRI
        print("Chargement et tri des carreaux...")
        df_micro = pd.read_parquet(self.raw_paths[0])

        # --- CORRECTION CRITIQUE 1 : TRI ET REMAPPING ---
        # On trie par code pour que les carreaux d'une commune soient contigus
        df_micro["old_idx"] = np.arange(len(df_micro))  # On garde la trace
        df_micro = df_micro.sort_values("code").reset_index(drop=True)

        # Création du vecteur de mapping : Old_ID -> New_ID
        # map_arr[old] = new
        map_arr = np.zeros(len(df_micro), dtype=np.int64)
        map_arr[df_micro["old_idx"].values] = np.arange(len(df_micro))

        # Chargement et mise à jour des arêtes globales
        edges_global = np.load(self.raw_paths[1])
        print("Ré-indexation des arêtes...")
        # On traduit les anciens indices vers les nouveaux (post-tri)
        edges_global = map_arr[edges_global]
        edge_index_global = torch.tensor(edges_global, dtype=torch.long)
        # ------------------------------------------------

        # Features Invariantes
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
            "log_amenity_density",
        ]
        # Features Variantes
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

        x_inv = torch.tensor(df_micro[cols_invariant].values, dtype=torch.float)
        x_var = torch.tensor(df_micro[cols_variant].values, dtype=torch.float)

        # STANDARDISATION
        print("Standardisation...")
        mean_inv = x_inv.mean(dim=0)
        std_inv = x_inv.std(dim=0)
        std_inv[std_inv == 0] = 1.0
        x_inv = (x_inv - mean_inv) / std_inv

        x_var_cont = x_var[:, :-1]
        x_var_flag = x_var[:, -1:]
        mean_var = x_var_cont.mean(dim=0)
        std_var = x_var_cont.std(dim=0)
        std_var[std_var == 0] = 1.0
        x_var = torch.cat([(x_var_cont - mean_var) / std_var, x_var_flag], dim=1)

        x_micro = torch.cat([x_inv, x_var], dim=1)

        # SAUVEGARDE STATS
        stats = {
            "mean_inv": mean_inv,
            "std_inv": std_inv,
            "mean_var": mean_var,
            "std_var": std_var,
        }
        torch.save(stats, self.processed_paths[0] + ".stats")

        # DÉCOUPAGE PAR COMMUNE
        commune_groups = df_micro.groupby("code")
        sorted_codes = sorted(list(commune_groups.groups.keys()))
        data_list_micro = []

        print(f"Création des {len(sorted_codes)} sous-graphes...")
        for i, code in enumerate(tqdm(sorted_codes)):
            indices = commune_groups.get_group(code).index.values
            min_idx, max_idx = indices.min(), indices.max()

            # Grâce au tri, min_idx et max_idx définissent un bloc contigu STRICT
            local_x = x_micro[min_idx: max_idx + 1]

            # Filtre rapide sur les arêtes (Optimisé)
            # On suppose que les arêtes sont intra-communales, donc si source est dans [min, max], cible aussi
            mask = (edge_index_global[0] >= min_idx) & (edge_index_global[0] <= max_idx)
            local_edges = edge_index_global[:, mask] - min_idx

            data_commune = Data(x=local_x, edge_index=local_edges)
            data_commune.global_idx = torch.tensor([i], dtype=torch.long)
            data_list_micro.append(data_commune)

        # 2. CHARGEMENT MACRO
        print("Chargement Macro...")
        df_macro = pd.read_parquet(self.raw_paths[2])

        # --- CORRECTION CRITIQUE 2 : ROBUSTESSE MACRO ---
        df_macro = df_macro.set_index("code")
        # On force l'index à correspondre à la liste Micro, en remplissant les trous par la moyenne
        df_macro = df_macro.reindex(sorted_codes)
        df_macro = df_macro.fillna(df_macro.mean(numeric_only=True))
        df_macro = df_macro.reset_index()
        # ------------------------------------------------

        x_macro = torch.tensor(
            df_macro[["macro_taux_retenue", "macro_taux_stabilite", "nb_equip_structurants"]].values,
            dtype=torch.float,
        )

        # ARÊTES MACRO
        graph_phys = torch.load(self.raw_paths[3], weights_only=False)
        phys_edge_index = graph_phys["edge_index"]
        phys_edge_attr = graph_phys["edge_attr"]
        phys_mapping = graph_phys["mapping"]
        map_commune = {code: i for i, code in enumerate(sorted_codes)}
        print("Remapping du graphe physique...")

        # On crée un vecteur de traduction : Old_ID -> New_ID
        # On initialise à -1 pour détecter les communes hors périmètre
        max_old_id = max(phys_mapping.keys()) if phys_mapping else 0
        translator = torch.full((max_old_id + 1,), -1, dtype=torch.long)

        for old_idx, code in phys_mapping.items():
            if code in map_commune:
                translator[old_idx] = map_commune[code]

        # Traduction des sources et destinations
        src_old, dst_old = phys_edge_index[0], phys_edge_index[1]
        src_new = translator[src_old]
        dst_new = translator[dst_old]

        # Filtrage : On ne garde que les liens où les DEUX bouts existent dans notre dataset
        mask_valid = (src_new != -1) & (dst_new != -1)

        phys_index_clean = torch.stack([src_new[mask_valid], dst_new[mask_valid]], dim=0)
        phys_attr_clean = phys_edge_attr[mask_valid]

        df_flux = pd.read_parquet(self.raw_paths[4])

        df_flux = df_flux[df_flux["code"].isin(map_commune) & df_flux["code_a"].isin(map_commune)]

        src_flux = torch.tensor(df_flux["code"].map(map_commune).values, dtype=torch.long)
        dst_flux = torch.tensor(df_flux["code_a"].map(map_commune).values, dtype=torch.long)
        edge_index_flux = torch.stack([src_flux, dst_flux], dim=0)
        edge_attr_flux = torch.tensor(df_flux[["d_t", "migra"]].values, dtype=torch.float)

        # FUSION FINALE (Avec les version nettoyées)
        final_index, final_attr = merge_macro_edges(
            phys_index_clean,
            phys_attr_clean,  # <--- On utilise les versions clean
            edge_index_flux,
            edge_attr_flux,
        )

        data_macro = Data(x=x_macro, edge_index=final_index, edge_attr=final_attr)

        torch.save((data_macro, data_list_micro), self.processed_paths[0])
        print("✅ Traitement terminé avec succès.")
