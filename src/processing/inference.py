import torch
import pandas as pd
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Import modules
from .dataset import FranceHierarchicalDataset
from .model import HierarchicalGNN

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# Chemins robustes
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data_GNN")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "out", "resultats_embeddings.parquet")

# Nom du checkpoint à charger (vérifiez le nom exact généré par train.py !)
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "checkpoint_epoch_20.pth")


def inference():
    print("--- GÉNÉRATION DES EMBEDDINGS ---")
    print(f"Chargement depuis : {DATA_ROOT}")

    # 1. Chargement Données
    dataset = FranceHierarchicalDataset(root=DATA_ROOT)

    # On utilise le fichier processed déjà généré par train.py
    # Attention : processed_paths[0] est un chemin absolu ou relatif géré par PyG
    data_full = torch.load(dataset.processed_paths[0], weights_only=False)

    data_macro = data_full[0].to(DEVICE)
    data_list_micro = data_full[1]

    # Loader (Shuffle=False pour garder l'ordre des communes !)
    loader = DataLoader(data_list_micro, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Chargement Modèle
    # On doit réinstancier exactement la même architecture
    model = HierarchicalGNN(
        micro_input_dim=18, macro_input_dim=3, latent_dim=32, social_output_dim=8
    ).to(DEVICE)

    # Chargement des poids
    if os.path.exists(MODEL_PATH):
        print(f"Chargement des poids depuis {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        # Fallback : cherche dans le dossier courant si pas trouvé dans src/processing
        fallback_path = "checkpoint_epoch_20.pth"
        if os.path.exists(fallback_path):
            print(f"Chargement des poids depuis {fallback_path}...")
            model.load_state_dict(torch.load(fallback_path, map_location=DEVICE, weights_only=True))
        else:
            print("⚠️ ALERTE : Checkpoint introuvable ! Vérifiez le chemin.")
            return

    model.eval()  # Mode évaluation (désactive le dropout)

    # 3. Préparation des buffers
    num_communes = data_macro.x.size(0)
    z_buffer = torch.zeros((num_communes, 32)).to(DEVICE)

    # --- PASSE 1 : Calculer tous les Z_morpho (Micro) ---
    print("Calcul des signatures morphologiques (Micro)...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(DEVICE)
            # On passe dans le GCN Micro
            z_batch = model.forward_micro(batch)
            # On stocke au bon endroit grâce à global_idx
            z_buffer[batch.global_idx] = z_batch

    # --- PASSE 2 : Contextualiser (Macro) ---
    print("Contextualisation macroscopique (GAT)...")
    with torch.no_grad():
        # On lance le GAT sur tout le graphe d'un coup avec le buffer complet
        z_final_all = model.forward_macro(z_buffer, data_macro)

    # 4. Export des Résultats
    print("Sauvegarde...")

    # Récupération des codes INSEE pour faire la jointure
    # On recharge le fichier source pour être sûr de l'ordre (trié par code dans dataset.py)
    df_micro = pd.read_parquet(os.path.join(DATA_ROOT, "statistiques_carreaux.parquet.gz"))
    sorted_codes = sorted(list(df_micro["code"].unique()))

    if len(sorted_codes) != num_communes:
        print(f"⚠️ Mismatch : {len(sorted_codes)} codes vs {num_communes} nœuds graph.")

    # Conversion en Numpy
    z_morpho_np = z_buffer.cpu().numpy()
    z_final_np = z_final_all.cpu().numpy()

    # Création du DataFrame final
    df_res = pd.DataFrame({"code_insee": sorted_codes})

    # Astuce : on sauvegarde les vecteurs sous forme de listes dans Parquet
    # C'est plus facile à gérer ensuite qu'avoir 64 colonnes
    df_res["z_morpho"] = list(z_morpho_np)
    df_res["z_final"] = list(z_final_np)

    # Sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_res.to_parquet(OUTPUT_FILE)
    print(f"✅ Terminé ! Fichier généré : {OUTPUT_FILE}")
    print("   -> Vous pouvez l'ouvrir avec Pandas ou QGIS (via plugin).")


if __name__ == "__main__":
    inference()
