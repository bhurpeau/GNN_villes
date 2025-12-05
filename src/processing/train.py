import torch
import torch.nn.functional as F
import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Import de vos modules précédents
from .dataset import FranceHierarchicalDataset
from .model import HierarchicalGNN

# --- CHEMINS ROBUSTES ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data_GNN")

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_MICRO = 64  # Nombre de communes par batch
EPOCHS = 20
LR = 0.001
MASK_RATE = 0.15

# Indices des colonnes dans le tenseur x (selon dataset.py)
IDX_INVARIANT_START = 0
IDX_INVARIANT_END = 10
IDX_VARIANT_START = 10
IDX_VARIANT_END = 18


def mask_features(x, mask_rate):
    """
    Masque aléatoire pour l'auto-supervision.
    Retourne : x_masked (entrée), mask (booléens), x_target (vérité terrain)
    """
    mask = torch.rand(x.size()) < mask_rate
    x_masked = x.clone()
    x_masked[mask] = 0.0  # On remplace par 0 (ou bruit gaussien)
    return x_masked, mask, x


def mask_variant_features(x, mask_rate=0.5):
    """
    Masque uniquement les variables sociales.
    On laisse la structure physique intacte car elle sert de point d'appui.
    """
    x_masked = x.clone()

    # Création du masque uniquement pour la partie variante
    # Shape [N_nodes, 8]
    variant_part = x[:, IDX_VARIANT_START:IDX_VARIANT_END]
    mask = torch.rand(variant_part.size()) < mask_rate

    x_masked[:, IDX_VARIANT_START:IDX_VARIANT_END][mask] = 0.0

    return x_masked, mask


def train():
    print(f"Initialisation sur {DEVICE}...")

    # 1. Chargement Données
    dataset = FranceHierarchicalDataset(root=DATA_ROOT)

    # On sépare Macro (élément 0) et la liste Micro (éléments 1 à fin)
    data_full = torch.load(dataset.processed_paths[0], weights_only=False)
    data_macro = data_full[0].to(DEVICE)
    data_list_micro = data_full[1]  # Liste des objets Data communes

    # Loader pour itérer sur les communes (niveau Micro)
    loader = DataLoader(data_list_micro, batch_size=BATCH_SIZE_MICRO, shuffle=True)

    # 2. Modèle & Optimiseur
    model = HierarchicalGNN(
        micro_input_dim=18, macro_input_dim=3, latent_dim=32, social_output_dim=8
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Si la loss stagne pendant 3 époques, on divise le LR par 2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    # 3. BUFFER GLOBAL Z_MORPHO
    num_communes = data_macro.x.size(0)
    z_global_buffer = torch.zeros((num_communes, 32)).to(DEVICE)
    z_global_buffer.requires_grad = False

    print("Début de l'entraînement...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        # Barre de progression
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_micro in pbar:
            batch_micro = batch_micro.to(DEVICE)
            optimizer.zero_grad()
            batch_micro.x_raw = batch_micro.x.clone()
            # 1. MASQUAGE INTELLIGENT
            x_input, mask_boolean = mask_variant_features(batch_micro.x, MASK_RATE)
            batch_micro.x = x_input

            # 2. FORWARD PASS (Micro -> Buffer -> Macro)
            z_batch = model.forward_micro(batch_micro)

            global_indices = batch_micro.global_idx
            z_context = z_global_buffer.clone()
            z_context[global_indices] = z_batch

            z_final = model.forward_macro(z_context, data_macro)

            # 3. RECONSTRUCTION
            z_batch_final = z_final[global_indices]

            social_reconstruction = model.decode(z_batch_final)

            # 4. PRÉPARATION Ground Truth
            target_social_list = []
            for i in range(len(global_indices)):
                mask_commune = batch_micro.batch == i
                real_profile = batch_micro.x_raw[
                    mask_commune, IDX_VARIANT_START:IDX_VARIANT_END
                ].mean(dim=0)
                target_social_list.append(real_profile)

            target_social = torch.stack(target_social_list)  # Shape [Batch, 8]

            # 5. CALCUL DE LA LOSS
            loss = F.huber_loss(social_reconstruction, target_social, delta=1.0)

            loss.backward()
            optimizer.step()
            z_global_buffer[global_indices] = z_batch.detach()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # Fin d'époque : Step du Scheduler
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        # Fin d'époque : Sauvegarde checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/checkpoint_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()
