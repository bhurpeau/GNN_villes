import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Import de vos modules précédents
from dataset import FranceHierarchicalDataset
from model import HierarchicalGNN

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_MICRO = 64  # Nombre de communes par batch
EPOCHS = 20
LR = 0.001
MASK_RATE = 0.15  # 15% des features sont masquées (comme BERT)


def mask_features(x, mask_rate):
    """
    Masque aléatoire pour l'auto-supervision.
    Retourne : x_masked (entrée), mask (booléens), x_target (vérité terrain)
    """
    mask = torch.rand(x.size()) < mask_rate
    x_masked = x.clone()
    x_masked[mask] = 0.0  # On remplace par 0 (ou bruit gaussien)
    return x_masked, mask, x


def train():
    print(f"Initialisation sur {DEVICE}...")

    # 1. Chargement Données
    dataset = FranceHierarchicalDataset(root="./data_GNN")

    # On sépare Macro (élément 0) et la liste Micro (éléments 1 à fin)
    data_full = torch.load(dataset.processed_paths[0])
    data_macro = data_full[0].to(DEVICE)
    data_list_micro = data_full[1]  # Liste des objets Data communes

    # Loader pour itérer sur les communes (niveau Micro)
    loader = DataLoader(data_list_micro, batch_size=BATCH_SIZE_MICRO, shuffle=True)

    # 2. Modèle & Optimiseur
    model = HierarchicalGNN(micro_input_dim=17, macro_input_dim=2, latent_dim=32).to(
        DEVICE
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3. BUFFER GLOBAL Z_MORPHO
    # Astuce : On stocke les embeddings de TOUTES les communes (35k) ici.
    # Au début, c'est des zéros (ou bruit). Il va s'affiner à chaque époque.
    num_communes = data_macro.x.size(0)
    z_global_buffer = torch.zeros((num_communes, 32)).to(DEVICE)
    # On détache le gradient pour l'historique (pour ne pas exploser la mémoire)
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

            # --- A. Masquage (Auto-supervision) ---
            # On masque les features d'entrée du niveau Micro
            # Le but est de reconstruire x_target en utilisant la structure et le contexte
            x_masked, mask, x_target = mask_features(batch_micro.x, MASK_RATE)
            batch_micro.x = x_masked  # Injection des features masquées

            # --- B. Forward Micro ---
            # On calcule les signatures Z pour ce batch uniquement
            z_batch = model.forward_micro(batch_micro)

            # --- C. Intégration dans le Buffer (Le pont Micro-Macro) ---
            # batch_micro.code_insee n'est pas dispo directement dans le Batch PyG standard
            # Astuce : On utilise l'attribut 'batch' ou on suppose que le DataLoader
            # renvoie les données dans un ordre qu'on peut mapper.
            # ICI : Pour simplifier, supposons que data_list_micro a un attribut 'global_idx'
            # (Il faudra l'ajouter au dataset si absent, voir Note en bas)

            # Solution temporaire robuste : On utilise l'attribut ajouté manuellement
            # Supposons que dans Dataset.py on ait ajouté : data.global_idx = i
            global_indices = batch_micro.global_idx

            # Mise à jour du buffer :
            # On prend le buffer existant (détaché)
            z_context = z_global_buffer.clone()
            # On y insère nos nouveaux Z (qui gardent leur gradient !)
            z_context[global_indices] = z_batch

            # --- D. Forward Macro ---
            # On lance le GAT sur TOUT le graphe, mais le gradient ne passera
            # que par les nœuds du batch (grâce à z_batch)
            z_final = model.forward_macro(z_context, data_macro)

            # --- E. Reconstruction / Prédiction ---
            # On essaie de prédire les valeurs masquées des communes DU BATCH
            # Pour cela, on projette z_final vers l'espace des features initiales
            # (Nécessite d'ajouter un decodeur au modèle, ici simplifié)

            # Pour l'exemple, supposons qu'on veuille reconstruire une propriété globale de la commune
            # (ex: densité moyenne masquée) à partir de Z_final

            # Simplification : On utilise une loss sur l'embedding lui-même ou une tâche proxy
            # Tâche papier : Reconstruction. Ajoutons une tête de reconstruction temporaire.
            reconstruction = model.predictor(z_final[global_indices])

            # Cible : Par exemple, reconstruire la densité moyenne réelle (qui était dans x_target)
            # On extrait la densité moyenne cible pour chaque commune du batch
            # (Nécessite un pooling manuel rapide sur x_target)
            target_vals = []
            for i in range(len(global_indices)):
                mask_commune = batch_micro.batch == i
                # On vise la reconstruction de la 1ere variable variante (densité)
                val = x_target[mask_commune, 9].mean()
                target_vals.append(val)
            target_tensor = torch.stack(target_vals).unsqueeze(1)

            # --- F. Loss & Backprop ---
            loss = F.mse_loss(reconstruction, target_tensor)

            loss.backward()
            optimizer.step()

            # Mise à jour du buffer "offline" pour le prochain tour
            # (On détache pour casser le graphe de calcul)
            z_global_buffer[global_indices] = z_batch.detach()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # Fin d'époque : Sauvegarde checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()
