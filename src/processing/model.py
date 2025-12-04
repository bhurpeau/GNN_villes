import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_max_pool


class HierarchicalGNN(nn.Module):
    def __init__(
        self,
        micro_input_dim=17,  # 9 Invariants + 8 Variants (dont is_imputed)
        macro_input_dim=2,  # Taux retenue + Stabilité
        hidden_dim=64,
        latent_dim=32,  # Taille du vecteur Z_morpho
        num_heads=4,
    ):
        super(HierarchicalGNN, self).__init__()

        # --- NIVEAU MICRO (GCN) ---
        # [cite_start]Lissage de l'information intra-communale [cite: 105-107]
        self.micro_conv1 = GCNConv(micro_input_dim, hidden_dim)
        self.micro_conv2 = GCNConv(hidden_dim, hidden_dim)

        # [cite_start]Readout: Compression du sous-graphe en un vecteur [cite: 112-113]
        self.micro_readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 car Mean + Max
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # --- NIVEAU MACRO (GAT) ---
        # Entrée Macro = Signature Micro (Z_morpho) + Attributs Macro
        gnn_macro_in = latent_dim + macro_input_dim
        # edge_dim=4 correspond à [Len, Perm, Navette, Migra]
        self.macro_gat1 = GATv2Conv(
            gnn_macro_in, hidden_dim, heads=num_heads, edge_dim=4, concat=True
        )
        self.macro_gat2 = GATv2Conv(
            hidden_dim * num_heads, latent_dim, heads=1, edge_dim=4, concat=False
        )
        # Tête de prédiction (pour l'auto-supervision)
        self.predictor = nn.Linear(latent_dim, 1)

    def forward_micro(self, data_micro):
        """Étape 1 : Calculer Z_morpho pour un batch de communes"""
        x, edge_index, batch = data_micro.x, data_micro.edge_index, data_micro.batch
        # Convolution
        x = F.relu(self.micro_conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.micro_conv2(x, edge_index)
        # Pooling (Mean + Max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        # Projection
        z_morpho = self.micro_readout(torch.cat([x_mean, x_max], dim=1))
        return z_morpho

    def forward_macro(self, z_morpho, data_macro):
        """Étape 2 : Contextualiser dans le graphe national"""
        # Concaténation [Signature Interne || Attributs Externe]
        x_combined = torch.cat([z_morpho, data_macro.x], dim=1)
        # Attention
        x = self.macro_gat1(
            x_combined, data_macro.edge_index, edge_attr=data_macro.edge_attr
        )
        x = F.elu(x)
        z_final = self.macro_gat2(
            x, data_macro.edge_index, edge_attr=data_macro.edge_attr
        )
        return z_final
