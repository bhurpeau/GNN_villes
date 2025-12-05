#!/bin/bash
# Script d'installation et d'exécution du pipeline GML Géospatial.
set -euo pipefail
# Nom de l'environnement virtuel pour l'isolation
VENV_DIR=".venv"

# Recopie des données archivées
parent_dir="/home/onyxia/work/GNN_villes"
s3_dir="s3/bhurpeau/graphe/villes"
dirs=(
    "BDALTI"
    "checkpoints"
    "data"
    "data_GNN"
    "out"
)

echo "--- Démarrage de la configuration de l'environnement  GNN_Villes ---"
sudo apt update && sudo apt install -y p7zip-full
# --- 1. Installation des dépendances ---
pip install --upgrade pip
pip install uv ipykernel
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync
for d in "${dirs[@]}"; do
    dst="$parent_dir/$d"
    src="$s3_dir/$d"
    mkdir "$dst"
    mc cp -r "$src" "$dst"
done
python -m ipykernel install --user --name=venv-gml --display-name "Python (.venv GNN_Villes)"