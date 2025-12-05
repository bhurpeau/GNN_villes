#!/bin/bash
# Script d'installation et d'exécution du pipeline GML Géospatial.

# Nom de l'environnement virtuel pour l'isolation
VENV_DIR=".venv"
echo "--- Démarrage de la configuration de l'environnement  GNN_Villes ---"
sudo apt update && sudo apt install -y p7zip-full
# --- 1. Installation des dépendances ---
pip install --upgrade pip
pip install uv ipykernel
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv sync
uv pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
python -m ipykernel install --user --name=venv-gml --display-name "Python (.venv GNN_Villes)"
mkdir ./data
mkdir ./data_GNN
mkdir ./BDALTI
mkdir ./checkpoints
mc cp -r  s3/bhurpeau/graphe/villes/data/ data/
mc cp -r  s3/bhurpeau/graphe/villes/data_GNN/ data_GNN/
mc cp  s3/bhurpeau/graphe/villes/BDALTI/bdalti25m.tif BDALTI/