#!/usr/bin/env bash
set -euo pipefail

# Recopie des données pour archivage
parent_dir="/home/onyxia/work/GNN_villes"
s3_dir="s3/bhurpeau/graphe/villes"

dirs=(
    "BDALTI"
    "checkpoints"
    "data"
    "data_GNN"
    "out"
)

# Vérification : mc est disponible
if ! command -v mc >/dev/null 2>&1; then
    echo "❌ Erreur : la commande 'mc' n'est pas disponible dans le PATH."
    exit 1
fi

# Vérification : le répertoire parent existe
if [[ ! -d "$parent_dir" ]]; then
    echo "❌ Erreur : le répertoire parent n'existe pas : $parent_dir"
    exit 1
fi

echo "🟦 Début de l'archivage"
echo "   Source locale : $parent_dir"
echo "   Cible S3      : $s3_dir"
echo

for d in "${dirs[@]}"; do
    src="$parent_dir/$d"
    dst="$s3_dir/$d"

    if [[ ! -d "$src" ]]; then
        echo "⚠️  Dossier absent, on le saute : $src"
        continue
    fi

    echo "➡️  Copie de : $src"
    echo "    Vers     : $dst"
    mc cp -r "$src" "$dst"
    echo "✅ Terminé pour : $d"
    echo
done

echo "🎉 Archivage terminé sans erreur."
