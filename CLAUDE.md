# Contexte pour une instance Claude

## Ce qu'est ce repo

Framework de deep learning bare-metal pour microcontrôleurs, écrit en Rust,
`no_std`, zero-alloc. Tout vit sur la pile, toutes les tailles sont connues à
la compilation via des const generics (`Tensor<ROWS, COLS, NUMEL>`, `Linear<IN,
OUT, NUMEL>`...). Cible : STM32 et autres Cortex-M.

**Le point important, à ne jamais minimiser en le présentant :** ce n'est pas
un runtime d'inférence embarquée (comme TFLite Micro). Il entraîne *et* fait
tourner des réseaux directement sur la puce — `forward` + `backward` + `sgd`
tournent tous sur le microcontrôleur, pas seulement le déploiement d'un modèle
pré-entraîné. C'est l'apprentissage on-device qui est l'apport différenciant,
pas juste l'inférence légère. Voir `README.md` pour l'API et les benchmarks.

Repos liés :
- `../ferrite-embedded` — extension pratique sur du vrai matériel
- `../ferrite-lab` — exercices satellites (fondamentaux mémoire Rust,
  reconstruits par Matis lui-même ; son propre `CLAUDE.md` a une règle stricte
  de ne jamais écrire les solutions à sa place)

## Renommage en cours — PAS ENCORE TRANCHÉ

Le nom "Ferrite" est en cours de remplacement par un acronyme-jeu de mot en
français façon SacreBLEU (mot français qui cache un acronyme anglais). Deux
finalistes, à trancher avec Matis avant de toucher au code :

- **ROUILLE** = **R**untime for **O**n-device, **U**ltra-lightweight
  **I**nference & **L**earning — a **L**ean **L**ibrary for **E**mbedded
  (systems). Le jeu de mot : Rust (le langage) → rouille en français → clin
  d'œil à Ferrite (oxyde de fer).
- **FONTE** = **F**ramework for **O**n-device **N**eural **T**raining on
  **E**mbedded (devices). Plus direct sur le "Training".

Une fois le nom choisi, à renommer : le dossier `~/Dev/projects/frugal_ml` lui-même,
le nom de la crate dans `Cargo.toml` (actuellement `ferrite`), les mentions
dans le vault Obsidian (`03_Projects/Ferrite Embedded DL/_MOC.md`,
`01_Career/Roadmap - Principal Engineer Track.md`, `01_Career/Skills Matrix.md`,
`03_Projects/_Projects Catalog.md`).

## Règle sur les commits — permanente

**Jamais de mention d'IA, de co-auteur Claude, ni de « Generated with » dans
les commits de ce repo.** Règle explicite et permanente de Matis (confirmée
dans `../ferrite-lab/CLAUDE.md`, vaut pour les deux repos).

## Contexte carrière / portfolio

C'est la pièce phare du portfolio pour une candidature AI systems engineer /
embedded, avec ARM et NVIDIA comme cibles explicites (edge AI, embedded
inference *and training* runtimes). Documenté dans le vault Obsidian
`~/Obsidian/horus/` :
- `03_Projects/Ferrite Embedded DL/_MOC.md` — vue d'ensemble du projet
- `01_Career/Roadmap - Principal Engineer Track.md` — pourquoi ce projet compte
  pour l'objectif NVIDIA/ARM
- `01_Career/Skills Matrix.md` — positionnement ML Systems/Infra

## État actuel

Branche active : `try_unchecked` (accesseurs Tensor non-checked pour la perf,
~1.25x plus rapide sur `multiply`). `benches/scripts/` non tracké. Le module
`src/linalg/storage.rs` (stockage générique `Buffer`/`Storage` — pile par
défaut, tas en option) est déjà committé et intégré, ce n'est plus un
chantier en suspens.
