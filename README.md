# FR 🇫🇷 - IA Snake — Apprentissage par Renforcement (DQN)

Un agent intelligent qui apprend à jouer au jeu Snake de zéro, sans règles écrites,
uniquement en observant les conséquences de ses actions.

---

## Présentation du projet

Ce projet implémente un agent **Deep Q-Learning (DQN)** entraîné sur le jeu Snake.
L'agent ne connaît pas les règles du jeu : il apprend par essais et erreurs,
en maximisant un score de récompense cumulé.

**Technologies utilisées :**
- Python 3.11
- PyTorch — réseau de neurones et entraînement
- Pygame — environnement de jeu
- NumPy — calculs vectoriels
- Matplotlib — visualisation de l'entraînement

---

## Résultats obtenus

| Métrique | Valeur |
|---|---|
| Games jouées (checkpoint) | ~1059 |
| Score record | Sauvegardé dans `model/checkpoint.pth` |
| Architecture réseau | 11 → 256 → 3 |
| Mémoire d'expériences | 100 000 transitions |
| Taille de batch | 1 000 |

---

## Structure du projet

```text
IA_snake/
├── agent.py              # Agent RL + boucle d'entraînement
├── game.py               # Environnement Snake pour l'IA
├── model.py              # Réseau de neurones + QTrainer
├── helper.py             # Graphique d'entraînement en temps réel
├── snake_game_human.py   # Version jouable au clavier
├── arial.ttf             # Police d'affichage
└── model/
    ├── model.pth         # Poids du modèle entraîné
    └── checkpoint.pth    # Checkpoint complet (modèle + optimiseur + état)
```

---

## Installation

### Prérequis

- Python 3.10 ou 3.11

### Installation des dépendances

```bash
pip install torch pygame numpy matplotlib ipython
```

---

## Lancement

> **Important** : toujours lancer depuis le dossier du projet pour que `arial.ttf` soit trouvé.

```bash
cd "chemin/vers/IA_snake"
```

### Entraîner l'IA

```bash
python agent.py
```

L'agent reprend automatiquement depuis le dernier checkpoint s'il en existe un.
Une fenêtre Pygame affiche le jeu en direct.
Un graphique Matplotlib montre l'évolution des scores.

### Jouer soi-même

```bash
python snake_game_human.py
```

Contrôles : **flèches directionnelles** du clavier.

---

## Fonctionnement technique

### 1. L'environnement

Le jeu Snake est adapté pour l'IA (`game.py`) :
- **Action** : 3 choix — tout droit, tourner à droite, tourner à gauche
- **Récompense** : +10 pour manger la nourriture, -10 en cas de collision
- **Fin de partie** : collision avec un mur ou avec le corps, ou timeout (100 × longueur du serpent)

### 2. L'état — 11 variables booléennes

L'agent perçoit le monde via un vecteur de **11 valeurs** (0 ou 1) :

| # | Description |
|---|---|
| 0 | Danger tout droit |
| 1 | Danger à droite |
| 2 | Danger à gauche |
| 3 | Direction actuelle : gauche |
| 4 | Direction actuelle : droite |
| 5 | Direction actuelle : haut |
| 6 | Direction actuelle : bas |
| 7 | Nourriture à gauche |
| 8 | Nourriture à droite |
| 9 | Nourriture en haut |
| 10 | Nourriture en bas |

### 3. Le réseau de neurones

```text
Entrée (11) → Couche cachée (256, ReLU) → Sortie (3)
```

La sortie donne une **valeur Q** pour chacune des 3 actions possibles.
L'agent choisit l'action avec la valeur Q maximale.

### 4. L'algorithme Q-Learning

L'agent apprend via l'**équation de Bellman** :

```text
Q(état, action) = récompense + γ × max(Q(état suivant, toutes actions))
```

Avec :
- **γ = 0.9** (facteur d'actualisation — importance du futur)
- **Loss = MSE** entre la valeur Q prédite et la valeur Q cible
- **Optimiseur = Adam** (lr = 0.001)

### 5. Exploration vs Exploitation (ε-greedy)

Pendant les 80 premières parties, l'agent explore aléatoirement pour découvrir
de nouvelles stratégies. Ensuite, il exploite les connaissances acquises.

```text
ε = max(0, 80 - nombre_de_parties)
Si random(0, 200) < ε  →  action aléatoire (exploration)
Sinon                  →  réseau de neurones (exploitation)
```

### 6. Experience Replay

Chaque transition `(état, action, récompense, état_suivant, terminé)` est stockée
dans une mémoire de **100 000 expériences**. À chaque fin de partie, un batch
de **1 000 transitions** est tiré aléatoirement pour l'entraînement.

Cela permet de :
- Briser les corrélations temporelles entre les expériences
- Réutiliser les expériences passées plusieurs fois
- Stabiliser l'entraînement

---

## Hyperparamètres

| Paramètre | Valeur |
|---|---|
| `MAX_MEMORY` | 100 000 |
| `BATCH_SIZE` | 1 000 |
| `LR` (learning rate) | 0.001 |
| `GAMMA` (discount) | 0.9 |
| `HIDDEN_SIZE` | 256 |
| `EPSILON_MAX` | 80 parties |
| `SPEED` | 40 FPS |

---

## Checkpoint et reprise d'entraînement

Le modèle se sauvegarde automatiquement :
- À chaque **nouveau record** de score
- Toutes les **50 parties**

Le fichier `model/checkpoint.pth` contient :
- Les poids du réseau de neurones
- L'état de l'optimiseur Adam
- Le nombre de parties jouées
- La valeur epsilon courante
- Le record actuel

Au prochain lancement, l'entraînement **reprend exactement** où il s'est arrêté.

---

## Auteur

Projet réalisé dans le cadre du cursus **Data IA B3 — YDays**.

---

# EN 🇬🇧 - IA Snake — Reinforcement Learning (DQN)

An intelligent agent that learns to play the Snake game from scratch, without any hard-coded rules,
simply by observing the consequences of its actions.

---

## Project Overview

This project implements a **Deep Q-Learning (DQN)** agent trained on the Snake game.
The agent does not know the rules of the game: it learns through trial and error,
maximizing a cumulative reward score.

**Technologies used:**
- Python 3.11
- PyTorch — neural network and training
- Pygame — game environment
- NumPy — vector computations
- Matplotlib — training visualization

---

## Results

| Metric | Value |
|---|---|
| Games played (checkpoint) | ~1059 |
| Best score | Saved in `model/checkpoint.pth` |
| Network architecture | 11 → 256 → 3 |
| Experience replay memory | 100,000 transitions |
| Batch size | 1,000 |

---

## Project Structure

```text
IA_snake/
├── agent.py              # RL agent + training loop
├── game.py               # Snake environment for the AI
├── model.py              # Neural network + QTrainer
├── helper.py             # Real-time training graph
├── snake_game_human.py   # Keyboard-playable version
├── arial.ttf             # Display font
└── model/
    ├── model.pth         # Trained model weights
    └── checkpoint.pth    # Complete checkpoint (model + optimizer + state)
```

---

## Installation

### Requirements

- Python 3.10 or 3.11

### Install Dependencies

```bash
pip install torch pygame numpy matplotlib ipython
```

---

## Running the Project

> **Important:** Always run from the project directory so that `arial.ttf` can be found.

```bash
cd "path/to/IA_snake"
```

### Train the AI

```bash
python agent.py
```

The agent automatically resumes training from the latest checkpoint if one exists.
A Pygame window displays the game in real time.
A Matplotlib graph shows the evolution of the scores.

### Play the Game Yourself

```bash
python snake_game_human.py
```

Controls: **keyboard arrow keys**.

---

## Technical Overview

### 1. The Environment

The Snake game is adapted for the AI (`game.py`):
- **Action:** 3 choices — move straight, turn right, or turn left
- **Reward:** +10 for eating food, -10 in case of a collision
- **Game over:** collision with a wall or the snake's body, or timeout (100 × snake length)

### 2. The State — 11 Boolean Values

The agent perceives the environment through a vector of **11 values** (0 or 1):

| # | Description |
|---|---|
| 0 | Danger straight ahead |
| 1 | Danger to the right |
| 2 | Danger to the left |
| 3 | Current direction: left |
| 4 | Current direction: right |
| 5 | Current direction: up |
| 6 | Current direction: down |
| 7 | Food to the left |
| 8 | Food to the right |
| 9 | Food above |
| 10 | Food below |

### 3. Neural Network

```text
Input (11) → Hidden Layer (256, ReLU) → Output (3)
```

The output provides a **Q-value** for each of the 3 possible actions.
The agent chooses the action with the highest Q-value.

### 4. The Q-Learning Algorithm

The agent learns using the **Bellman equation**:

```text
Q(state, action) = reward + γ × max(Q(next state, all actions))
```

With:
- **γ = 0.9** (discount factor — importance of future rewards)
- **Loss = MSE** between the predicted Q-value and the target Q-value
- **Optimizer = Adam** (lr = 0.001)

### 5. Exploration vs. Exploitation (ε-greedy)

During the first 80 games, the agent explores randomly to discover
new strategies. Afterwards, it exploits the knowledge it has acquired.

```text
ε = max(0, 80 - number_of_games)
If random(0, 200) < ε  →  random action (exploration)
Otherwise              →  neural network (exploitation)
```

### 6. Experience Replay

Each transition `(state, action, reward, next_state, done)` is stored
in a memory of **100,000 experiences**. At the end of each game, a batch
of **1,000 transitions** is randomly selected for training.

This allows the agent to:
- Break temporal correlations between experiences
- Reuse past experiences multiple times
- Stabilize the training process

---

## Hyperparameters

| Parameter | Value |
|---|---|
| `MAX_MEMORY` | 100,000 |
| `BATCH_SIZE` | 1,000 |
| `LR` (learning rate) | 0.001 |
| `GAMMA` (discount) | 0.9 |
| `HIDDEN_SIZE` | 256 |
| `EPSILON_MAX` | 80 games |
| `SPEED` | 40 FPS |

---

## Checkpoint and Training Resumption

The model is automatically saved:
- Whenever a **new high score** is achieved
- Every **50 games**

The `model/checkpoint.pth` file contains:
- Neural network weights
- Adam optimizer state
- Number of games played
- Current epsilon value
- Current high score

When the project is launched again, training **resumes exactly** where it left off.

---

## Author

Project developed as part of the **Data & AI B3 — YDays** curriculum.
