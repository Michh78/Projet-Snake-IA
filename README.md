# 🌍 [English](#english) | [Français](#français)

---

<a name="english"></a>
# AI Snake — Reinforcement Learning (DQN)

An intelligent agent that learns to play the game Snake from scratch, without written rules, solely by observing the consequences of its actions.

---

## Project Overview

This project implements a **Deep Q-Learning (DQN)** agent trained on the Snake game.
The agent does not know the rules of the game: it learns through trial and error, maximizing a cumulative reward score.

**Technologies Used:**
- Python 3.11
- PyTorch — neural network and training
- Pygame — game environment
- NumPy — vector calculations
- Matplotlib — training visualization

---

## Results

| Metric | Value |
|---|---|
| Games played (checkpoint) | ~1059 |
| High score | Saved in `model/checkpoint.pth` |
| Network architecture | 11 → 256 → 3 |
| Experience memory | 100,000 transitions |
| Batch size | 1,000 |

---

## Project Structure
