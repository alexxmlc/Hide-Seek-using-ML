# Multi-Agent Hide & Seek: Deep Recurrent Q-Network (DRQN)

A custom Multi-Agent Reinforcement Learning (MARL) environment built from scratch in Python. This project simulates an adversarial "Hide & Seek" scenario where autonomous AI agents learn spatial navigation, target tracking, and evasion entirely through trial, error, and neural network backpropagation.

## 🧠 The AI Architecture

The agents are powered by a **Deep Recurrent Q-Network (DRQN)**, transitioning from standard static routing to dynamic, temporal awareness:
* **Frame Stacking & TimeDistributed CNNs:** The agent processes a sequence of 4 chronological frames `(1, 4, 5, 5, 1)` through Convolutional layers to extract spatial maze geometry.
* **LSTM Integration:** The CNN outputs are fed into a Long Short-Term Memory (LSTM) layer, granting the agent temporal context to perceive its own movement patterns and break out of localized loops ("wiggling").
* **Target Networks (Double DQN):** Utilizes a split Active (Student) and Target (Teacher) brain system to stabilize the Bellman equations and prevent Q-Value collapse.
* **Intrinsic Motivation (Curiosity):** A custom footprint tracker punishes the agent for revisiting tiles, organically forcing map exploration without relying on hard-coded heuristics.
* **Flashbulb Memory:** A lightweight implementation of Prioritized Experience Replay that duplicates winning capture sequences in the Replay Buffer to overcome the "Sparse Reward" problem.

## 🛠️ Tech Stack
* **Core Logic:** Python 3.x
* **Deep Learning:** TensorFlow / Keras (CNNs, LSTMs, Q-Learning)
* **Mathematics:** NumPy (High-performance matrix operations)
* **Environment Rendering:** Pygame
* **Telemetry & Analytics:** Matplotlib

## 📂 Project Structure
* `main.py` - The core execution loop. Handles the environment stepping, frame synchronization, agent spawning, and triggering Matplotlib performance graphs on exit.
* `agent.py` - Contains the `Agent` class. Manages the Q-learning logic, Epsilon-greedy exploration, memory buffer, and the `think_and_move` function.
* `brain.py` - The neural network architect. Contains the TensorFlow sequential model definition (`create_brain`) including the `TimeDistributed` and `LSTM` layers.
* `environment.py` - Defines the 2D grid, static maze geometry, vision overlay (dynamic coordinate injection), and Pygame drawing logic.
* `settings.py` - Global configuration variables (Grid size, colors, hyperparameters, etc.).
* `seeker.keras` / `hider.keras` - Saved, trained weights for the neural networks.

## 🚀 How to Run

**1. Install Dependencies**
Ensure you have Python installed, then install the required libraries:
```bash
pip install tensorflow pygame numpy matplotlib
