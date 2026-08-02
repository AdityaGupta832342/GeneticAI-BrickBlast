# GeneticAI BrickBlast

A GPU-oriented Brick Blast training project that combines:

- a tensorized batched game environment for fast parallel simulation
- a genetic algorithm (GA) for evolving CNN/MLP policies
- reinforcement learning agents (DQN and PPO) built on PyTorch
- a lightweight project test suite for validation

## Project structure

- `ai/` - model and optimization code
  - `genome.py` - neural network genome implementation
  - `ga.py` - CPU multiprocessing GA
  - `tensor_ga.py` - batched tensor GA
  - `rl.py` - DQN and PPO agents plus trainer harness
- `env/` - environment wrappers
  - `brickblast_env.py` - CPU game environment
  - `tensor_env.py` - batched tensor environment for GPU-friendly rollout simulation
- `brickblast/` - game engine and brick-breaker logic
- `saved_models/` - saved best model checkpoints
- `train_tensor_ga.py` - GPU training entry point for tensor GA
- `train_genetic.py` - classic genetic training entry point
- `watch_ai.py` - run trained models visually
- `play.py` - human gameplay
- `test_game.py` - project tests

## Features

### Tensorized training environment

The project includes a vectorized `TensorBrickBlastEnv` that simulates many game instances in parallel using PyTorch tensors. This is designed for CUDA-friendly batch processing and is used by the tensor GA training pipeline.

### Genetic algorithm evolution

The repository has two GA implementations:

- `GeneticAlgorithm` in `ai/ga.py` for traditional multiprocessing evolution
- `TensorGeneticAlgorithm` in `ai/tensor_ga.py` for batched evaluation on tensors

### RL support

The RL module in `ai/rl.py` includes:

- `DQNAgent`
- `PPOAgent`
- `TensorPolicyTrainer`

These can collect rollout trajectories from the tensor environment and perform optimization steps using real transitions.

## Setup

Create and activate a virtual environment:

```bash
cd /home/aditya/code/RL_brickblast
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
cd /home/aditya/code/RL_brickblast
. venv/bin/activate
pytest -q
```

## Train the tensor GA on GPU

```bash
cd /home/aditya/code/RL_brickblast
. venv/bin/activate
python train_tensor_ga.py --generations 1000 --pop-size 200 --max-turns 100 --model-type cnn --device cuda --seed 42
```

This saves the best model to:

```bash
saved_models/best_tensor_model.json
```

## Train the classic GA

```bash
cd /home/aditya/code/RL_brickblast
. venv/bin/activate
python train_genetic.py --generations 1000 --pop-size 200 --max-turns 100 --seed 42
```

## Play the game

```bash
cd /home/aditya/code/RL_brickblast
. venv/bin/activate
python play.py
```

## Watch a trained AI play

```bash
cd /home/aditya/code/RL_brickblast
. venv/bin/activate
python watch_ai.py
```

## Notes

- CUDA execution is preferred for the tensorized training path.
- The CPU/pygame path remains useful for debugging and local development.
- The repository is structured for experimentation and iterative training improvements.

## License

This project is provided as-is for educational and research purposes.
