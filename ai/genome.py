"""
Genome class representing a PyTorch MLP Neural Network evolved via Genetic Algorithm.
Evaluates candidate aiming angles based on board state observation.
"""
import json
import numpy as np
import torch
import torch.nn as nn


class Genome(nn.Module):
    def __init__(self, input_size=21, hidden_size=16, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size

        # PyTorch Neural Network MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

        # Kaiming / He normal initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.fitness = 0.0
        self.turns_survived = 0

    @torch.no_grad()
    def forward(self, x):
        """
        Forward pass through PyTorch MLP: Input(21) -> Hidden(16, ReLU) -> Output(1, Tanh in [-1.0, 1.0]).
        Accepts either numpy array or torch.Tensor.
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        elif isinstance(x, (list, tuple)):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x.float()

        h = self.relu(self.fc1(x_tensor))
        out = self.tanh(self.fc2(h))
        return float(out.item())

    @torch.no_grad()
    def select_action(self, env):
        """
        Predict continuous aiming angle in [3°, 177°].
        Supports continuous prediction (input_size=21) and legacy discrete evaluation (input_size=24).
        """
        if self.input_size == 21:
            obs = env.get_observation()
            action_val = self.forward(obs)  # value in [-1.0, 1.0]
            angle_deg = 90.0 + action_val * 87.0
            return max(3.0, min(177.0, float(angle_deg)))
        else:
            # Legacy fallback for saved models with input_size=24
            best_angle = 90.0
            best_score = -float("inf")
            for angle in range(3, 178, 5):
                feats = env.get_action_eval_features(float(angle))
                score = self.forward(feats)
                if score > best_score:
                    best_score = score
                    best_angle = float(angle)
            return best_angle

    @torch.no_grad()
    def mutate(self, mutation_rate=0.15, mutation_scale=0.25):
        """
        Apply Gaussian mutation directly to PyTorch parameter tensors.
        """
        for param in self.parameters():
            mask = torch.rand_like(param) < mutation_rate
            noise = torch.randn_like(param) * mutation_scale
            param.add_(mask * noise)

    @torch.no_grad()
    def crossover(self, other):
        """
        Uniform crossover between self and another PyTorch Genome to produce a child.
        """
        child = Genome(self.input_size, self.hidden_size)
        for p_child, p_self, p_other in zip(child.parameters(), self.parameters(), other.parameters()):
            mask = torch.rand_like(p_self) < 0.5
            p_child.copy_(torch.where(mask, p_self, p_other))
        return child

    def to_dict(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "W1": self.fc1.weight.detach().cpu().numpy().tolist(),
            "b1": self.fc1.bias.detach().cpu().numpy().tolist(),
            "W2": self.fc2.weight.detach().cpu().numpy().tolist(),
            "b2": self.fc2.bias.detach().cpu().numpy().tolist(),
            "fitness": float(self.fitness),
            "turns_survived": int(self.turns_survived),
        }

    @classmethod
    def from_dict(cls, data):
        g = cls(data["input_size"], data["hidden_size"])
        with torch.no_grad():
            g.fc1.weight.copy_(torch.tensor(data["W1"], dtype=torch.float32))
            g.fc1.bias.copy_(torch.tensor(data["b1"], dtype=torch.float32))
            g.fc2.weight.copy_(torch.tensor(data["W2"], dtype=torch.float32))
            g.fc2.bias.copy_(torch.tensor(data["b2"], dtype=torch.float32))
        g.fitness = float(data.get("fitness", 0.0))
        g.turns_survived = int(data.get("turns_survived", 0))
        return g

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
