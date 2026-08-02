"""
Genome class representing a PyTorch MLP Neural Network evolved via Genetic Algorithm.
Evaluates candidate aiming angles based on board state observation.
"""
import json
import numpy as np
import torch
import torch.nn as nn


class Genome(nn.Module):
    def __init__(self, input_size=21, hidden_size=16, model_type="mlp", device=None, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model_type = model_type
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        if self.model_type == "cnn":
            # 2D Convolutional Backbone (2 channels: Brick HP, Powerups -> 16 -> 32)
            self.conv = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Fully Connected Head: 32*10*8 (2560) spatial features + 2 global features [launch_x, turn] = 2562
            self.fc = nn.Sequential(
                nn.Linear(2562, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )
        else:
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
        self.to(self.device)

    @torch.no_grad()
    def forward_tensor(self, x):
        """
        Forward pass returning a PyTorch Tensor (for vmap/functional_call).
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float().to(self.device)
        elif isinstance(x, (list, tuple)):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x_tensor = x.float().to(self.device)

        h = self.relu(self.fc1(x_tensor))
        out = self.tanh(self.fc2(h))
        return out.squeeze(-1)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """
        Forward pass through PyTorch Genome. Supports MLP (1 arg x) or CNN (2 args grid, globals_arr).
        """
        if self.model_type == "cnn" or "grid" in kwargs or len(args) == 2:
            if "grid" in kwargs and "globals_arr" in kwargs:
                grid = kwargs["grid"]
                globals_arr = kwargs["globals_arr"]
            elif len(args) == 2:
                grid, globals_arr = args
            else:
                raise ValueError("Expected grid and globals_arr for CNN forward")
            return self.forward_cnn_tensor(grid, globals_arr)
        else:
            x = args[0] if len(args) > 0 else kwargs["x"]
            return float(self.forward_tensor(x).item())

    @torch.no_grad()
    def forward_cnn_tensor(self, grid, globals_arr):
        """
        Tensor-output forward pass for CNN (for vmap/functional_call):
        grid: (2, 10, 8) or (B, 2, 10, 8)
        globals_arr: (2,) or (B, 2)
        """
        if isinstance(grid, np.ndarray):
            grid_t = torch.from_numpy(grid).float().to(self.device)
        else:
            grid_t = grid.float().to(self.device)
        if grid_t.ndim == 3:
            grid_t = grid_t.unsqueeze(0)

        if isinstance(globals_arr, np.ndarray):
            glob_t = torch.from_numpy(globals_arr).float().to(self.device)
        else:
            glob_t = globals_arr.float().to(self.device)
        if glob_t.ndim == 1:
            glob_t = glob_t.unsqueeze(0)

        spatial_feats = self.conv(grid_t)
        combined = torch.cat([spatial_feats, glob_t], dim=-1)
        out = self.fc(combined)
        return out.squeeze(-1).squeeze(-1)

    @torch.no_grad()
    def forward_cnn(self, grid, globals_arr):
        """
        Forward pass for CNN:
        grid: (2, 10, 8) or (B, 2, 10, 8)
        globals_arr: (2,) or (B, 2)
        """
        return float(self.forward_cnn_tensor(grid, globals_arr).item())

    @torch.no_grad()
    def select_action(self, env):
        """
        Predict continuous aiming angle in [3°, 177°].
        Supports CNN spatial grid prediction, MLP continuous prediction, and legacy discrete evaluation.
        """
        if self.model_type == "cnn":
            grid, glob = env.get_grid_observation()
            action_val = self.forward_cnn(grid, glob)
            angle_deg = 90.0 + action_val * 87.0
            return max(3.0, min(177.0, float(angle_deg)))
        elif self.input_size == 21:
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
        child = Genome(self.input_size, self.hidden_size, model_type=self.model_type, device=self.device)
        for p_child, p_self, p_other in zip(child.parameters(), self.parameters(), other.parameters()):
            mask = torch.rand_like(p_self) < 0.5
            p_child.copy_(torch.where(mask, p_self, p_other))
        return child

    def to_dict(self):
        state_dict_serializable = {
            k: v.detach().cpu().numpy().tolist() for k, v in self.state_dict().items()
        }
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "state_dict": state_dict_serializable,
            "W1": getattr(self, "fc1", nn.Linear(21, 16)).weight.detach().cpu().numpy().tolist() if self.model_type == "mlp" else [],
            "b1": getattr(self, "fc1", nn.Linear(21, 16)).bias.detach().cpu().numpy().tolist() if self.model_type == "mlp" else [],
            "W2": getattr(self, "fc2", nn.Linear(16, 1)).weight.detach().cpu().numpy().tolist() if self.model_type == "mlp" else [],
            "b2": getattr(self, "fc2", nn.Linear(16, 1)).bias.detach().cpu().numpy().tolist() if self.model_type == "mlp" else [],
            "fitness": float(self.fitness),
            "turns_survived": int(self.turns_survived),
        }

    @classmethod
    def from_dict(cls, data, device=None):
        model_type = data.get("model_type", "mlp")
        g = cls(data["input_size"], data["hidden_size"], model_type=model_type, device=device)
        with torch.no_grad():
            if "state_dict" in data and data["state_dict"]:
                state_dict = {
                    k: torch.tensor(v, dtype=torch.float32, device=g.device) for k, v in data["state_dict"].items()
                }
                g.load_state_dict(state_dict, strict=False)
            elif model_type == "mlp" and "W1" in data and len(data["W1"]) > 0:
                g.fc1.weight.copy_(torch.tensor(data["W1"], dtype=torch.float32, device=g.device))
                g.fc1.bias.copy_(torch.tensor(data["b1"], dtype=torch.float32, device=g.device))
                g.fc2.weight.copy_(torch.tensor(data["W2"], dtype=torch.float32, device=g.device))
                g.fc2.bias.copy_(torch.tensor(data["b2"], dtype=torch.float32, device=g.device))
        g.fitness = float(data.get("fitness", 0.0))
        g.turns_survived = int(data.get("turns_survived", 0))
        g.to(g.device)
        return g

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath, device=None):
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data, device=device)
