"""
Policy Network for Satellite UPF Service Optimization.
Implements the 4-layer policy network from Fig. 3:
  Input layer  -> Calculation layer (theta_u = w * v_u + b)
              -> Softmax layer (o_u = e^theta_u / sum_k e^theta_k)
              -> Output layer (probability distribution over UPFs)

Per the paper: the calculation layer applies convolution operations to each
feature vector v_u(t) (kernel size 1 = per-UPF linear transform with shared
weights), and the softmax layer converts the result into UPF-selection
probabilities.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class PolicyNetworkTorch(nn.Module):
    """
    Policy Network using PyTorch (paper's Fig. 3 architecture).

    Architecture:
      Input:  state matrix M(t) of shape (U, F) with F=4 features per UPF.
      Calc:   per-UPF Conv1D with kernel size 1, shared weights across UPFs.
              theta_u = w * v_u(t) + b   (Eq. in Fig. 3)
      Softmax: o_u = e^{theta_u} / sum_k e^{theta_k}
      Output: vector of selection probabilities of length U.

    Conv1D(kernel_size=1) is mathematically equivalent to applying the same
    linear transform to each UPF feature vector v_u independently, exactly
    matching the paper's "convolution operations to each vector v_u(t)".
    """

    def __init__(self, num_upfs, feature_dim=4, hidden_dim=64, device=None):
        super(PolicyNetworkTorch, self).__init__()
        self.num_upfs = num_upfs
        self.feature_dim = feature_dim

        from config import Config
        self.device = torch.device(device or Config.get_device())

        # Calculation layer: Conv1D with kernel size 1 on the UPF dimension.
        # Input shape:  (batch, F, U)
        # Output shape: (batch, 1, U)  -> theta_u for each UPF
        # Two-stage Conv1D gives the network capacity to model non-linear
        # interactions while preserving the per-UPF (kernel=1) structure that
        # the paper's Fig. 3 specifies.
        self.conv1 = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state_matrix, available_mask=None):
        """
        Forward pass.
        Args:
            state_matrix: (batch, U, F) or (U, F)
            available_mask: (batch, U) or (U,) binary mask
        Returns:
            probabilities: probability distribution over UPFs (Eq. softmax in Fig. 3)
        """
        if state_matrix.dim() == 2:
            state_matrix = state_matrix.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # (B, U, F) -> (B, F, U) for Conv1D over the UPF dimension
        x = state_matrix.transpose(1, 2)
        x = F.relu(self.conv1(x))      # (B, hidden, U)
        scores = self.conv2(x).squeeze(1)  # (B, U)  - theta_u per UPF

        if available_mask is not None:
            if available_mask.dim() == 1:
                available_mask = available_mask.unsqueeze(0)
            scores = scores.masked_fill(available_mask == 0, -1e9)

        probabilities = F.softmax(scores, dim=-1)

        if squeeze:
            probabilities = probabilities.squeeze(0)

        return probabilities

    def select_action(self, state_matrix, available_mask=None, deterministic=False):
        """Select a target UPF based on policy network output."""
        state_tensor = torch.FloatTensor(state_matrix).to(self.device)
        mask_tensor = None
        if available_mask is not None:
            mask_tensor = torch.FloatTensor(available_mask).to(self.device)

        with torch.no_grad() if deterministic else torch.enable_grad():
            probs = self.forward(state_tensor, mask_tensor)

        if deterministic:
            action = torch.argmax(probs).item()
            log_prob = torch.log(probs[action] + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)

        return action, log_prob

    def get_log_prob(self, state_matrix, action, available_mask=None):
        """Get log probability for a specific action."""
        state_tensor = torch.FloatTensor(state_matrix).to(self.device)
        if state_tensor.dim() == 2:
            state_tensor = state_tensor.unsqueeze(0)

        mask_tensor = None
        if available_mask is not None:
            mask_tensor = torch.FloatTensor(available_mask).to(self.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)

        probs = self.forward(state_tensor, mask_tensor)
        probs = probs.squeeze(0)

        action_tensor = torch.tensor(action, device=self.device)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action_tensor)


class PolicyNetworkTF:
    """Policy Network using TensorFlow/Keras (matches Fig. 3 architecture)."""

    def __init__(self, num_upfs, feature_dim=4, hidden_dim=64):
        self.num_upfs = num_upfs
        self.feature_dim = feature_dim
        self.model = self._build_model(hidden_dim)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def _build_model(self, hidden_dim):
        # Input: (U, F)  -> Conv1D kernel=1 (per-UPF linear) -> theta_u -> softmax
        inputs = keras.Input(shape=(self.num_upfs, self.feature_dim))
        x = layers.Conv1D(hidden_dim, 1, activation='relu')(inputs)
        x = layers.Conv1D(1, 1)(x)
        scores = layers.Flatten()(x)
        probabilities = layers.Softmax()(scores)
        return keras.Model(inputs=inputs, outputs=probabilities)

    def predict(self, state_matrix):
        if len(state_matrix.shape) == 2:
            state_matrix = np.expand_dims(state_matrix, 0)
        return self.model.predict(state_matrix, verbose=0)

    def select_action(self, state_matrix, available_mask=None, deterministic=False):
        probs = self.predict(state_matrix)[0]

        if available_mask is not None:
            probs = probs * available_mask
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = available_mask / np.sum(available_mask)

        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(len(probs), p=probs)

        log_prob = np.log(probs[action] + 1e-8)
        return action, log_prob


def create_policy_network(num_upfs=None, feature_dim=4, hidden_dim=64):
    """Factory function to create the appropriate policy network."""
    if num_upfs is None:
        from config import Config
        num_upfs = Config.NUM_UPFS

    if TORCH_AVAILABLE:
        print("Using PyTorch policy network")
        return PolicyNetworkTorch(num_upfs, feature_dim, hidden_dim)
    elif TF_AVAILABLE:
        print("Using TensorFlow policy network")
        return PolicyNetworkTF(num_upfs, feature_dim, hidden_dim)
    else:
        raise RuntimeError("Neither PyTorch nor TensorFlow is available.")
