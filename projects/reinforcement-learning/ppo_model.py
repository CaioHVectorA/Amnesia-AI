"""
PPO Actor-Critic Neural Network Architecture
============================================
Inherits the ContextualActionScoringNet backbone from Behavioral Cloning
and adds a Value Critic Head V(s) for Advantage Estimation (GAE) in PPO.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning")))
from model import ContextualActionScoringNet


class PPOActorCriticNet(nn.Module):
    def __init__(
        self,
        state_dim: int = 76,
        action_dim: int = 16,
        hidden_dim: int = 128,
        num_actions: int = 9,
        dropout: float = 0.05
    ):
        super().__init__()
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 1. State Feature Extractor (Shared Backbone)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 2. Dynamic Context Gating Layer (alpha(S))
        self.context_gate = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 3. Action Candidate Encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # 4. Actor Head (Cross-Scorer)
        self.actor_scorer = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2), hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 5. Critic Head (Value Function V(s) in [-1.0, +1.0])
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(
        self,
        state: torch.Tensor,
        candidate_actions: torch.Tensor,
        action_mask: torch.Tensor
    ):
        """Returns action logits and state value estimate V(s)."""
        # Encode State & Context Gating
        s_emb = self.state_encoder(state)
        gating = self.context_gate(state)
        s_gated = s_emb * gating

        # Value Function Estimate
        value = self.critic_head(s_gated).squeeze(-1)  # [Batch]

        # Actor Action Scoring
        s_expanded = s_gated.unsqueeze(1).expand(-1, self.num_actions, -1)
        a_emb = self.action_encoder(candidate_actions)
        combined = torch.cat([s_expanded, a_emb], dim=-1)
        logits = self.actor_scorer(combined).squeeze(-1)  # [Batch, num_actions]

        # Action Masking
        mask_penalty = (1.0 - action_mask) * 1e9
        masked_logits = logits - mask_penalty

        return masked_logits, value

    def load_from_bc_checkpoint(self, checkpoint_path: str):
        """Warm-starts the Actor from Behavioral Cloning weights."""
        if not os.path.exists(checkpoint_path):
            print(f"[!] Warning: BC Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        
        # Load matching keys (state_encoder, context_gate, action_encoder, scorer -> actor_scorer)
        my_dict = self.state_dict()
        loaded_keys = 0
        for k, v in state_dict.items():
            mapped_k = k
            if k.startswith("scorer."):
                mapped_k = k.replace("scorer.", "actor_scorer.")
            if mapped_k in my_dict and my_dict[mapped_k].shape == v.shape:
                my_dict[mapped_k] = v
                loaded_keys += 1

        self.load_state_dict(my_dict)
        print(f"[+] Warm-Started PPO Model from BC Checkpoint ({loaded_keys} weight tensors loaded)")
