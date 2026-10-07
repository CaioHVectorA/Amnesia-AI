"""
Behavioral Cloning Policy Network (Context-Gated Action-Scoring Architecture)
=============================================================================
A permutation-invariant, candidate-scoring neural network for Showdown actions
featuring Contextual Gating & Dynamic Weight Modulation.

Instead of static linear combinations, this architecture employs:
1. A Context Gating Network (sigmoid gating weights) to modulate feature salience
   based on game phase (alive count, HP ratio, opponent boost threats).
2. Cross-Scoring Head combining state and candidate action representations.
3. Strict Action Masking (illegal moves, sleep clause violations, empty PP).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualActionScoringNet(nn.Module):
    def __init__(
        self,
        state_dim: int = 76,
        action_dim: int = 16,
        hidden_dim: int = 128,
        num_actions: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 1. State Feature Extractor
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 2. Dynamic Contextual Gating Layer (Learned Weight Coefficients alpha(S))
        self.context_gate = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # Multiplicative gating coefficients in [0, 1]
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

        # 4. Cross-Scoring Head (combines gated state and action embeddings)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2), hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        state: torch.Tensor,               # [Batch, state_dim]
        candidate_actions: torch.Tensor,   # [Batch, num_actions, action_dim]
        action_mask: torch.Tensor          # [Batch, num_actions] (1 for legal, 0 for illegal)
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        # 1. Encode Base State Features
        s_emb = self.state_encoder(state)   # [Batch, hidden_dim]

        # 2. Compute Contextual Gating Weights alpha(S)
        gating = self.context_gate(state)   # [Batch, hidden_dim]

        # 3. Apply Context Gating Modulation
        s_gated = s_emb * gating            # [Batch, hidden_dim]
        s_expanded = s_gated.unsqueeze(1).expand(-1, self.num_actions, -1)

        # 4. Encode Action Candidates
        a_emb = self.action_encoder(candidate_actions)  # [Batch, num_actions, hidden_dim // 2]

        # 5. Combined Cross-Scoring
        combined = torch.cat([s_expanded, a_emb], dim=-1)
        logits = self.scorer(combined).squeeze(-1)       # [Batch, num_actions]

        # 6. Strict Action Masking
        mask_penalty = (1.0 - action_mask) * 1e9
        masked_logits = logits - mask_penalty

        return masked_logits

    def predict_action(
        self,
        state: torch.Tensor,
        candidate_actions: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float = 0.0
    ) -> int:
        """Runs inference for a single state and returns the optimal action index."""
        self.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if candidate_actions.dim() == 2:
                candidate_actions = candidate_actions.unsqueeze(0)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

            logits = self.forward(state, candidate_actions, action_mask)

            if temperature <= 0.01:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                action = int(torch.multinomial(probs, num_samples=1).item())

            return action


# Alias for backward compatibility
ActionScoringPolicyNet = ContextualActionScoringNet
