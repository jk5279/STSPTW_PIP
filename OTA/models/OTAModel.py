import torch
import torch.nn as nn
import torch.nn.functional as F
from .SINGLEModel import (SINGLE_Encoder, EncoderLayer, reshape_by_heads,
                          multi_head_attention, Add_And_Normalization_Module,
                          FeedForward)

__all__ = ['OTAModel']


class OTAModel(nn.Module):
    """
    OTA Model for STSPTW using same encoder architecture as POMO for fair comparison.
    Hierarchical value functions (low/high) and actors (low/high) for temporal abstraction.
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.problem = self.model_params['problem']

        # Shared encoder from POMO (same architecture)
        self.encoder = SINGLE_Encoder(**model_params)
        self.encoded_nodes = None

        embedding_dim = model_params['embedding_dim']
        # obs_dim: dimension of the rich observation vector fed to value/actor heads.
        # Falls back to embedding_dim for backward compatibility.
        obs_dim = model_params.get('obs_dim', embedding_dim)
        self.obs_dim = obs_dim
        rep_dim = model_params.get('rep_dim', 16)
        value_hidden_dims = model_params.get('value_hidden_dims', (256, 256))
        actor_hidden_dims = model_params.get('actor_hidden_dims', (256, 256))

        # ------------------------------------------------------------------ #
        # Shared GC-Encoder: obs_dim -> gc_hidden -> gc_hidden -> embedding   #
        # 3-layer MLP so the model can mix per-node features before the goal  #
        # representation bottleneck.  State AND goal share the same weights.  #
        # ------------------------------------------------------------------ #
        gc_hidden_dim = model_params.get('gc_hidden_dim', 256)
        self.gc_encoder = nn.Sequential(
            nn.Linear(obs_dim, gc_hidden_dim),
            nn.LayerNorm(gc_hidden_dim),
            nn.ReLU(),
            nn.Linear(gc_hidden_dim, gc_hidden_dim),
            nn.LayerNorm(gc_hidden_dim),
            nn.ReLU(),
            nn.Linear(gc_hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # Goal representation network: phi([enc(s); enc(g)]) -> rep_dim
        # Input: 2 * embedding_dim  (both sides encoded by gc_encoder)
        goal_rep_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in value_hidden_dims:
            goal_rep_layers.append(nn.Linear(input_dim, hidden_dim))
            goal_rep_layers.append(nn.ReLU())
            input_dim = hidden_dim
        goal_rep_layers.append(nn.Linear(input_dim, rep_dim))
        self.goal_rep = nn.Sequential(*goal_rep_layers)

        # Value/actor heads now receive (embedding_dim + rep_dim) as input
        head_in = embedding_dim + rep_dim

        # Low-level value heads (ensemble: 2 networks)
        self.low_value_1 = self._build_value_head(head_in, value_hidden_dims)
        self.low_value_2 = self._build_value_head(head_in, value_hidden_dims)

        # High-level value heads (ensemble: 2 networks)
        self.high_value_1 = self._build_value_head(head_in, value_hidden_dims)
        self.high_value_2 = self._build_value_head(head_in, value_hidden_dims)

        # Target networks (for temporal difference learning)
        self.target_low_value_1 = self._build_value_head(head_in, value_hidden_dims)
        self.target_low_value_2 = self._build_value_head(head_in, value_hidden_dims)
        self.target_high_value_1 = self._build_value_head(head_in, value_hidden_dims)
        self.target_high_value_2 = self._build_value_head(head_in, value_hidden_dims)

        # Copy initial weights to target networks
        self.target_low_value_1.load_state_dict(self.low_value_1.state_dict())
        self.target_low_value_2.load_state_dict(self.low_value_2.state_dict())
        self.target_high_value_1.load_state_dict(self.high_value_1.state_dict())
        self.target_high_value_2.load_state_dict(self.high_value_2.state_dict())

        # Low-level actor (predicts actions given state and subgoal representation)
        self.low_actor = self._build_actor_head(head_in, actor_hidden_dims,
                                                model_params.get('action_dim', 100))

        # High-level actor (predicts subgoal representations given state and goal)
        self.high_actor = self._build_actor_head(head_in, actor_hidden_dims, rep_dim)

        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']

    def _build_value_head(self, input_dim, hidden_dims):
        """Build value network: input -> hidden layers -> scalar output"""
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def _build_actor_head(self, input_dim, hidden_dims, output_dim):
        """Build actor network: input -> hidden layers -> action output"""
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def pre_forward(self, reset_state):
        """
        Encode the problem instance.
        Returns encoded nodes and features (same as POMO).
        """
        if not self.problem.startswith('TSP') and not self.problem.startswith('STSP'):
            depot_xy = reset_state.depot_xy
            node_demand = reset_state.node_demand
        else:
            depot_xy = None

        node_xy = reset_state.node_xy

        if self.problem in ["STSPTW"]:
            node_tw_start = reset_state.node_tw_start
            node_tw_end = reset_state.node_tw_end
            tw_start = node_tw_start[:, :, None]
            tw_end = node_tw_end[:, :, None]
            if self.model_params.get("tw_normalize", False):
                tw_end_max = node_tw_end[:, :1, None]
                tw_start = tw_start / tw_end_max
                tw_end = tw_end / tw_end_max
            feature = torch.cat((node_xy, tw_start, tw_end), dim=2)
        else:
            raise NotImplementedError(f"Problem {self.problem} not supported in OTA")

        self.encoded_nodes = self.encoder(depot_xy, feature)
        # shape: (batch, problem+1, embedding)

        return self.encoded_nodes, feature

    def compute_state_embedding(self, raw_obs):
        """
        Encode a raw observation vector through the shared GC-Encoder.

        Args:
            raw_obs: (batch, obs_dim)  — rich TW-aware state vector

        Returns:
            state_emb: (batch, embedding_dim)
        """
        return self.gc_encoder(raw_obs)

    def compute_goal_embedding(self, raw_goal):
        """
        Encode a raw goal observation through the shared GC-Encoder.
        Shares weights with compute_state_embedding (same gc_encoder).

        Args:
            raw_goal: (batch, obs_dim)  — goal obs (e.g. _make_goal_obs output)

        Returns:
            goal_emb: (batch, embedding_dim)
        """
        return self.gc_encoder(raw_goal)

    def compute_goal_representation(self, state_raw, goal_raw):
        """
        Compute goal representation phi([enc(s); enc(g)]) with length normalisation.

        Encodes both inputs through gc_encoder, concatenates, passes through
        goal_rep MLP, then applies length normalisation:
            rep = F.normalize(rep, dim=-1) * sqrt(rep_dim)

        Args:
            state_raw: (batch, obs_dim)  — raw state observation
            goal_raw:  (batch, obs_dim)  — raw goal observation

        Returns:
            goal_rep: (batch, rep_dim)
        """
        state_emb = self.compute_state_embedding(state_raw)   # (B, embedding_dim)
        goal_emb  = self.compute_goal_embedding(goal_raw)     # (B, embedding_dim)
        combined  = torch.cat([state_emb, goal_emb], dim=-1)  # (B, 2*embedding_dim)
        rep = self.goal_rep(combined)                          # (B, rep_dim)
        rep = F.normalize(rep, dim=-1) * (rep.shape[-1] ** 0.5)
        return rep

    def get_low_value(self, state_raw, goal_rep):
        """
        Get low-level value function V_low(enc(s), phi([s;g])).

        Args:
            state_raw: (batch, obs_dim)  — raw state observation
            goal_rep:  (batch, rep_dim)

        Returns:
            v1, v2: (batch,) ensemble values
        """
        state_emb = self.compute_state_embedding(state_raw)
        combined = torch.cat([state_emb, goal_rep], dim=-1)
        v1 = self.low_value_1(combined).squeeze(-1)
        v2 = self.low_value_2(combined).squeeze(-1)
        return v1, v2

    def get_high_value(self, state_raw, goal_rep):
        """
        Get high-level value function V_high(enc(s), phi([s;w])).

        Args:
            state_raw: (batch, obs_dim)  — raw state observation
            goal_rep:  (batch, rep_dim)

        Returns:
            v1, v2: (batch,) ensemble values
        """
        state_emb = self.compute_state_embedding(state_raw)
        combined = torch.cat([state_emb, goal_rep], dim=-1)
        v1 = self.high_value_1(combined).squeeze(-1)
        v2 = self.high_value_2(combined).squeeze(-1)
        return v1, v2

    def get_target_low_value(self, state_raw, goal_rep):
        """Target network for low-level value (for TD learning).

        Args:
            state_raw: (batch, obs_dim)  — raw state observation
            goal_rep:  (batch, rep_dim)
        """
        state_emb = self.compute_state_embedding(state_raw)
        combined = torch.cat([state_emb, goal_rep], dim=-1)
        v1 = self.target_low_value_1(combined).squeeze(-1)
        v2 = self.target_low_value_2(combined).squeeze(-1)
        return v1, v2

    def get_target_high_value(self, state_raw, goal_rep):
        """Target network for high-level value (for TD learning).

        Args:
            state_raw: (batch, obs_dim)  — raw state observation
            goal_rep:  (batch, rep_dim)
        """
        state_emb = self.compute_state_embedding(state_raw)
        combined = torch.cat([state_emb, goal_rep], dim=-1)
        v1 = self.target_high_value_1(combined).squeeze(-1)
        v2 = self.target_high_value_2(combined).squeeze(-1)
        return v1, v2

    def get_low_actor_logits(self, state_raw, goal_rep):
        """
        Get low-level actor output (action logits).

        Args:
            state_raw: (batch, obs_dim)  — raw state observation
            goal_rep:  (batch, rep_dim)

        Returns:
            logits: (batch, action_dim)
        """
        state_emb = self.compute_state_embedding(state_raw)
        combined = torch.cat([state_emb, goal_rep], dim=-1)
        logits = self.low_actor(combined)
        return logits

    def get_high_actor_logits(self, state_raw, goal_rep):
        """
        Get high-level actor output (subgoal representation).

        Args:
            state_raw: (batch, obs_dim)  — raw state observation
            goal_rep:  (batch, rep_dim)

        Returns:
            subgoal_rep: (batch, rep_dim)
        """
        state_emb = self.compute_state_embedding(state_raw)
        combined = torch.cat([state_emb, goal_rep], dim=-1)
        subgoal_rep = self.high_actor(combined)
        return subgoal_rep

    def update_target_networks(self, tau=0.005):
        """
        Soft update of target networks: target = tau * source + (1-tau) * target
        """
        for param, target_param in zip(self.low_value_1.parameters(), self.target_low_value_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.low_value_2.parameters(), self.target_low_value_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.high_value_1.parameters(), self.target_high_value_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.high_value_2.parameters(), self.target_high_value_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def forward(self, state_raw, goal_raw, mode='low_value'):
        """
        Generic forward pass for different components.

        Args:
            state_raw: (batch, obs_dim)  — raw state observation
            goal_raw:  (batch, obs_dim)  — raw goal observation
            mode: one of ['low_value', 'high_value', 'low_actor', 'high_actor', 'goal_rep']
        """
        goal_rep = self.compute_goal_representation(state_raw, goal_raw)

        if mode == 'low_value':
            return self.get_low_value(state_raw, goal_rep)
        elif mode == 'high_value':
            return self.get_high_value(state_raw, goal_rep)
        elif mode == 'low_actor':
            return self.get_low_actor_logits(state_raw, goal_rep)
        elif mode == 'high_actor':
            return self.get_high_actor_logits(state_raw, goal_rep)
        elif mode == 'goal_rep':
            return goal_rep
        else:
            raise ValueError(f"Unknown mode: {mode}")
