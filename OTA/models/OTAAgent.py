import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

__all__ = ['OTAAgent']


class OTAAgent(nn.Module):
    """
    OTA Agent: Implements Option-aware Temporally Abstracted value learning.
    Hierarchical structure with low-level and high-level actors/values.
    """

    def __init__(self, model, optimizer, scheduler, device, **agent_params):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.agent_params = agent_params

        # OTA hyperparameters
        self.low_alpha = agent_params.get('low_alpha', 3.0)
        self.high_alpha = agent_params.get('high_alpha', 3.0)
        self.low_discount = agent_params.get('low_discount', 0.99)
        self.high_discount = agent_params.get('high_discount', 0.99)
        self.expectile = agent_params.get('expectile', 0.7)
        self.tau = agent_params.get('tau', 0.005)
        self.subgoal_steps = agent_params.get('subgoal_steps', 25)
        self.abstraction_factor = agent_params.get('abstraction_factor', 5)
        self.rep_dim = agent_params.get('rep_dim', 16)

    @staticmethod
    def expectile_loss(adv, diff, expectile=0.7):
        """
        Asymmetric L2 loss: weight by advantage sign.
        When adv >= 0, weight by expectile (focus on underestimation).
        When adv < 0, weight by (1 - expectile) (focus on overestimation).
        
        Args:
            adv: (batch,) advantage = Q - V
            diff: (batch,) difference = Q_target - V_current
            expectile: float, typically 0.7
        
        Returns:
            loss: (batch,) scalar losses
        """
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def low_value_loss(self, batch, current_step):
        """
        Low-level value loss (short-horizon, single-step TD).
        Uses IQL-style expectile loss.

        Batch keys (matching reference ota.py):
            observations                   : s_t
            next_observations              : s_{t+1}
            value_goals                    : goal g
            rewards                        : r_t
            masks                          : done mask (0 at terminal)
        """
        s      = batch['observations'].to(self.device)
        s_next = batch['next_observations'].to(self.device)
        g      = batch['value_goals'].to(self.device)
        r      = batch['rewards'].to(self.device)
        mask   = batch['masks'].to(self.device)

        # Compute goal representations
        goal_rep = self.model.compute_goal_representation(s, g)
        goal_rep_next = self.model.compute_goal_representation(s_next, g)

        # Target values (no gradient)
        with torch.no_grad():
            v1_next_t, v2_next_t = self.model.get_target_low_value(s_next, goal_rep_next)
            v_next_t = torch.minimum(v1_next_t, v2_next_t)
            q_target = r + self.low_discount * mask * v_next_t

            v1_t, v2_t = self.model.get_target_low_value(s, goal_rep)
            v_t = (v1_t + v2_t) / 2
            adv = q_target - v_t

        # Current values (with gradient)
        v1, v2 = self.model.get_low_value(s, goal_rep)

        # Compute Q targets for both heads
        with torch.no_grad():
            q1_target = r + self.low_discount * mask * v1_next_t
            q2_target = r + self.low_discount * mask * v2_next_t

        # Expectile loss
        loss1 = self.expectile_loss(adv, q1_target - v1, self.expectile).mean()
        loss2 = self.expectile_loss(adv, q2_target - v2, self.expectile).mean()
        loss = loss1 + loss2

        info = {
            'low_value_loss': loss.item(),
            'low_v_mean': v1.mean().item(),
            'low_v_max': v1.max().item(),
            'low_v_min': v1.min().item(),
        }

        return loss, info

    def high_value_loss(self, batch, current_step):
        """
        High-level value loss (long-horizon, with temporal abstraction).
        Looks `abstraction_factor` steps ahead for computing targets.

        Batch keys (matching reference ota.py):
            observations                   : s_t
            high_value_option_observations : s_{t+abstraction_factor}
            high_value_goals               : goal g (long-horizon)
            high_value_rewards             : mean edge cost (negated) over K_abs steps
            high_value_masks               : done mask at t+K
        """
        s        = batch['observations'].to(self.device)
        s_abs    = batch['high_value_option_observations'].to(self.device)
        g        = batch['high_value_goals'].to(self.device)
        r_abs    = batch['high_value_rewards'].to(self.device)
        mask_abs = batch['high_value_masks'].to(self.device)

        # Compute goal representations
        goal_rep = self.model.compute_goal_representation(s, g)
        goal_rep_abs = self.model.compute_goal_representation(s_abs, g)

        # Target values
        with torch.no_grad():
            v1_abs_t, v2_abs_t = self.model.get_target_high_value(s_abs, goal_rep_abs)
            v_abs_t = torch.minimum(v1_abs_t, v2_abs_t)
            q_target = r_abs + self.high_discount * mask_abs * v_abs_t

            v1_t, v2_t = self.model.get_target_high_value(s, goal_rep)
            v_t = (v1_t + v2_t) / 2
            adv = q_target - v_t

        # Current values
        v1, v2 = self.model.get_high_value(s, goal_rep)

        # Compute Q targets
        with torch.no_grad():
            q1_target = r_abs + self.high_discount * mask_abs * v1_abs_t
            q2_target = r_abs + self.high_discount * mask_abs * v2_abs_t

        # Expectile loss
        loss1 = self.expectile_loss(adv, q1_target - v1, self.expectile).mean()
        loss2 = self.expectile_loss(adv, q2_target - v2, self.expectile).mean()
        loss = loss1 + loss2

        info = {
            'high_value_loss': loss.item(),
            'high_v_mean': v1.mean().item(),
            'high_v_max': v1.max().item(),
            'high_v_min': v1.min().item(),
        }

        return loss, info

    def low_actor_loss(self, batch, current_step):
        """
        Low-level actor loss (AWR-style advantage-weighted behavioral cloning),
        aligned with the reference OTA implementation.

        Reference behavior:
        adv = V(s_next, g) - V(s, g)   (computed with ONLINE low_value)
        exp_a = exp(low_alpha * adv) clipped to 100
        actor_loss = -(exp_a * log_prob(a|s, phi([s; g]))).mean()

        Notes for your discrete setting:
        -log_prob == cross_entropy(logits, action_idx)
        """
        s      = batch['observations'].to(self.device)
        s_next = batch['next_observations'].to(self.device)
        g      = batch['low_actor_goals'].to(self.device)
        a      = batch['actions'].to(self.device)

        # --- Advantage with ONLINE critics (match reference) ---
        with torch.no_grad():
            v1, v2 = self.model.get_low_value(s, self.model.compute_goal_representation(s, g))
            nv1, nv2 = self.model.get_low_value(s_next, self.model.compute_goal_representation(s_next, g))
            v  = (v1 + v2) / 2.0
            nv = (nv1 + nv2) / 2.0
            adv = nv - v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            exp_a = torch.clamp(torch.exp(self.low_alpha * adv), max=100.0)

        # --- Goal representation for the actor: phi([s; g]) ---
        goal_rep = self.model.compute_goal_representation(s, g)

        # Optional: match config['low_actor_rep_grad'] (default False in OTA)
        low_actor_rep_grad = self.agent_params.get('low_actor_rep_grad', False)
        if not low_actor_rep_grad:
            goal_rep = goal_rep.detach()

        # --- Discrete policy logits and weighted BC loss ---
        action_logits = self.model.get_low_actor_logits(s, goal_rep)  # (B, action_dim)
        action_idx = a.argmax(dim=-1)                                 # (B,)
        ce_per_sample = F.cross_entropy(action_logits, action_idx, reduction='none')  # (B,)

        actor_loss = (exp_a * ce_per_sample).mean()

        info = {
            'low_actor_loss': actor_loss.item(),
            'low_actor_adv': adv.mean().item(),
            'low_actor_exp_a': exp_a.mean().item(),
            'low_actor_bc_ce': ce_per_sample.mean().item(),
        }
        return actor_loss, info

    def high_actor_loss(self, batch, current_step):
        """
        High-level actor loss (AWR-style advantage-weighted regression),
        aligned with the reference OTA implementation.

        Reference behavior:
        adv = V_H(s_target, g) - V_H(s, g)   (ONLINE high_value)
        exp_a = exp(high_alpha * adv) clipped to 100
        dist = Normal(mean=high_actor(s, g), std=const_std)
        target = phi([s; s_target])   (goal_rep on concatenation [obs, target])
        actor_loss = -(exp_a * log_prob(target | dist)).mean()

        IMPORTANT:
        target is phi([s; s_target])  NOT phi([s_target; g])
        In your API, compute_goal_representation(s, g) is assumed to implement phi([s; g]),
        so we set g := s_target to get phi([s; s_target]).
        """
        s        = batch['observations'].to(self.device)
        s_target = batch['high_actor_targets'].to(self.device)
        g        = batch['high_actor_goals'].to(self.device)

        # --- Advantage with ONLINE high-level critics (match reference) ---
        with torch.no_grad():
            v1, v2 = self.model.get_high_value(s, self.model.compute_goal_representation(s, g))
            nv1, nv2 = self.model.get_high_value(s_target, self.model.compute_goal_representation(s_target, g))
            v  = (v1 + v2) / 2.0
            nv = (nv1 + nv2) / 2.0
            adv = nv - v
            # Normalize advantages so exp_a doesn't saturate as value estimates grow
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            exp_a = torch.clamp(torch.exp(self.high_alpha * adv), max=100.0)

        # --- Actor predicts subgoal rep distribution params given (s, final_goal g) ---
        subgoal_rep_pred = self.model.get_high_actor_logits(s, self.model.compute_goal_representation(s, g))  # (B, rep_dim)

        const_std = float(self.agent_params.get('actor_std', 1.0))
        pred_dist = Normal(
            subgoal_rep_pred,
            torch.full_like(subgoal_rep_pred, const_std),
        )

        # --- Target subgoal representation: phi([s; s_target]) ---
        # Detach to match the reference (target computed without grad_params)
        target_rep = self.model.compute_goal_representation(s, s_target).detach()  # (B, rep_dim)

        nll_per_sample = -pred_dist.log_prob(target_rep).mean(dim=-1)  # (B,) — mean over rep_dim to avoid loss scaling with rep_dim
        actor_loss = (exp_a * nll_per_sample).mean()

        info = {
            'high_actor_loss': actor_loss.item(),
            'high_actor_adv': adv.mean().item(),
            'high_actor_exp_a': exp_a.mean().item(),
            'high_actor_nll': nll_per_sample.mean().item(),
            'high_actor_mse': ((subgoal_rep_pred - target_rep) ** 2).mean().item(),
        }
        return actor_loss, info

    def update(self, batch, current_step):
        """
        Perform one update step with all losses.
        
        Args:
            batch: training batch
            current_step: current training step
        
        Returns:
            loss: total loss
            info: dict with all logging information
        """
        self.optimizer.zero_grad()

        # Compute all losses
        low_v_loss, low_v_info = self.low_value_loss(batch, current_step)
        high_v_loss, high_v_info = self.high_value_loss(batch, current_step)
        low_a_loss, low_a_info = self.low_actor_loss(batch, current_step)
        high_a_loss, high_a_info = self.high_actor_loss(batch, current_step)

        # Total loss
        total_loss = low_v_loss + high_v_loss + low_a_loss + high_a_loss

        # Backward and optimize
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target networks
        self.model.update_target_networks(self.tau)

        # Compile info
        info = {
            'total_loss': total_loss.item(),
            **low_v_info,
            **high_v_info,
            **low_a_info,
            **high_a_info,
        }

        return total_loss, info

    def state_dict(self):
        """Save agent state"""
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load agent state"""
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
