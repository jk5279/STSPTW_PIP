"""
STSPTWEnv_v2: Stochastic TSPTW Environment (Two-Layer Design)

Layer 1 (Backbone): Deterministic TSPTW instances from TSPTWEnv (unchanged)
Layer 2 (Stochastic Overlay): Mean-preserving noise on travel times

Supported distributions:
  - "gamma": Gamma(k, mu/k) with CV = 1/sqrt(k)  [Tas et al., EJOR 2014]
  - "two_point": Bernoulli mixture               [Zhang et al., OR]

Design principles:
  - E[t_ij] = d_ij  (mean-preserving perturbation)
  - out_of_tw uses deterministic distances (monotonic accumulation is correct)
  - PIP mask uses deterministic lookahead (baseline)
  - Reward = negative geometric tour length (same objective as TSPTWEnv)
  - CV = 0 recovers exact TSPTWEnv behavior (sanity check)
"""

from dataclasses import dataclass
import torch
import os, pickle
import sys
import numpy as np
from collections import namedtuple

__all__ = ['STSPTWEnv_v2']
dg = sys.modules[__name__]

STSPTW_SET = namedtuple("STSPTW_SET",
                        ["node_loc", "node_tw", "durations",
                         "service_window", "time_factor", "loc_factor"])


@dataclass
class Reset_State:
    node_xy: torch.Tensor = None
    node_service_time: torch.Tensor = None
    node_tw_start: torch.Tensor = None
    node_tw_end: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    PROBLEM: str = None
    selected_count: int = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None
    infeasible: torch.Tensor = None
    current_time: torch.Tensor = None
    length: torch.Tensor = None
    current_coord: torch.Tensor = None
    # shape (batch, pomo, problem). Pre-sampled stochastic travel times
    # exposed to model when pre-decision noise is enabled.
    next_travel_time: torch.Tensor = None


class STSPTWEnv_v2:

    def __init__(self, **env_params):
        # Core params (identical to TSPTWEnv)
        self.problem = "STSPTW"
        self.env_params = env_params
        self.hardness = env_params['hardness']
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.loc_scaler = env_params.get('loc_scaler', None)
        self.device = (torch.device('cuda', torch.cuda.current_device())
                       if 'device' not in env_params else env_params['device'])
        self.speed = 1.0

        # --- Stochastic overlay params ---
        self.noise_type = env_params.get('noise_type', 'gamma')
        self.cv = env_params.get('cv', 0.5)          # CV = 1/sqrt(k); 0 = deterministic
        # alpha and n_mc_samples reserved for future MC-PIP extension
        # (currently using deterministic PIP baseline)

        # Two-point distribution params
        self.two_point_delta = env_params.get('two_point_delta', 0.3)
        self.two_point_p = env_params.get('two_point_p', 0.5)

        # Pre-decision noise: sample and reveal travel times before action
        self.reveal_delay_before_action = env_params.get('reveal_delay_before_action', False)
        self.pre_sampled_pairwise_travel = None

        # Problem data (set at load_problems)
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        self.node_xy = None
        self.node_service_time = None
        self.node_tw_start = None
        self.node_tw_end = None

        # Dynamic state
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.timestamps = None
        self.infeasibility_list = None
        self.timeout_list = None

        # Masks and flags
        self.visited_ninf_flag = None
        self.simulated_ninf_flag = None
        self.global_mask = None
        self.global_mask_ninf_flag = None
        self.out_of_tw_ninf_flag = None
        self.ninf_mask = None
        self.finished = None
        self.infeasible = None
        self.current_time = None
        self.length = None
        self.current_coord = None

        # States to return
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    # =================================================================
    #  Stochastic travel-time sampling
    # =================================================================

    def _sample_travel_time(self, distance):
        """
        Sample a single realization of stochastic travel time per edge.

        Args:
            distance: tensor of deterministic L2 distances (any shape)
        Returns:
            travel times with same shape; E[t] = distance / speed
        """
        if self.cv <= 0:
            return distance / self.speed

        safe_dist = distance.clamp(min=1e-8)

        if self.noise_type == 'gamma':
            k = 1.0 / (self.cv ** 2)  # shape param; CV = 1/sqrt(k)
            concentration = torch.full_like(safe_dist, k)
            rate = k / safe_dist       # E[X] = k / rate = safe_dist
            travel = torch.distributions.Gamma(concentration, rate).sample()

        elif self.noise_type == 'two_point':
            delta = self.two_point_delta
            p = self.two_point_p
            epsilon = p * delta / (1 - p)  # mean-preserving
            coin = torch.rand_like(safe_dist)
            travel = torch.where(
                coin < p,
                safe_dist * (1 - delta),
                safe_dist * (1 + epsilon),
            )
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

        # Zero distance → zero travel
        travel = torch.where(distance < 1e-8, torch.zeros_like(travel), travel)
        return travel / self.speed

    # =================================================================
    #  Problem loading (delegates to TSPTWEnv backbone)
    # =================================================================

    def get_random_problems(self, batch_size, problem_size,
                            coord_factor=100, max_tw_size=100):
        from . import TSPTWEnv
        temp = TSPTWEnv.TSPTWEnv(
            problem_size=problem_size,
            pomo_size=self.pomo_size,
            hardness=self.hardness,
            device=self.device,
            loc_scaler=self.loc_scaler,
            k_sparse=self.env_params.get("k_sparse", problem_size),
        )
        return temp.get_random_problems(
            batch_size, problem_size,
            coord_factor=coord_factor, max_tw_size=max_tw_size)

    def load_problems(self, batch_size, problems=None,
                      aug_factor=1, normalize=True):
        from . import TSPTWEnv

        if problems is not None:
            node_xy, service_time, tw_start, tw_end = problems
        else:
            node_xy, service_time, tw_start, tw_end = \
                self.get_random_problems(batch_size, self.problem_size,
                                         max_tw_size=100)

        if normalize:
            loc_factor = 100
            node_xy = node_xy / loc_factor
            tw_start = tw_start / loc_factor
            tw_end = tw_end / loc_factor
            tw_end[:, 0] = (
                torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1)
                + tw_end[:, 1:]
            ).max(dim=-1)[0]

        self.batch_size = node_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                node_xy = TSPTWEnv.TSPTWEnv.augment_xy_data_by_8_fold(
                    dg, node_xy)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
            else:
                raise NotImplementedError

        self.node_xy = node_xy
        self.node_service_time = service_time
        self.node_tw_start = tw_start
        self.node_tw_end = tw_end

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(
            self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(
            self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.node_xy = node_xy
        self.reset_state.node_service_time = service_time
        self.reset_state.node_tw_start = tw_start
        self.reset_state.node_tw_end = tw_end

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(
            start=1, end=self.pomo_size + 1
        )[None, :].expand(self.batch_size, -1).to(self.device)
        self.step_state.PROBLEM = self.problem

        # ---- Neighbor structures (identical to TSPTWEnv) ----
        k_sparse = self.env_params["k_sparse"]
        node_xy_expanded = node_xy[:, :, None, :]
        node_xy_expanded_T = node_xy[:, None, :, :]
        distances = torch.sqrt(
            torch.sum((node_xy_expanded - node_xy_expanded_T) ** 2, dim=-1))
        diag_mask = (torch.eye(self.problem_size).unsqueeze(0)
                     .repeat(self.batch_size, 1, 1) * 1e9)
        distances += diag_mask

        if k_sparse < self.problem_size:
            self.is_sparse = True
            print("Sparse, ", k_sparse)

            _, topk_indices1 = torch.topk(
                distances, k=k_sparse, dim=-1, largest=False)
            dist_neighbors_index = torch.cat([
                torch.repeat_interleave(
                    torch.arange(self.problem_size), repeats=k_sparse
                ).reshape(1, self.problem_size, -1
                ).repeat(self.batch_size, 1, 1).unsqueeze(-1),
                topk_indices1.unsqueeze(-1)
            ], dim=-1)

            start_node_tw_start = tw_start[:, :1]
            tw_start_differences = tw_start - start_node_tw_start
            tw_start_differences[tw_start_differences <= 0] = float('inf')
            _, topk_indices2 = torch.topk(
                tw_start_differences, k=k_sparse, dim=-1, largest=False)
            edge_index0 = torch.cat([
                torch.repeat_interleave(
                    torch.tensor(0), repeats=k_sparse
                ).reshape(1, -1).repeat(self.batch_size, 1).unsqueeze(-1),
                topk_indices2.unsqueeze(-1)
            ], dim=-1)

            start_times = tw_start[:, 1:].unsqueeze(-1).expand(
                -1, -1, self.problem_size - 1)
            end_times = tw_end[:, 1:].unsqueeze(-1).expand(
                -1, -1, self.problem_size - 1)
            start_max = torch.max(start_times, start_times.transpose(1, 2))
            end_min = torch.min(end_times, end_times.transpose(1, 2))
            overlap_matrix = torch.clamp(end_min - start_max, min=0)
            eye_matrix = (torch.eye(self.problem_size - 1).unsqueeze(0)
                          .repeat(self.batch_size, 1, 1).bool())
            overlap_matrix[eye_matrix] = 0.
            del eye_matrix
            _, topk_indices3 = torch.topk(
                overlap_matrix, k=k_sparse, dim=-1)
            topk_indices3 += 1
            edge_index1 = torch.cat([
                torch.repeat_interleave(
                    torch.arange(1, self.problem_size), repeats=k_sparse
                ).reshape(1, self.problem_size - 1, -1
                ).repeat(self.batch_size, 1, 1).unsqueeze(-1),
                topk_indices3.unsqueeze(-1)
            ], dim=-1)

            tw_neighbors_index = torch.concat(
                [edge_index0.unsqueeze(1), edge_index1], dim=1)
            self.neighbour_index = torch.concat(
                [dist_neighbors_index, tw_neighbors_index], dim=2)
            self.k_neigh_ninf_flag = torch.full(
                (self.batch_size, self.problem_size, self.problem_size),
                float('-inf'))
            indices = self.neighbour_index.view(self.batch_size, -1, 2)
            batch_arange = torch.arange(self.batch_size).view(-1, 1).expand_as(
                indices[:, :, 0])
            self.k_neigh_ninf_flag[
                batch_arange, indices[:, :, 0], indices[:, :, 1]] = 0
            self.k_neigh_ninf_flag[
                batch_arange, indices[:, :, 1], indices[:, :, 0]] = 0
        else:
            self.is_sparse = False

    # =================================================================
    #  Reset / pre_step / step
    # =================================================================

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0), dtype=torch.long
        ).to(self.device)
        self.timestamps = torch.zeros(
            (self.batch_size, self.pomo_size, 0)).to(self.device)
        self.timeout_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0)).to(self.device)
        self.infeasibility_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0), dtype=torch.bool
        ).to(self.device)

        self.visited_ninf_flag = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size)
        ).to(self.device)
        self.simulated_ninf_flag = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size)
        ).to(self.device)
        self.global_mask_ninf_flag = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size)
        ).to(self.device)
        self.global_mask = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size)
        ).to(self.device)
        self.out_of_tw_ninf_flag = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size)
        ).to(self.device)
        self.ninf_mask = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size)
        ).to(self.device)
        self.finished = torch.zeros(
            size=(self.batch_size, self.pomo_size), dtype=torch.bool
        ).to(self.device)
        self.infeasible = torch.zeros(
            size=(self.batch_size, self.pomo_size), dtype=torch.bool
        ).to(self.device)
        self.current_time = torch.zeros(
            size=(self.batch_size, self.pomo_size)).to(self.device)
        self.length = torch.zeros(
            size=(self.batch_size, self.pomo_size)).to(self.device)
        self.current_coord = self.node_xy[:, :1, :]

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        if self.reveal_delay_before_action:
            # Pre-decision: sample stochastic travel times for all edges
            current_coord = self.current_coord
            if current_coord.size(1) == 1 and self.pomo_size > 1:
                current_coord = current_coord.expand(-1, self.pomo_size, -1)
            pairwise_dist = (
                current_coord[:, :, None, :]
                - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            ).norm(p=2, dim=-1)
            self.pre_sampled_pairwise_travel = self._sample_travel_time(pairwise_dist)
            self.step_state.next_travel_time = self.pre_sampled_pairwise_travel
        else:
            self.step_state.next_travel_time = None

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, visit_mask_only=True, out_reward=False,
             generate_PI_mask=False, use_predicted_PI_mask=False, pip_step=1):
        # ---- Dynamic-1: book-keeping (identical to TSPTWEnv) ----
        selected = selected.clamp(0, self.problem_size - 1)
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2)

        current_coord = self.node_xy[
            torch.arange(self.batch_size)[:, None], selected]

        # ---- Geometric distance (for reward) ----
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)

        # ---- Stochastic travel time (for time progression) ----
        if self.reveal_delay_before_action and self.pre_sampled_pairwise_travel is not None:
            # Use pre-sampled travel times from pre_step
            flat_travel = self.pre_sampled_pairwise_travel.view(
                self.batch_size * self.pomo_size, self.problem_size)
            flat_selected = selected.view(-1)
            flat_indices = torch.arange(
                self.batch_size * self.pomo_size, device=self.device)
            stochastic_travel = flat_travel[
                flat_indices, flat_selected].view(self.batch_size, self.pomo_size)
        else:
            stochastic_travel = self._sample_travel_time(new_length)

        self.length = self.length + new_length   # geometric, for reward
        self.current_coord = current_coord

        # ---- Mark visited ----
        self.visited_ninf_flag[
            self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')

        # ---- Update current_time with STOCHASTIC travel ----
        self.current_time = (
            torch.max(
                self.current_time + stochastic_travel,
                self.node_tw_start[
                    torch.arange(self.batch_size)[:, None], selected]
            )
            + self.node_service_time[
                torch.arange(self.batch_size)[:, None], selected]
        )
        self.timestamps = torch.cat(
            (self.timestamps, self.current_time[:, :, None]), dim=2)

        # ---- out_of_tw: deterministic distances, stochastic current_time ----
        # Monotonic accumulation is correct here because current_time only
        # increases and deterministic distances are fixed.  A node that becomes
        # unreachable stays unreachable.
        round_error_epsilon = 0.00001
        det_pairwise_dist = (
            self.current_coord[:, :, None, :]
            - self.node_xy[:, None, :, :].expand(
                -1, self.pomo_size, -1, -1)
        ).norm(p=2, dim=-1)

        next_arrival_time = torch.max(
            self.current_time[:, :, None] + det_pairwise_dist / self.speed,
            self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
        out_of_tw = (
            next_arrival_time
            > self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1)
            + round_error_epsilon)
        self.out_of_tw_ninf_flag[out_of_tw] = float('-inf')

        # ---- PIP mask (stochastic Monte Carlo version) ----
        if generate_PI_mask and self.selected_count < self.problem_size - 1:
            self._calculate_PIP_mask(
                pip_step, selected, next_arrival_time, det_pairwise_dist)

        # ---- Timeout ----
        total_timeout = (
            self.current_time
            - self.node_tw_end[
                torch.arange(self.batch_size)[:, None], selected])
        total_timeout = torch.where(
            total_timeout < 0, torch.zeros_like(total_timeout), total_timeout)
        self.timeout_list = torch.cat(
            (self.timeout_list, total_timeout[:, :, None]), dim=2)

        # ---- Build action mask ----
        self.ninf_mask = self.visited_ninf_flag.clone()
        if not visit_mask_only:
            self.ninf_mask[out_of_tw] = float('-inf')
        if (generate_PI_mask
                and self.selected_count < self.problem_size - 1
                and (not use_predicted_PI_mask)):
            self.ninf_mask = torch.where(
                self.simulated_ninf_flag == float('-inf'),
                float('-inf'), self.ninf_mask)
            all_infsb = (
                (self.ninf_mask == float('-inf')).all(dim=-1)
            ).unsqueeze(-1).expand(-1, -1, self.problem_size)
            self.ninf_mask = torch.where(
                all_infsb, self.visited_ninf_flag, self.ninf_mask)

        # ---- Infeasibility detection ----
        newly_infeasible = (
            ((self.visited_ninf_flag == 0).int()
             + (self.out_of_tw_ninf_flag == float('-inf')).int()) == 2
        ).any(dim=2)
        self.infeasible = self.infeasible + newly_infeasible
        self.infeasibility_list = torch.cat(
            (self.infeasibility_list, self.infeasible[:, :, None]), dim=2)
        infeasible = 0.

        newly_finished = (
            self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished

        # ---- Update step state ----
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        # ---- Return ----
        done = self.finished.all()
        if done:
            if not out_reward:
                reward = -self._get_travel_distance()
            else:
                dist_reward = -self._get_travel_distance()
                total_timeout_reward = -self.timeout_list.sum(dim=-1)
                timeout_nodes_reward = -torch.where(
                    self.timeout_list > 0,
                    torch.ones_like(self.timeout_list),
                    self.timeout_list).sum(-1).int()
                reward = [dist_reward, total_timeout_reward,
                          timeout_nodes_reward]
            infeasible = self.infeasible
        else:
            reward = None

        return self.step_state, reward, done, infeasible

    # =================================================================
    #  Stochastic PIP mask  (Monte Carlo)
    # =================================================================

    def _calculate_PIP_mask(self, pip_step, selected,
                            next_arrival_time, det_pairwise_dist):
        """
        Deterministic PIP mask (same as TSPTWEnv baseline).

        Uses deterministic distances for lookahead — does NOT account for
        stochastic travel times. This is intentional: we are evaluating how
        the deterministic PIP baseline performs under stochastic noise.
        """
        round_error_epsilon = 0.00001
        node_tw_end = self.node_tw_end[:, None, :].expand(
            -1, self.pomo_size, -1)

        if pip_step == 0:
            # Zero-step: pure reachability (deterministic)
            if self.is_sparse:
                print("Warning! pip_step=0 with k_sparse is not supported!")
            out_of_tw = (
                next_arrival_time > node_tw_end + round_error_epsilon)
            self.simulated_ninf_flag = torch.zeros(
                (self.batch_size, self.pomo_size, self.problem_size))
            self.simulated_ninf_flag[out_of_tw] = float('-inf')

        elif pip_step == 1:
            # One-step deterministic lookahead (same logic as TSPTWEnv)

            if self.is_sparse:
                self.not_neigh_ninf_flag = (
                    self.k_neigh_ninf_flag[:, None, :, :]
                    .repeat(1, self.pomo_size, 1, 1))
                self.not_neigh_ninf_flag = self.not_neigh_ninf_flag[
                    self.BATCH_IDX, self.POMO_IDX, selected]
                self.visited_and_notneigh_ninf_flag = (
                    (self.not_neigh_ninf_flag == 0)
                    & (self.visited_ninf_flag == 0))
                counts = self.visited_and_notneigh_ninf_flag.sum(dim=-1)
                max_count = counts.max().item()
                _, topk = torch.topk(
                    self.visited_and_notneigh_ninf_flag.int(),
                    dim=-1, largest=True, k=max_count)
                unvisited = topk.sort(dim=-1)[0]
                del topk, counts
            else:
                unvisited = torch.masked_select(
                    torch.arange(self.problem_size).unsqueeze(0).unsqueeze(0)
                        .expand(self.batch_size, self.pomo_size,
                                self.problem_size),
                    self.visited_ninf_flag != float('-inf')
                ).reshape(self.batch_size, self.pomo_size, -1)

            simulate_size = unvisited.size(-1)
            two_step_unvisited = unvisited.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            diag_element = torch.eye(simulate_size).view(1, 1, simulate_size, simulate_size).repeat(self.batch_size, self.pomo_size, 1, 1)
            two_step_idx = torch.masked_select(two_step_unvisited, diag_element == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1)

            # First-step: coordinates and arrival time
            first_step_current_coord = self.node_xy.unsqueeze(1).repeat(1, self.pomo_size, 1, 1).gather(dim=2, index=unvisited.unsqueeze(3).expand(-1, -1, -1, 2))
            first_step_arrival_time = next_arrival_time.gather(dim=-1, index=unvisited)

            # Second-step: tw_end
            two_step_tw_end = node_tw_end.gather(dim=-1, index=unvisited)
            two_step_tw_end = two_step_tw_end.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            two_step_tw_end = torch.masked_select(two_step_tw_end, diag_element == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1)

            # Second-step: tw_start
            node_tw_start = self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1)
            two_step_tw_start = node_tw_start.gather(dim=-1, index=unvisited)
            two_step_tw_start = two_step_tw_start.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            two_step_tw_start = torch.masked_select(two_step_tw_start, diag_element == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1)

            # Second-step: service time
            node_service_time = self.node_service_time[:, None, :].expand(-1, self.pomo_size, -1)
            two_step_node_service_time = node_service_time.gather(dim=-1, index=unvisited)
            two_step_node_service_time = two_step_node_service_time.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            two_step_node_service_time = torch.masked_select(two_step_node_service_time, diag_element == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1)

            # Second-step: coordinates and deterministic travel
            two_step_current_coord = first_step_current_coord.unsqueeze(2).repeat(1, 1, simulate_size, 1, 1)
            two_step_current_coord = torch.masked_select(two_step_current_coord, diag_element.unsqueeze(-1).expand(-1, -1, -1, -1, 2) == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1, 2)
            second_step_new_length = (two_step_current_coord - first_step_current_coord.unsqueeze(3).repeat(1, 1, 1, simulate_size - 1, 1)).norm(p=2, dim=-1)
            first_step_arrival_time = first_step_arrival_time.unsqueeze(-1).repeat(1, 1, 1, simulate_size - 1)
            second_step_arrival_time = torch.max(first_step_arrival_time + second_step_new_length / self.speed, two_step_tw_start) + two_step_node_service_time

            # Feasibility check: deterministic
            infeasible_mark = (second_step_arrival_time > two_step_tw_end + round_error_epsilon)
            selectable = (infeasible_mark == False).all(dim=-1)

            # Write to simulated_ninf_flag
            self.simulated_ninf_flag = torch.full((self.batch_size, self.pomo_size, self.problem_size), float('-inf'))
            selected_indices = selectable.nonzero(as_tuple=False)
            unvisited_indices = unvisited[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]
            self.simulated_ninf_flag[selected_indices[:, 0], selected_indices[:, 1], unvisited_indices] = 0.

        else:
            raise NotImplementedError(
                f"pip_step={pip_step} not implemented")

    # =================================================================
    #  Reward / utility
    # =================================================================

    def _get_travel_distance(self):
        return self.length

    def load_dataset(self, path, offset=0, num_samples=10000,
                     disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", \
            "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset + num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(
                    len(data), type(data), path))
        node_xy = [i[0] for i in data]
        service_time = [i[1] for i in data]
        tw_start = [i[2] for i in data]
        tw_end = [i[3] for i in data]
        data = (torch.Tensor(node_xy), torch.Tensor(service_time),
                torch.Tensor(tw_start), torch.Tensor(tw_end))
        return data
