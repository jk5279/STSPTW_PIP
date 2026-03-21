from dataclasses import dataclass
import torch
import os, pickle
import sys
import numpy as np
from collections import namedtuple
import math

__all__ = ['STSPTWEnv']
dg = sys.modules[__name__]

STSPTW_SET = namedtuple("STSPTW_SET",
                       ["node_loc",
                        "node_tw",
                        "durations", #set as 0
                        "service_window",
                        "time_factor", "loc_factor"])


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
    # Optional: pre-sampled travel times from current node to all nodes,
    # shape (batch, pomo, problem). Used when reveal_delay_before_action=True.
    next_travel_time: torch.Tensor = None


class STSPTWEnv:
    """
    Stochastic variant of TSPTWEnv.

    Deterministic travel time is identical to TSPTWEnv (L2 distance with speed=1).
    At each step, we add a stochastic delay term that depends on current_time and distance,
    loosely inspired by vrp-benchmarks/vrp_bench/travel_time_generator.py.
    """

    def __init__(self, **env_params):

        # Const @INIT
        self.problem = "STSPTW"
        self.env_params = env_params
        self.hardness = env_params['hardness']
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else env_params['device']

        # Deterministic travel component
        self.speed = 1.0

        # Stochastic delay controls (dimensionless, in same units as deterministic time)
        # These can be tuned; keep them modest for n=10.
        self.delay_scale = env_params.get('delay_scale', 0.1)  # relative magnitude vs deterministic travel
        self.time_scale = env_params.get('time_scale', 10.0)   # map normalized time -> pseudo clock [0, time_scale]

        # Whether to reveal edge-specific stochastic travel times to the agent
        # before action selection (pre-decision noise) or only realize them
        # after an action is chosen (post-decision noise, default).
        self.reveal_delay_before_action = env_params.get('reveal_delay_before_action', False)
        self.pre_sampled_pairwise_travel = None

        # Const @Load_Problem
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        self.node_xy = None
        self.node_service_time = None
        self.node_tw_start = None
        self.node_tw_end = None

        # Dynamic-1
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.timestamps = None
        self.infeasibility_list = None
        self.timeout_list = None

        # Dynamic-2
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

        # states to return
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    # -------- stochastic delay helpers (scalar, then lifted via torch ops) --------

    @staticmethod
    def _normal_pdf(x, mean, std):
        return torch.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))

    def _time_factor(self, current_time_norm):
        """
        current_time_norm in normalized units.
        Map to pseudo-clock [0, time_scale] then apply two Gaussian peaks.
        """
        t = current_time_norm * self.time_scale
        morning_peak = self._normal_pdf(t, 0.3 * self.time_scale, 0.06 * self.time_scale)
        evening_peak = self._normal_pdf(t, 0.7 * self.time_scale, 0.06 * self.time_scale)
        return 0.5 + 2.0 * (morning_peak + evening_peak)

    def _random_factor(self, current_time_norm):
        """
        Lognormal-like multiplicative noise; implemented via exp(N(mu, sigma^2)).
        """
        t = current_time_norm * self.time_scale
        rush = self._normal_pdf(t, 0.3 * self.time_scale, 0.06 * self.time_scale) + \
               self._normal_pdf(t, 0.7 * self.time_scale, 0.06 * self.time_scale)
        mu = 0.1 * rush
        sigma = 0.3 + 0.2 * rush
        eps = torch.randn_like(t)
        return torch.exp(mu + sigma * eps)

    def _sample_delay(self, distance, current_time_norm):
        """
        distance, current_time_norm: tensors with same shape.
        Returns delay in the same normalized time units as deterministic travel.
        """
        if self.delay_scale <= 0:
            return torch.zeros_like(distance)

        time_fac = self._time_factor(current_time_norm)
        distance_factor = 1 - torch.exp(-distance / 0.5)
        base_delay = 0.25 * time_fac * distance_factor
        rand_factor = self._random_factor(current_time_norm)
        delay = base_delay * rand_factor

        # scale relative to distance to keep variance controlled for small instances
        return self.delay_scale * delay

    def get_random_problems(self, batch_size, problem_size, coord_factor=100, max_tw_size=100):
        """Delegate to TSPTWEnv so Trainer can call env.get_random_problems(batch_size, problem_size)."""
        from . import TSPTWEnv
        temp = TSPTWEnv.TSPTWEnv(
            problem_size=problem_size,
            pomo_size=self.pomo_size,
            hardness=self.hardness,
            device=self.device,
            loc_scaler=self.loc_scaler,
        )
        return temp.get_random_problems(batch_size, problem_size, coord_factor=coord_factor, max_tw_size=max_tw_size)

    # -------- core env API (mirrors TSPTWEnv) --------

    def load_problems(self, batch_size, problems=None, aug_factor=1, normalize=True):
        from . import TSPTWEnv

        if problems is not None:
            node_xy, service_time, tw_start, tw_end = problems
        else:
            node_xy, service_time, tw_start, tw_end = self.get_random_problems(
                batch_size, self.problem_size, max_tw_size=100
            )

        if normalize:
            loc_factor = 100
            node_xy = node_xy / loc_factor
            tw_start = tw_start / loc_factor
            tw_end = tw_end / loc_factor
            tw_end[:, 0] = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw_end[:, 1:]).max(dim=-1)[0]

        self.batch_size = node_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                node_xy = TSPTWEnv.TSPTWEnv.augment_xy_data_by_8_fold(dg, node_xy)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
            else:
                raise NotImplementedError

        self.node_xy = node_xy
        self.node_service_time = service_time
        self.node_tw_start = tw_start
        self.node_tw_end = tw_end

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.node_xy = node_xy
        self.reset_state.node_service_time = service_time
        self.reset_state.node_tw_start = tw_start
        self.reset_state.node_tw_end = tw_end

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size + 1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.step_state.PROBLEM = self.problem

        # Neighbor structures reused from deterministic env
        k_sparse = self.env_params["k_sparse"]
        node_xy_expanded = node_xy[:, :, None, :]
        node_xy_expanded_T = node_xy[:, None, :, :]
        distances = torch.sqrt(torch.sum((node_xy_expanded - node_xy_expanded_T) ** 2, dim=-1))
        diag_mask = torch.eye(self.problem_size).unsqueeze(0).repeat(self.batch_size, 1, 1) * (1e9)
        distances += diag_mask

        if k_sparse < self.problem_size:
            self.is_sparse = True
            print("Sparse, ", k_sparse)
            _, topk_indices1 = torch.topk(distances, k=k_sparse, dim=-1, largest=False)
            dist_neighbors_index = torch.cat([
                torch.repeat_interleave(torch.arange(self.problem_size), repeats=k_sparse).reshape(1, self.problem_size, -1).repeat(self.batch_size, 1, 1).unsqueeze(-1),
                topk_indices1.unsqueeze(-1)
            ], dim=-1)

            start_node_tw_start = tw_start[:, :1]
            tw_start_differences = tw_start - start_node_tw_start
            tw_start_differences[tw_start_differences <= 0] = float('inf')
            _, topk_indices2 = torch.topk(tw_start_differences, k=k_sparse, dim=-1, largest=False)
            edge_index0 = torch.cat([
                torch.repeat_interleave(torch.tensor(0), repeats=k_sparse).reshape(1, -1).repeat(self.batch_size, 1).unsqueeze(-1),
                topk_indices2.unsqueeze(-1)
            ], dim=-1)

            start_times = tw_start[:, 1:].unsqueeze(-1).expand(-1, -1, self.problem_size - 1)
            end_times = tw_end[:, 1:].unsqueeze(-1).expand(-1, -1, self.problem_size - 1)
            start_max = torch.max(start_times, start_times.transpose(1, 2))
            end_min = torch.min(end_times, end_times.transpose(1, 2))
            overlap_matrix = torch.clamp(end_min - start_max, min=0)
            eye_matrix = torch.eye(self.problem_size - 1).unsqueeze(0).repeat(self.batch_size, 1, 1).bool()
            overlap_matrix[eye_matrix] = 0.
            del eye_matrix
            _, topk_indices3 = torch.topk(overlap_matrix, k=k_sparse, dim=-1)
            topk_indices3 += 1
            edge_index1 = torch.cat([
                torch.repeat_interleave(torch.arange(1, self.problem_size), repeats=k_sparse).reshape(1, self.problem_size - 1, -1).repeat(self.batch_size, 1, 1).unsqueeze(-1),
                topk_indices3.unsqueeze(-1)
            ], dim=-1)
            tw_neighbors_index = torch.concat([edge_index0.unsqueeze(1), edge_index1], dim=1)
            self.neighbour_index = torch.concat([dist_neighbors_index, tw_neighbors_index], dim=2)
            self.k_neigh_ninf_flag = torch.full((self.batch_size, self.problem_size, self.problem_size), float('-inf'))
            indices = self.neighbour_index.view(self.batch_size, -1, 2)
            self.k_neigh_ninf_flag[torch.arange(self.batch_size).view(-1, 1).expand_as(indices[:, :, 0]), indices[:, :, 0], indices[:, :, 1]] = 0
            self.k_neigh_ninf_flag[torch.arange(self.batch_size).view(-1, 1).expand_as(indices[:, :, 0]), indices[:, :, 1], indices[:, :, 0]] = 0
        else:
            self.is_sparse = False

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.timestamps = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.timeout_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.bool).to(self.device)

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.simulated_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.out_of_tw_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        self.infeasible = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        self.current_coord = self.node_xy[:, :1, :]

        # clear any pre-sampled travel times
        self.pre_sampled_pairwise_travel = None
        self.step_state.next_travel_time = None

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

        # Optionally sample and reveal stochastic travel times before action selection.
        if self.reveal_delay_before_action:
            # current_coord may have shape (batch, 1, 2) right after reset; expand if needed
            current_coord = self.current_coord
            if current_coord.size(1) == 1 and self.pomo_size > 1:
                current_coord = current_coord.expand(-1, self.pomo_size, -1)

            pairwise_dist = (current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1)
            pairwise_delay = self._sample_delay(pairwise_dist, self.current_time[:, :, None].expand_as(pairwise_dist))
            self.pre_sampled_pairwise_travel = (pairwise_dist + pairwise_delay) / self.speed
            self.step_state.next_travel_time = self.pre_sampled_pairwise_travel
        else:
            self.pre_sampled_pairwise_travel = None
            self.step_state.next_travel_time = None

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, visit_mask_only=True, out_reward=False, generate_PI_mask=False, use_predicted_PI_mask=False, pip_step=1):
        # Clamp selected to valid node indices (model may occasionally output OOB under masking)
        selected = selected.clamp(0, self.problem_size - 1)
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        current_coord = self.node_xy[torch.arange(self.batch_size)[:, None], selected]

        # Base geometric distance for logging / deterministic length
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)

        # Decide how to realize stochastic travel time for the chosen edge.
        if self.reveal_delay_before_action and self.pre_sampled_pairwise_travel is not None:
            # Use the pre-sampled travel times from pre_step for the selected edges.
            # pre_sampled_pairwise_travel shape: (batch, pomo, problem)
            flat_travel = self.pre_sampled_pairwise_travel.view(self.batch_size * self.pomo_size, self.problem_size)
            flat_selected = selected.view(-1)
            flat_indices = torch.arange(self.batch_size * self.pomo_size, device=self.device)
            step_travel = flat_travel[flat_indices, flat_selected].view(self.batch_size, self.pomo_size)
            effective_travel_time = step_travel
        else:
            # Default: sample delay only after the action is chosen (post-decision noise).
            current_time_norm = self.current_time
            delay = self._sample_delay(new_length, current_time_norm)
            effective_travel_time = (new_length + delay) / self.speed

        self.length = self.length + new_length
        self.current_coord = current_coord

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')

        self.current_time = (
            torch.max(
                self.current_time + effective_travel_time,
                self.node_tw_start[torch.arange(self.batch_size)[:, None], selected]
            )
            + self.node_service_time[torch.arange(self.batch_size)[:, None], selected]
        )
        self.timestamps = torch.cat((self.timestamps, self.current_time[:, :, None]), dim=2)

        round_error_epsilon = 0.00001
        # recompute arrival to all nodes using same noisy travel model
        pairwise_dist = (self.current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1)
        pairwise_delay = self._sample_delay(pairwise_dist, self.current_time[:, :, None].expand_as(pairwise_dist))
        pairwise_travel = (pairwise_dist + pairwise_delay) / self.speed
        next_arrival_time = torch.max(
            self.current_time[:, :, None] + pairwise_travel,
            self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1)
        )
        out_of_tw = next_arrival_time > self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1) + round_error_epsilon
        self.out_of_tw_ninf_flag[out_of_tw] = float('-inf')

        if generate_PI_mask and self.selected_count < self.problem_size - 1:
            self._calculate_PIP_mask(pip_step, selected, next_arrival_time)

        total_timeout = self.current_time - self.node_tw_end[torch.arange(self.batch_size)[:, None], selected]
        total_timeout = torch.where(total_timeout < 0, torch.zeros_like(total_timeout), total_timeout)
        self.timeout_list = torch.cat((self.timeout_list, total_timeout[:, :, None]), dim=2)

        self.ninf_mask = self.visited_ninf_flag.clone()
        if not visit_mask_only:
            self.ninf_mask[out_of_tw] = float('-inf')
        if generate_PI_mask and self.selected_count < self.problem_size - 1 and (not use_predicted_PI_mask):
            self.ninf_mask = torch.where(self.simulated_ninf_flag == float('-inf'), float('-inf'), self.ninf_mask)
            all_infsb = ((self.ninf_mask == float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1, -1, self.problem_size)
            self.ninf_mask = torch.where(all_infsb, self.visited_ninf_flag, self.ninf_mask)

        newly_infeasible = (((self.visited_ninf_flag == 0).int() + (self.out_of_tw_ninf_flag == float('-inf')).int()) == 2).any(dim=2)
        self.infeasible = self.infeasible + newly_infeasible
        self.infeasibility_list = torch.cat((self.infeasibility_list, self.infeasible[:, :, None]), dim=2)
        infeasible = 0.

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + newly_finished

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        done = self.finished.all()
        if done:
            if not out_reward:
                reward = -self._get_travel_distance()
            else:
                dist_reward = -self._get_travel_distance()
                total_timeout_reward = -self.timeout_list.sum(dim=-1)
                timeout_nodes_reward = -torch.where(self.timeout_list > 0, torch.ones_like(self.timeout_list), self.timeout_list).sum(-1).int()
                reward = [dist_reward, total_timeout_reward, timeout_nodes_reward]
            infeasible = self.infeasible
        else:
            reward = None

        return self.step_state, reward, done, infeasible

    def _calculate_PIP_mask(self, pip_step, selected, next_arrival_time):
        round_error_epsilon = 0.00001
        node_tw_end = self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1)

        if pip_step == 0:
            if self.is_sparse:
                print("Warning! Performing zero-step PIP masking on k nearest neighbors is not supported! Consider all the unvisited nodes instead!")
            out_of_tw = next_arrival_time > node_tw_end + round_error_epsilon  # (batch, pomo, problem)
            self.simulated_ninf_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
            self.simulated_ninf_flag[out_of_tw] = float('-inf')
        elif pip_step == 1:
            # 이 부분은 TSPTWEnv._calculate_PIP_mask의 pip_step==1 구현을 그대로 따른다.
            if self.is_sparse:  # calculate the k_sparse_mask
                self.not_neigh_ninf_flag = self.k_neigh_ninf_flag[:, None, :, :].repeat(1, self.pomo_size, 1, 1)

                self.not_neigh_ninf_flag = self.not_neigh_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected]
                # shape: (batch, pomo, problem)

                # Mark the unvisited neighbors as True
                self.visited_and_notneigh_ninf_flag = (self.not_neigh_ninf_flag == 0) & (self.visited_ninf_flag == 0)  # (B, P, N)
                # calculate the count of the unvisited neighbors for each instance
                unvisited_and_neigh_counts = self.visited_and_notneigh_ninf_flag.sum(dim=-1)  # (B, P)
                max_count = unvisited_and_neigh_counts.max().item()  # (self.batch_size, P, N)
                # extract the unvisited neighbors
                _, unvisited_and_neigh = torch.topk(self.visited_and_notneigh_ninf_flag.int(), dim=-1,
                                                    largest=True, k=max_count)  # (B, P, N)
                unvisited = unvisited_and_neigh.sort(dim=-1)[0]  # shape: (batch, pomo, max_count)
                del unvisited_and_neigh, unvisited_and_neigh_counts

            else:  # all the unvisited nodes will be considered
                unvisited = torch.masked_select(
                    torch.arange(self.problem_size).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.pomo_size, self.problem_size),
                    self.visited_ninf_flag != float('-inf')
                ).reshape(self.batch_size, self.pomo_size, -1)

            simulate_size = unvisited.size(-1)
            two_step_unvisited = unvisited.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            diag_element = torch.eye(simulate_size).view(1, 1, simulate_size, simulate_size).repeat(self.batch_size, self.pomo_size, 1, 1)
            two_step_idx = torch.masked_select(two_step_unvisited, diag_element == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1)

            # add arrival_time of the first-step nodes
            first_step_current_coord = self.node_xy.unsqueeze(1).repeat(1, self.pomo_size, 1, 1).gather(
                dim=2, index=unvisited.unsqueeze(3).expand(-1, -1, -1, 2)
            )
            first_step_arrival_time = next_arrival_time.gather(dim=-1, index=unvisited)

            # add arrival_time of the second-step nodes
            two_step_tw_end = node_tw_end.gather(dim=-1, index=unvisited)
            two_step_tw_end = two_step_tw_end.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            two_step_tw_end = torch.masked_select(two_step_tw_end, diag_element == 0).reshape(
                self.batch_size, self.pomo_size, simulate_size, -1
            )

            node_tw_start = self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1)
            two_step_tw_start = node_tw_start.gather(dim=-1, index=unvisited)
            two_step_tw_start = two_step_tw_start.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            two_step_tw_start = torch.masked_select(two_step_tw_start, diag_element == 0).reshape(
                self.batch_size, self.pomo_size, simulate_size, -1
            )

            node_service_time = self.node_service_time[:, None, :].expand(-1, self.pomo_size, -1)
            two_step_node_service_time = node_service_time.gather(dim=-1, index=unvisited)
            two_step_node_service_time = two_step_node_service_time.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            two_step_node_service_time = torch.masked_select(two_step_node_service_time, diag_element == 0).reshape(
                self.batch_size, self.pomo_size, simulate_size, -1
            )

            two_step_current_coord = first_step_current_coord.unsqueeze(2).repeat(1, 1, simulate_size, 1, 1)
            two_step_current_coord = torch.masked_select(
                two_step_current_coord,
                diag_element.unsqueeze(-1).expand(-1, -1, -1, -1, 2) == 0
            ).reshape(self.batch_size, self.pomo_size, simulate_size, -1, 2)
            second_step_new_length = (
                two_step_current_coord
                - first_step_current_coord.unsqueeze(3).repeat(1, 1, 1, simulate_size - 1, 1)
            ).norm(p=2, dim=-1)
            first_step_arrival_time = first_step_arrival_time.unsqueeze(-1).repeat(1, 1, 1, simulate_size - 1)
            second_step_arrival_time = torch.max(
                first_step_arrival_time + second_step_new_length / self.speed,
                two_step_tw_start
            ) + two_step_node_service_time

            # time window constraint
            infeasible_mark = (second_step_arrival_time > two_step_tw_end + round_error_epsilon)
            selectable = (infeasible_mark == False).all(dim=-1)

            # mark the selectable unvisited nodes
            self.simulated_ninf_flag = torch.full((self.batch_size, self.pomo_size, self.problem_size), float('-inf'))
            selected_indices = selectable.nonzero(as_tuple=False)
            if selected_indices.numel() > 0:
                unvisited_indices = unvisited[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]
                self.simulated_ninf_flag[selected_indices[:, 0], selected_indices[:, 1], unvisited_indices] = 0.

    def _get_travel_distance(self):
        return self.length

    def load_dataset(self, path, offset=0, num_samples=10000, disable_print=True):
        """
        Reuse the same dataset format as TSPTWEnv: list of (node_xy, service_time, tw_start, tw_end).
        """
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset + num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        node_xy, service_time, tw_start, tw_end = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
        node_xy, service_time, tw_start, tw_end = torch.Tensor(node_xy), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)

        data = (node_xy, service_time, tw_start, tw_end)
        return data

