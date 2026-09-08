"""A compact PPO implementation (actor-critic, GAE, clipped surrogate loss)
for the PvP arena. One network instance controls every player-slot in
every arena simultaneously - since observations are self/team-relative,
this single shared policy IS the self-play opponent, no separate process
or league needed.

Action head layout:
  - 4 continuous dims (move_x, move_z, yaw_delta, pitch_delta): Gaussian
    with state-independent learned log-std, squashed through tanh to [-1, 1].
  - 4 binary dims (jump, attack, sprint, use_item): independent Bernoulli
    via sigmoid logits.
  - 1 categorical dim (physical hotbar slot, `HOTBAR_SLOTS` classes): a
    softmax over `kit::Item`. Selecting a different valid item on the same
    tick as `attack` triggers the vanilla attribute-swap bug.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Distribution, Normal

from features import OBS_DIM, CONTINUOUS_ACTION_DIM, BINARY_ACTION_DIM, HOTBAR_ACTION_DIM

# Every distribution here is fed values we constructed ourselves (a fresh
# sample, or a stored action being re-evaluated), so the per-call argument /
# sample validation is pure overhead on the rollout hot path.
Distribution.set_default_validate_args(False)

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

# Default trunk width / depth - a balanced default: enough capacity for
# nuanced combat, still small enough that CPU rollout collection stays
# latency-bound. Overridable per-run via train.py; the values used are
# saved into the checkpoint so export_model.py can rebuild the same shape.
HIDDEN = 256
NUM_LAYERS = 2


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_size: int = HIDDEN,
        num_layers: int = NUM_LAYERS,
        slot_dim: int = HOTBAR_ACTION_DIM,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.slot_dim = slot_dim

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.Tanh())
            in_dim = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.continuous_mean = nn.Linear(hidden_size, CONTINUOUS_ACTION_DIM)
        self.continuous_log_std = nn.Parameter(torch.zeros(CONTINUOUS_ACTION_DIM) - 0.5)
        self.binary_logits = nn.Linear(hidden_size, BINARY_ACTION_DIM)
        self.slot_logits = nn.Linear(hidden_size, slot_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = self.continuous_mean(h)
        # Every continuous action is tanh-squashed, so a std much above 1.0
        # in this raw pre-squash space makes the squashed action a near-
        # uniform +-1 coin flip regardless of the mean; cap log_std at 0.
        log_std = self.continuous_log_std.expand_as(mean).clamp(-3.0, 0.0)
        binary_logits = self.binary_logits(h)
        slot_logits = self.slot_logits(h)
        value = self.value_head(h).squeeze(-1)
        return mean, log_std, binary_logits, slot_logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """Samples an action for rollout collection.

        Runs every env step, so the sampling is done with plain tensor ops
        rather than `torch.distributions` objects (measurably cheaper on the
        tiny per-step batch). The log-prob formulas are exactly what
        `Normal` / `Bernoulli(logits=)` / `Categorical(logits=)` compute, so
        `evaluate()` - which still uses the distribution classes - reproduces
        these values (guarded by test_evaluate_matches_act_logprob).
        """
        mean, log_std, binary_logits, slot_logits, value = self.forward(obs)
        std = log_std.exp()

        # Continuous: reparameterised Gaussian sample + its log-prob.
        eps = torch.randn_like(mean)
        raw_cont = mean + std * eps
        cont_logprob = (-0.5 * eps.pow(2) - log_std - _HALF_LOG_2PI).sum(-1)
        squashed_cont = torch.tanh(raw_cont)

        # Binary: independent Bernoulli per dim from logits.
        binary_action = (torch.rand_like(binary_logits) < torch.sigmoid(binary_logits)).to(mean.dtype)
        binary_logprob = (
            binary_action * F.logsigmoid(binary_logits)
            + (1.0 - binary_action) * F.logsigmoid(-binary_logits)
        ).sum(-1)

        # Slot: one categorical draw from logits.
        slot_log_probs = F.log_softmax(slot_logits, dim=-1)
        slot_action = torch.multinomial(slot_log_probs.exp(), 1).squeeze(-1)
        slot_logprob = slot_log_probs.gather(-1, slot_action.unsqueeze(-1)).squeeze(-1)

        return {
            "squashed_cont": squashed_cont,
            "raw_cont": raw_cont,
            "binary_action": binary_action,
            "slot_action": slot_action,
            "logprob": cont_logprob + binary_logprob + slot_logprob,
            "value": value,
        }

    def evaluate(
        self,
        obs: torch.Tensor,
        raw_cont: torch.Tensor,
        binary_action: torch.Tensor,
        slot_action: torch.Tensor,
    ):
        """Recomputes log-probs/entropy/value for a PPO update pass."""
        mean, log_std, binary_logits, slot_logits, value = self.forward(obs)
        std = log_std.exp()

        cont_dist = Normal(mean, std)
        cont_logprob = cont_dist.log_prob(raw_cont).sum(-1)
        cont_entropy = cont_dist.entropy().sum(-1)

        bin_dist = Bernoulli(logits=binary_logits)
        binary_logprob = bin_dist.log_prob(binary_action).sum(-1)
        binary_entropy = bin_dist.entropy().sum(-1)

        slot_dist = Categorical(logits=slot_logits)
        slot_logprob = slot_dist.log_prob(slot_action)
        slot_entropy = slot_dist.entropy()

        logprob = cont_logprob + binary_logprob + slot_logprob
        entropy = cont_entropy + binary_entropy + slot_entropy
        return logprob, entropy, value


class RolloutBuffer:
    """Fixed-length rollout storage for `num_slots` parallel agents."""

    def __init__(self, rollout_len: int, num_slots: int, obs_dim: int, device: torch.device):
        self.rollout_len = rollout_len
        self.num_slots = num_slots
        self.device = device

        shape = (rollout_len, num_slots)
        self.obs = torch.zeros(*shape, obs_dim, device=device)
        self.raw_cont = torch.zeros(*shape, CONTINUOUS_ACTION_DIM, device=device)
        self.binary_action = torch.zeros(*shape, BINARY_ACTION_DIM, device=device)
        self.slot_action = torch.zeros(*shape, dtype=torch.long, device=device)
        self.logprob = torch.zeros(*shape, device=device)
        self.reward = torch.zeros(*shape, device=device)
        self.done = torch.zeros(*shape, device=device)
        self.value = torch.zeros(*shape, device=device)
        self.ptr = 0

    def add(self, obs, raw_cont, binary_action, slot_action, logprob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs
        self.raw_cont[i] = raw_cont
        self.binary_action[i] = binary_action
        self.slot_action[i] = torch.as_tensor(slot_action, device=self.device, dtype=torch.long)
        self.logprob[i] = logprob
        self.reward[i] = torch.as_tensor(reward, device=self.device)
        self.done[i] = torch.as_tensor(done, device=self.device, dtype=torch.float32)
        self.value[i] = value
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.rollout_len

    def reset(self):
        self.ptr = 0

    def to(self, device: torch.device) -> "RolloutBuffer":
        """Returns a shallow copy with every tensor moved to `device`
        (returns `self` unchanged if already there). Rollout collection runs
        on CPU (a per-step `.cpu().numpy()` would force a GPU sync every env
        step for a network too small to benefit); the whole filled buffer is
        moved here in one bulk transfer before the PPO update.
        """
        if device == self.device:
            return self
        other = RolloutBuffer.__new__(RolloutBuffer)
        other.rollout_len = self.rollout_len
        other.num_slots = self.num_slots
        other.device = device
        other.ptr = self.ptr
        for name in (
            "obs",
            "raw_cont",
            "binary_action",
            "slot_action",
            "logprob",
            "reward",
            "done",
            "value",
        ):
            setattr(other, name, getattr(self, name).to(device))
        return other

    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        advantages = torch.zeros_like(self.reward)
        last_gae = torch.zeros(self.num_slots, device=self.device)
        for t in reversed(range(self.rollout_len)):
            next_value = last_value if t == self.rollout_len - 1 else self.value[t + 1]
            next_nonterminal = 1.0 - self.done[t]
            delta = self.reward[t] + gamma * next_value * next_nonterminal - self.value[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.value
        return advantages, returns


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    epochs: int = 4,
    minibatch_size: int = 4096,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    sample_mask: torch.Tensor | None = None,
):
    """`sample_mask`: optional bool `[num_slots]` - when given, only slots
    marked `True` (the learner-controlled ones) contribute to the update.
    Opponent-controlled slots are still collected into the buffer (their
    transitions drive the sim) but must not train the policy toward imitating
    a frozen snapshot or the scripted bot."""
    T, N = buffer.rollout_len, buffer.num_slots
    obs = buffer.obs.reshape(T * N, -1)
    raw_cont = buffer.raw_cont.reshape(T * N, -1)
    binary_action = buffer.binary_action.reshape(T * N, -1)
    slot_action = buffer.slot_action.reshape(T * N)
    old_logprob = buffer.logprob.reshape(T * N)
    adv = advantages.reshape(T * N)
    ret = returns.reshape(T * N)

    if sample_mask is not None:
        # buffer tensors are [T, N, ...] flattened row-major, so slot n lives
        # at indices n, N+n, 2N+n, ... -> tiling the [N] mask T times lines up.
        keep = sample_mask.to(device=obs.device, dtype=torch.bool).repeat(T)
        obs, raw_cont, binary_action = obs[keep], raw_cont[keep], binary_action[keep]
        slot_action, old_logprob = slot_action[keep], old_logprob[keep]
        adv, ret = adv[keep], ret[keep]

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    total_size = obs.shape[0]
    # Accumulated as device tensors and pulled to Python floats once at the
    # end - `.item()` forces a host<->device sync.
    policy_loss_sum = torch.zeros((), device=obs.device)
    value_loss_sum = torch.zeros((), device=obs.device)
    entropy_sum = torch.zeros((), device=obs.device)
    approx_kl_sum = torch.zeros((), device=obs.device)
    clip_frac_sum = torch.zeros((), device=obs.device)
    num_updates = 0

    for _ in range(epochs):
        perm = torch.randperm(total_size, device=obs.device)
        for start in range(0, total_size, minibatch_size):
            idx = perm[start : start + minibatch_size]

            logprob, entropy, value = model.evaluate(
                obs[idx], raw_cont[idx], binary_action[idx], slot_action[idx]
            )
            logratio = logprob - old_logprob[idx]
            ratio = logratio.exp()

            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((value - ret[idx]) ** 2).mean()
            entropy_bonus = entropy.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                # Schulman's low-variance approx KL(old||new); clip fraction
                # is the share of samples the PPO ratio clamp actually bit.
                approx_kl_sum += ((ratio - 1.0) - logratio).mean()
                clip_frac_sum += ((ratio - 1.0).abs() > clip_ratio).float().mean()

            policy_loss_sum += policy_loss.detach()
            value_loss_sum += value_loss.detach()
            entropy_sum += entropy_bonus.detach()
            num_updates += 1

    denom = max(num_updates, 1)
    return {
        "policy_loss": (policy_loss_sum / denom).item(),
        "value_loss": (value_loss_sum / denom).item(),
        "entropy": (entropy_sum / denom).item(),
        "approx_kl": (approx_kl_sum / denom).item(),
        "clip_frac": (clip_frac_sum / denom).item(),
    }
