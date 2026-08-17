"""Mixed-dataset batch sampler with parametric size-weighted probability.

Follows BiomedParse (Zhao et al. 2025, Nature Methods, Methods L1659-1697):

    p_m(tau) = |D_m|^tau / sum_{m'} |D_{m'}|^tau

* tau = 1   : pure size-proportional
* tau = 0   : uniform across datasets
* tau = 1/2 : square-root of size

Schedule: tau = 1/2 for the first 25% of epochs, tau = 1 for the
middle 50%, and tau = 1/2 for the final 25%.
"""
from __future__ import annotations
import random
from typing import Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class TauScheduledMultiDatasetSampler(Sampler[int]):
    def __init__(
        self,
        dataset_sizes: Sequence[int],
        epochs_total: int = 50,
        schedule=(0.25, 0.50, 0.25),
        tau_warmup: float = 0.5,
        tau_main: float = 1.0,
        tau_cooldown: float = 0.5,
        seed: int = 42,
    ):
        self.sizes = list(dataset_sizes)
        self.M = len(self.sizes)
        self.total = sum(self.sizes)
        self.epochs_total = epochs_total

        assert abs(sum(schedule) - 1.0) < 1e-6, "schedule must sum to 1"
        self.warmup_end = round(epochs_total * schedule[0])
        self.main_end = self.warmup_end + round(epochs_total * schedule[1])

        self.lam_warm = tau_warmup
        self.lam_main = tau_main
        self.lam_cool = tau_cooldown
        self.seed = seed
        self._epoch = 0

        self._offsets = [0]
        for s in self.sizes[:-1]:
            self._offsets.append(self._offsets[-1] + s)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _current_tau(self) -> float:
        if self._epoch < self.warmup_end:
            return self.lam_warm
        elif self._epoch < self.main_end:
            return self.lam_main
        return self.lam_cool

    def _dataset_probs(self, lam: float) -> List[float]:
        weights = [s ** lam for s in self.sizes]
        Z = sum(weights)
        return [w / Z for w in weights]

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)
        torch_gen = torch.Generator()
        torch_gen.manual_seed(self.seed + self._epoch)

        lam = self._current_tau()
        probs = self._dataset_probs(lam)

        n_iters = self.total
        for _ in range(n_iters):
            m = _categorical_sample(probs, rng)
            local_idx = rng.randrange(self.sizes[m])
            yield self._offsets[m] + local_idx

    def __len__(self) -> int:
        return self.total


def _categorical_sample(probs: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


if __name__ == "__main__":
    # OmniTumorData cohort sizes (slice-level)
    sizes = [
        126353,  # BraTS
         33755,  # MSD Task01
          1373,  # LGG
         21315,  # AbdomenCT-1K
         19163,  # MSD Task03
          4965,  # MSD Task08
         20342,  # ULS23 Part 1
         35693,  # ULS23 Part 2
         23747,  # ULS23 Part 3
          8496,  # LUNA16
          3767,  # LNDb
          1844,  # COVID-19
    ]
    sampler = TauScheduledMultiDatasetSampler(sizes, epochs_total=50, seed=42)
    for ep in [0, 5, 12, 25, 38, 49]:
        sampler.set_epoch(ep)
        lam = sampler._current_tau()
        probs = sampler._dataset_probs(lam)
        print(f"epoch {ep:3d}  tau={lam:.2f}  "
              f"BraTS={probs[0]:.3f}  COVID={probs[-1]:.4f}  "
              f"min={min(probs):.4f}  max={max(probs):.3f}")
