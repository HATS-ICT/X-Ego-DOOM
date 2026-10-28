import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.data.tensor_specs import UnboundedContinuous
from torchrl.envs.transforms import ObservationTransform
from torchrl.envs.transforms.utils import _set_missing_tolerance


class AHWCToTensorResize(ObservationTransform):
    """(A,H,W,C) uint8 → (A,h,w,C) float32, normalised to [0,1], bilinear resize."""

    def __init__(
        self,
        key=("agent", "observation"),
        h: int = 72,
        w: int = 96,
        mode: str = "bilinear",
        antialias: bool = True,
    ):
        super().__init__(in_keys=[key], out_keys=[key])
        self.key = key
        self.h, self.w = int(h), int(w)
        self.mode = mode
        self.antialias = antialias

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)
        if obs.ndim != 4:
            raise ValueError(f"{self.key} must be 4-D AHWC, got {tuple(obs.shape)}")
        x = obs.to(torch.float32).div(255).permute(0, 3, 1, 2)  # A,C,H,W
        align = dict(align_corners=False) if self.mode in ("bilinear", "bicubic") else {}
        x = F.interpolate(
            x, size=(self.h, self.w), mode=self.mode, antialias=self.antialias, **align
        )
        return x.permute(0, 2, 3, 1).contiguous()  # A,h,w,C

    def transform_observation_spec(self, obs_spec: Composite) -> Composite:
        leaf = obs_spec[self.key]
        A, H, W, C = leaf.shape
        obs_spec[self.key] = UnboundedContinuous(
            shape=torch.Size([A, self.h, self.w, C]),
            device=leaf.device,
            dtype=torch.float32,
        )
        return obs_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset
