from dataclasses import dataclass
from math import prod
from typing import Optional, Sequence, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torch import nn
from torchrl.data import TensorSpec

from benchmarl.models.common import Model, ModelConfig
from doom_marl.models.resnet import _apply_shared_module, _make_mlp


class _HFDinoV3Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_name: str,
        image_size: int | Sequence[int],
        freeze_backbone: bool,
        normalize: bool,
        feature_source: str,
        local_files_only: bool,
        encoder_batch_size: int | None,
        device,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "HFDinoV3 requires transformers>=4.56.0. "
                "Install project dependencies before using model.arch=hf_dinov3."
            ) from exc

        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(device)
        self.feature_source = feature_source
        self.normalize = normalize
        self.encoder_batch_size = encoder_batch_size
        self.image_size = (
            tuple(image_size) if not isinstance(image_size, int) else image_size
        )
        if in_channels == 3:
            self.input_adapter = nn.Identity()
        else:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
            if in_channels % 3 == 0:
                groups = in_channels // 3
                with torch.no_grad():
                    self.input_adapter.weight.zero_()
                    self.input_adapter.bias.zero_()
                    for group in range(groups):
                        for channel in range(3):
                            self.input_adapter.weight[channel, group * 3 + channel, 0, 0] = (
                                1.0 / groups
                            )
        self.input_adapter = self.input_adapter.to(device)
        if freeze_backbone:
            self.model.requires_grad_(False)
            self.input_adapter.requires_grad_(False)
            self.model.eval()

        self.feature_dim = self._feature_dim()
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _feature_dim(self) -> int:
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Could not infer hidden_size from Hugging Face config {self.model.config}"
            )
        return int(hidden_size)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.to(torch.float32).div_(255)
        else:
            x = x.to(torch.float32)
        x = self.input_adapter(x)
        if self.image_size is not None:
            size = (
                (self.image_size, self.image_size)
                if isinstance(self.image_size, int)
                else tuple(self.image_size)
            )
            x = F.interpolate(
                x,
                size=size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.normalize:
            image_mean = self.image_mean.to(device=x.device, dtype=x.dtype)
            image_std = self.image_std.to(device=x.device, dtype=x.dtype)
            x = (x - image_mean) / image_std
        return x

    def _select_features(self, outputs) -> torch.Tensor:
        pooler_output = getattr(outputs, "pooler_output", None)
        if self.feature_source == "pooler_output" and pooler_output is not None:
            return pooler_output
        if self.feature_source in ("cls", "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]
        if pooler_output is not None:
            return pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        batch_size = self.encoder_batch_size
        if batch_size is not None and batch_size > 0 and x.shape[0] > batch_size:
            features = torch.cat(
                [
                    self._select_features(
                        self.model(pixel_values=self._preprocess(x[i : i + batch_size]))
                    )
                    for i in range(0, x.shape[0], batch_size)
                ],
                dim=0,
            )
        else:
            outputs = self.model(pixel_values=self._preprocess(x))
            features = self._select_features(outputs)
        return features.reshape(*leading_shape, features.shape[-1])


class HFDinoV3(Model):
    """BenchMARL image model backed by a Hugging Face DINOv3 encoder."""

    def __init__(self, **kwargs):
        self.model_name = kwargs.pop("model_name")
        self.image_size = kwargs.pop("image_size")
        self.freeze_backbone = kwargs.pop("freeze_backbone")
        self.normalize = kwargs.pop("normalize")
        self.feature_source = kwargs.pop("feature_source")
        self.local_files_only = kwargs.pop("local_files_only")
        self.encoder_batch_size = kwargs.pop("encoder_batch_size")
        self.mlp_num_cells = tuple(kwargs.pop("mlp_num_cells"))
        self.mlp_layer_class = kwargs.pop("mlp_layer_class")
        self.mlp_activation_class = kwargs.pop("mlp_activation_class")
        self.mlp_activation_kwargs = kwargs.pop("mlp_activation_kwargs")

        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        image_channels = self.input_features_images
        if self.input_has_agent_dim and self.centralised:
            image_channels *= self.n_agents

        n_encoders = (
            self.n_agents if self.output_has_agent_dim and not self.share_params else 1
        )
        self.encoders = nn.ModuleList(
            [
                _HFDinoV3Encoder(
                    in_channels=image_channels,
                    model_name=self.model_name,
                    image_size=self.image_size,
                    freeze_backbone=self.freeze_backbone,
                    normalize=self.normalize,
                    feature_source=self.feature_source,
                    local_files_only=self.local_files_only,
                    encoder_batch_size=self.encoder_batch_size,
                    device=self.device,
                )
                for _ in range(n_encoders)
            ]
        )

        head_input_features = self.encoders[0].feature_dim + self._head_tensor_features()
        self.heads = nn.ModuleList(
            [
                _make_mlp(
                    in_features=head_input_features,
                    out_features=self.output_leaf_spec.shape[-1],
                    hidden_cells=self.mlp_num_cells,
                    layer_class=self.mlp_layer_class,
                    activation_class=self.mlp_activation_class,
                    activation_kwargs=self.mlp_activation_kwargs,
                    device=self.device,
                )
                for _ in range(n_encoders if self.output_has_agent_dim else 1)
            ]
        )

    def _perform_checks(self):
        super()._perform_checks()
        self.image_in_keys = []
        self.tensor_in_keys = []
        image_shape = None
        tensor_shape = None

        for input_key, input_spec in self.input_spec.items(True, True):
            if self._is_image_spec(input_spec):
                self.image_in_keys.append(input_key)
                if image_shape is None:
                    image_shape = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != image_shape:
                    raise ValueError(
                        f"HFDinoV3 image inputs must share shape except channels, got {self.input_spec}"
                    )
            elif self._is_tensor_spec(input_spec):
                self.tensor_in_keys.append(input_key)
                if tensor_shape is None:
                    tensor_shape = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != tensor_shape:
                    raise ValueError(
                        f"HFDinoV3 tensor inputs must share shape except features, got {self.input_spec}"
                    )
            else:
                raise ValueError(
                    f"HFDinoV3 input value {input_key} from {self.input_spec} has an invalid shape"
                )

        if not self.image_in_keys:
            raise ValueError("HFDinoV3 found no image inputs")
        if self.input_has_agent_dim and image_shape[-3] != self.n_agents:
            raise ValueError(
                "If the HFDinoV3 input has the agent dimension, image specs "
                "must have the number of agents as their fourth-to-last dimension"
            )
        if (
            self.input_has_agent_dim
            and tensor_shape is not None
            and tensor_shape[-1] != self.n_agents
        ):
            raise ValueError(
                "If the HFDinoV3 input has the agent dimension, tensor specs "
                "must have the number of agents as their second-to-last dimension"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the HFDinoV3 output has the agent dimension, output specs "
                "must have the number of agents as their second-to-last dimension"
            )

        self.input_features_images = sum(
            self.input_spec[key].shape[-1] for key in self.image_in_keys
        )
        self.input_features_tensors = sum(
            self.input_spec[key].shape[-1] for key in self.tensor_in_keys
        )

    def _is_image_spec(self, input_spec: TensorSpec) -> bool:
        return (self.input_has_agent_dim and len(input_spec.shape) == 4) or (
            not self.input_has_agent_dim and len(input_spec.shape) == 3
        )

    def _is_tensor_spec(self, input_spec: TensorSpec) -> bool:
        return (self.input_has_agent_dim and len(input_spec.shape) == 2) or (
            not self.input_has_agent_dim and len(input_spec.shape) == 1
        )

    def _head_tensor_features(self) -> int:
        if not self.input_features_tensors:
            return 0
        if self.input_has_agent_dim and self.centralised:
            return self.input_features_tensors * self.n_agents
        return self.input_features_tensors

    def _gather_images(self, tensordict: TensorDictBase) -> torch.Tensor:
        if len(self.image_in_keys) == 1:
            x = tensordict.get(self.image_in_keys[0])
        else:
            x = torch.cat(
                [tensordict.get(in_key) for in_key in self.image_in_keys], dim=-1
            )
        x = x.transpose(-3, -1).transpose(-2, -1)
        if self.input_has_agent_dim and self.centralised:
            x = x.reshape(*x.shape[:-4], self.n_agents * x.shape[-3], *x.shape[-2:])
        return x

    def _gather_tensors(self, tensordict: TensorDictBase) -> torch.Tensor | None:
        if not self.tensor_in_keys:
            return None
        tensors = torch.cat(
            [tensordict.get(in_key) for in_key in self.tensor_in_keys], dim=-1
        )
        if self.input_has_agent_dim and self.centralised:
            tensors = tensors.reshape(*tensors.shape[:-2], -1)
            if self.output_has_agent_dim:
                tensors = tensors.unsqueeze(-2).expand(
                    *tensors.shape[:-1], self.n_agents, tensors.shape[-1]
                )
        elif not self.input_has_agent_dim and self.output_has_agent_dim:
            tensors = tensors.unsqueeze(-2).expand(
                *tensors.shape[:-1], self.n_agents, tensors.shape[-1]
            )
        return tensors

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.input_has_agent_dim and not self.centralised:
            if self.share_params:
                return self.encoders[0](images)
            return torch.stack(
                [
                    encoder(images[..., i, :, :, :])
                    for i, encoder in enumerate(self.encoders)
                ],
                dim=-2,
            )

        if self.output_has_agent_dim:
            return torch.stack([encoder(images) for encoder in self.encoders], dim=-2)

        return self.encoders[0](images)

    def _apply_heads(self, features: torch.Tensor) -> torch.Tensor:
        if self.output_has_agent_dim:
            if self.share_params:
                return _apply_shared_module(self.heads[0], features)
            return torch.stack(
                [head(features[..., i, :]) for i, head in enumerate(self.heads)],
                dim=-2,
            )
        return self.heads[0](features)

    def _forward_once(self, tensordict: TensorDictBase) -> TensorDictBase:
        features = self._encode(self._gather_images(tensordict))
        tensor_features = self._gather_tensors(tensordict)
        if tensor_features is not None:
            features = torch.cat([features, tensor_features], dim=-1)
        tensordict.set(self.out_key, self._apply_heads(features))
        return tensordict

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        batch_size = self.encoder_batch_size
        if batch_size is None or batch_size <= 0:
            return self._forward_once(tensordict)

        sample = tensordict.get(self.image_in_keys[0])
        image_rank = 4 if self.input_has_agent_dim else 3
        leading_shape = sample.shape[:-image_rank]
        leading_count = prod(leading_shape)
        if leading_count <= batch_size:
            return self._forward_once(tensordict)

        flat_tensordict = tensordict.reshape(leading_count)
        outputs = []
        for start in range(0, leading_count, batch_size):
            chunk = flat_tensordict[start : start + batch_size]
            outputs.append(self._forward_once(chunk).get(self.out_key))

        output = torch.cat(outputs, dim=0)
        tensordict.set(self.out_key, output.reshape(*leading_shape, *output.shape[1:]))
        return tensordict


@dataclass
class HFDinoV3Config(ModelConfig):
    """Config for a Hugging Face DINOv3 image backbone."""

    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    image_size: int | Sequence[int] = 224
    freeze_backbone: bool = True
    normalize: bool = True
    feature_source: str = "pooler_output"
    local_files_only: bool = False
    encoder_batch_size: int | None = 64

    mlp_num_cells: Sequence[int] = (512,)
    mlp_layer_class: Type[nn.Module] = nn.Linear
    mlp_activation_class: Type[nn.Module] = nn.ReLU
    mlp_activation_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return HFDinoV3
