from dataclasses import dataclass
from typing import Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.data import TensorSpec
from torchvision.models.resnet import BasicBlock, ResNet

from benchmarl.models.common import Model, ModelConfig


def _make_mlp(
    in_features: int,
    out_features: int,
    hidden_cells: Sequence[int],
    layer_class: Type[nn.Module],
    activation_class: Type[nn.Module],
    activation_kwargs: Optional[dict],
    device,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_features = in_features
    for hidden_features in hidden_cells:
        layers.append(layer_class(last_features, hidden_features))
        layers.append(activation_class(**(activation_kwargs or {})))
        last_features = hidden_features
    layers.append(layer_class(last_features, out_features))
    return nn.Sequential(*layers).to(device)


def _apply_shared_module(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    leading_shape = x.shape[:-1]
    y = module(x.reshape(-1, x.shape[-1]))
    return y.reshape(*leading_shape, y.shape[-1])


class _SmallResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        layers: Sequence[int],
        stem_kernel_size: int,
        stem_stride: int,
        stem_padding: int,
        use_maxpool: bool,
    ):
        super().__init__()
        if len(layers) != 4:
            raise ValueError(f"SmallResNet layers must have length 4, got {layers!r}")
        self.net = ResNet(BasicBlock, list(layers), num_classes=feature_dim)
        stem_out_channels = self.net.conv1.out_channels
        self.net.conv1 = nn.Conv2d(
            in_channels,
            stem_out_channels,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_padding,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.net.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        if not use_maxpool:
            self.net.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        y = self.net(x)
        return y.reshape(*leading_shape, y.shape[-1])


class SmallResNet(Model):
    """BenchMARL model backed by a small torchvision ResNet."""

    def __init__(self, **kwargs):
        self.feature_dim = kwargs.pop("feature_dim")
        self.resnet_layers = tuple(kwargs.pop("resnet_layers"))
        self.stem_kernel_size = kwargs.pop("stem_kernel_size")
        self.stem_stride = kwargs.pop("stem_stride")
        self.stem_padding = kwargs.pop("stem_padding")
        self.use_maxpool = kwargs.pop("use_maxpool")
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
                _SmallResNetEncoder(
                    in_channels=image_channels,
                    feature_dim=self.feature_dim,
                    layers=self.resnet_layers,
                    stem_kernel_size=self.stem_kernel_size,
                    stem_stride=self.stem_stride,
                    stem_padding=self.stem_padding,
                    use_maxpool=self.use_maxpool,
                ).to(self.device)
                for _ in range(n_encoders)
            ]
        )

        head_input_features = self.feature_dim + self._head_tensor_features()
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
                        f"SmallResNet image inputs must share shape except channels, got {self.input_spec}"
                    )
            elif self._is_tensor_spec(input_spec):
                self.tensor_in_keys.append(input_key)
                if tensor_shape is None:
                    tensor_shape = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != tensor_shape:
                    raise ValueError(
                        f"SmallResNet tensor inputs must share shape except features, got {self.input_spec}"
                    )
            else:
                raise ValueError(
                    f"SmallResNet input value {input_key} from {self.input_spec} has an invalid shape"
                )

        if not self.image_in_keys:
            raise ValueError("SmallResNet found no image inputs")
        if self.input_has_agent_dim and image_shape[-3] != self.n_agents:
            raise ValueError(
                "If the SmallResNet input has the agent dimension, image specs "
                "must have the number of agents as their fourth-to-last dimension"
            )
        if (
            self.input_has_agent_dim
            and tensor_shape is not None
            and tensor_shape[-1] != self.n_agents
        ):
            raise ValueError(
                "If the SmallResNet input has the agent dimension, tensor specs "
                "must have the number of agents as their second-to-last dimension"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the SmallResNet output has the agent dimension, output specs "
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
        x = torch.cat(
            [tensordict.get(in_key) for in_key in self.image_in_keys], dim=-1
        )
        if x.dtype == torch.uint8:
            x = x.to(torch.float32).div_(255)
        else:
            x = x.to(torch.float32)
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

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        features = self._encode(self._gather_images(tensordict))
        tensor_features = self._gather_tensors(tensordict)
        if tensor_features is not None:
            features = torch.cat([features, tensor_features], dim=-1)
        tensordict.set(self.out_key, self._apply_heads(features))
        return tensordict


@dataclass
class SmallResNetConfig(ModelConfig):
    """Config for a compact torchvision ResNet backbone."""

    feature_dim: int = 256
    resnet_layers: Sequence[int] = (1, 1, 1, 1)
    stem_kernel_size: int = 3
    stem_stride: int = 1
    stem_padding: int = 1
    use_maxpool: bool = False

    mlp_num_cells: Sequence[int] = (512,)
    mlp_layer_class: Type[nn.Module] = nn.Linear
    mlp_activation_class: Type[nn.Module] = nn.ReLU
    mlp_activation_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return SmallResNet
