from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import ValueEstimators

from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.mappo import Mappo
from benchmarl.models.common import ModelConfig

from doom_marl.algorithms._contrastive import TensorHead, compute_contrastive_loss


CONTRASTIVE_EPISODE_KEY = "contrastive_episode_id"
CONTRASTIVE_STEP_KEY = "contrastive_step_id"


class ContrastiveClipPPOLoss(ClipPPOLoss):
    def __init__(
        self,
        *args,
        group: str,
        embedding_key: tuple[str, str],
        contrastive_coef: float,
        contrastive_temperature: float,
        contrastive_normalize: bool,
        contrastive_loss: str,
        contrastive_w_pos: int,
        contrastive_w_neg: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group = group
        self.embedding_key = embedding_key
        self.contrastive_coef = contrastive_coef
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_normalize = contrastive_normalize
        self.contrastive_loss = contrastive_loss.lower()
        self.contrastive_w_pos = int(contrastive_w_pos)
        self.contrastive_w_neg = int(contrastive_w_neg)
        if self.contrastive_loss not in {"nt_xent", "sigmoid"}:
            raise ValueError(
                "contrastive_loss must be either 'nt_xent' or 'sigmoid', "
                f"got {contrastive_loss!r}."
            )
        if self.contrastive_w_neg < self.contrastive_w_pos:
            raise ValueError(
                f"contrastive_w_neg ({self.contrastive_w_neg}) must be >= "
                f"contrastive_w_pos ({self.contrastive_w_pos})."
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_out = super().forward(tensordict)
        embeddings = self._compute_embeddings(tensordict)
        ep_id, step_id = _extract_contrastive_tags(tensordict, embeddings)
        loss_contrastive = compute_contrastive_loss(
            embeddings,
            ep_id,
            step_id,
            w_pos=self.contrastive_w_pos,
            w_neg=self.contrastive_w_neg,
            temperature=self.contrastive_temperature,
            normalize=self.contrastive_normalize,
            mode=self.contrastive_loss,
        )
        td_out.set("loss_contrastive", loss_contrastive)
        td_out.set(
            "loss_objective",
            td_out["loss_objective"] + self.contrastive_coef * loss_contrastive,
        )
        return td_out

    def _compute_embeddings(self, tensordict: TensorDictBase) -> torch.Tensor:
        td = tensordict.clone(False)
        self.actor_network(td)
        embeddings = td.get(self.embedding_key)
        if embeddings is None:
            raise KeyError(
                f"Expected actor to write embeddings at key {self.embedding_key}, but none were found."
            )
        return embeddings


def _extract_contrastive_tags(
    tensordict: TensorDictBase, embeddings: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    ep_id = tensordict.get(CONTRASTIVE_EPISODE_KEY, default=None)
    step_id = tensordict.get(CONTRASTIVE_STEP_KEY, default=None)
    if ep_id is None or step_id is None:
        raise KeyError(
            f"Contrastive loss expected '{CONTRASTIVE_EPISODE_KEY}' and "
            f"'{CONTRASTIVE_STEP_KEY}' on the tensordict — wire up the collector "
            "postproc in VizdoomExperiment._setup_collector."
        )
    leading_shape = embeddings.shape[:-2]
    return ep_id.reshape(leading_shape), step_id.reshape(leading_shape)


@dataclass
class ContrastiveMappoConfig:
    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING

    contrastive_coef: float = MISSING
    contrastive_embedding_dim: int = MISSING
    contrastive_temperature: float = MISSING
    contrastive_hidden_dim: int = MISSING
    contrastive_normalize: bool = True
    contrastive_loss: str = "sigmoid"
    contrastive_w_pos: int = 4
    contrastive_w_neg: int = 16

    def get_algorithm(self, experiment) -> Algorithm:
        return self.associated_class()(**self.__dict__, experiment=experiment)

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return ContrastiveMappo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True

    @staticmethod
    def has_independent_critic() -> bool:
        return False

    def has_critic(self) -> bool:
        return True


class ContrastiveMappo(Mappo):
    def __init__(
        self,
        contrastive_coef: float,
        contrastive_embedding_dim: int,
        contrastive_temperature: float,
        contrastive_hidden_dim: int,
        contrastive_normalize: bool = True,
        contrastive_loss: str = "sigmoid",
        contrastive_w_pos: int = 4,
        contrastive_w_neg: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contrastive_coef = contrastive_coef
        self.contrastive_embedding_dim = contrastive_embedding_dim
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_hidden_dim = contrastive_hidden_dim
        self.contrastive_normalize = contrastive_normalize
        self.contrastive_loss = contrastive_loss
        self.contrastive_w_pos = contrastive_w_pos
        self.contrastive_w_neg = contrastive_w_neg

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        loss_module = ContrastiveClipPPOLoss(
            actor_network=policy_for_loss,
            critic_network=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coef,
            critic_coeff=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            group=group,
            embedding_key=(group, "contrastive_embedding"),
            contrastive_coef=self.contrastive_coef,
            contrastive_temperature=self.contrastive_temperature,
            contrastive_normalize=self.contrastive_normalize,
            contrastive_loss=self.contrastive_loss,
            contrastive_w_pos=self.contrastive_w_pos,
            contrastive_w_neg=self.contrastive_w_neg,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if n_agents < 2:
            raise ValueError(
                f"Contrastive MAPPO requires at least 2 agents in group '{group}', got {n_agents}."
            )

        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        hidden_dim = self.contrastive_hidden_dim
        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )
        hidden_output_spec = Composite(
            {
                group: Composite(
                    {"contrastive_hidden": Unbounded(shape=(n_agents, hidden_dim))},
                    shape=(n_agents,),
                )
            }
        )
        actor_backbone = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=hidden_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )

        logits_head = TensorDictModule(
            TensorHead(
                in_features=hidden_dim,
                out_features=logits_shape[-1],
                n_agents=n_agents,
                share_params=self.experiment_config.share_policy_params,
                device=self.device,
            ),
            in_keys=[(group, "contrastive_hidden")],
            out_keys=[(group, "logits")],
        )
        embedding_head = TensorDictModule(
            TensorHead(
                in_features=hidden_dim,
                out_features=self.contrastive_embedding_dim,
                n_agents=n_agents,
                share_params=self.experiment_config.share_policy_params,
                device=self.device,
            ),
            in_keys=[(group, "contrastive_hidden")],
            out_keys=[(group, "contrastive_embedding")],
        )
        actor_module = TensorDictSequential(actor_backbone, logits_head, embedding_head)

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )
        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )

        return policy

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        loss_vals = super().process_loss_vals(group, loss_vals)
        if "loss_contrastive" in loss_vals.keys():
            return loss_vals
        loss_vals.set("loss_contrastive", torch.zeros((), device=self.device))
        return loss_vals
