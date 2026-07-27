from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StructureConfig:
    use_query_size_film: bool
    branch_projection: str

    def __post_init__(self) -> None:
        if self.branch_projection not in {
            "dense",
            "small",
            "small2_relu",
            "small2_gelu",
            "linear2x",
            "linear3x",
            "off",
            "off2_relu",
            "off2_gelu",
            "group2x_relu",
            "group2x_relu_raw",
            "group2x_gelu",
            "group2x_gelu_raw",
            "group5x_relu",
            "group5x_gelu_raw",
            "group5x_residual_gelu_raw",
            "group5x_channel_gelu_raw",
            "residual2x_relu",
            "residual2x_gelu",
            "residual3x_relu",
            "residual3x_gelu",
            "monotonic_static",
            "monotonic_context",
        }:
            raise ValueError(
                "unknown branch projection/encoder mode: "
                f"{self.branch_projection}"
            )

    @property
    def name(self) -> str:
        film = "FiLM" if self.use_query_size_film else "NoFiLM"
        projection = {
            "dense": "DenseProjection",
            "small": "SmallProjection",
            "small2_relu": "SmallProjectionFusion2ReLU",
            "small2_gelu": "SmallProjectionFusion2GELU",
            "linear2x": "Linear2xProjection",
            "linear3x": "Linear3xProjection",
            "off": "NoProjection",
            "off2_relu": "NoProjectionFusion2ReLU",
            "off2_gelu": "NoProjectionFusion2GELU",
            "group2x_relu": "Group2xReLU",
            "group2x_relu_raw": "Group2xReLURaw",
            "group2x_gelu": "Group2xGELU",
            "group2x_gelu_raw": "Group2xGELURaw",
            "group5x_relu": "Group5xReLU",
            "group5x_gelu_raw": "Group5xGELURaw",
            "group5x_residual_gelu_raw": "Group5xResidualGELURaw",
            "group5x_channel_gelu_raw": "Group5xChannelGELURaw",
            "residual2x_relu": "Residual2xReLU",
            "residual2x_gelu": "Residual2xGELU",
            "residual3x_relu": "Residual3xReLU",
            "residual3x_gelu": "Residual3xGELU",
            "monotonic_static": "MonotonicStaticGap",
            "monotonic_context": "MonotonicContextGap",
        }[self.branch_projection]
        return f"{projection}-{film}"


class ConservativeMonotonicMLP(nn.Module):
    GAP_INDICES = (3, 4, 6, 7, 8)
    CONTEXT_INDICES = (0, 1, 2, 5, 9, 10)

    def __init__(
        self,
        contextual_slopes: bool,
        hidden_dim: int = 16,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.contextual_slopes = contextual_slopes
        self.context_encoder = nn.Sequential(
            nn.Linear(len(self.CONTEXT_INDICES), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Linear(hidden_dim, 1)
        if contextual_slopes:
            self.slope_head: nn.Module = nn.Linear(
                hidden_dim,
                len(self.GAP_INDICES),
            )
            nn.init.zeros_(self.slope_head.weight)
            nn.init.constant_(self.slope_head.bias, -2.0)
        else:
            self.raw_gap_slopes = nn.Parameter(
                torch.full((len(self.GAP_INDICES),), -2.0)
            )

    def forward(
        self,
        features: torch.Tensor,
        query_size_condition: torch.Tensor,
    ) -> torch.Tensor:
        del query_size_condition
        context = features[:, list(self.CONTEXT_INDICES)]
        gaps = features[:, list(self.GAP_INDICES)]
        hidden = self.context_encoder(context)
        base_logit = self.base_head(hidden).squeeze(-1)
        if self.contextual_slopes:
            slopes = F.softplus(self.slope_head(hidden))
        else:
            slopes = F.softplus(self.raw_gap_slopes).expand(
                len(features),
                -1,
            )
        monotonic_gap_logit = (
            slopes * gaps
        ).sum(dim=1) / math.sqrt(len(self.GAP_INDICES))
        return torch.sigmoid(base_logit + monotonic_gap_logit)


class ConfigurableClean3224MLP(nn.Module):
    def __init__(
        self,
        group_indices: list[list[int]],
        config: StructureConfig,
        condition_hidden: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.config = config
        self.group_indices = [tuple(group) for group in group_indices]
        self.branch_include_raw = True
        self.branch_residual = False
        fusion_depth = 1
        output_activation: type[nn.Module] = nn.GELU

        if config.branch_projection == "dense":
            learned_widths = [
                5 * len(group) for group in self.group_indices
            ]
            projection_factory = lambda size, width: nn.Linear(size, width)
        elif config.branch_projection in {
            "small",
            "small2_relu",
            "small2_gelu",
        }:
            learned_widths = [
                len(group) for group in self.group_indices
            ]
            projection_factory = lambda size, width: nn.Linear(size, width)
            if config.branch_projection.startswith("small2_"):
                fusion_depth = 2
                output_activation = (
                    nn.ReLU
                    if config.branch_projection.endswith("_relu")
                    else nn.GELU
                )
        elif config.branch_projection in {"linear2x", "linear3x"}:
            multiplier = (
                2 if config.branch_projection == "linear2x" else 3
            )
            learned_widths = [
                multiplier * len(group)
                for group in self.group_indices
            ]
            projection_factory = lambda size, width: nn.Linear(size, width)
        elif (
            config.branch_projection.startswith("group2x_")
            or config.branch_projection in {
                "group5x_relu",
                "group5x_gelu_raw",
            }
        ):
            multiplier = (
                5
                if config.branch_projection.startswith("group5x_")
                else 2
            )
            learned_widths = [
                multiplier * len(group)
                for group in self.group_indices
            ]
            use_relu = "relu" in config.branch_projection
            activation: type[nn.Module] = nn.ReLU if use_relu else nn.GELU
            output_activation = activation
            self.branch_include_raw = (
                config.branch_projection.endswith("_raw")
            )

            def projection_factory(
                size: int,
                width: int,
            ) -> nn.Module:
                layers: list[nn.Module] = [
                    nn.Linear(size, width),
                    activation(),
                ]
                if config.branch_projection != "group5x_gelu_raw":
                    layers.append(nn.Dropout(dropout))
                return nn.Sequential(*layers)
        elif config.branch_projection == "group5x_residual_gelu_raw":
            learned_widths = [
                5 * len(group) for group in self.group_indices
            ]

            def projection_factory(
                size: int,
                width: int,
            ) -> nn.Module:
                return GatedResidualGELU(size, width)
        elif config.branch_projection == "group5x_channel_gelu_raw":
            learned_widths = [
                5 * len(group) for group in self.group_indices
            ]

            def projection_factory(
                size: int,
                width: int,
            ) -> nn.Module:
                return GatedResidualGELU(
                    size,
                    width,
                    channelwise=True,
                )
        elif config.branch_projection.startswith("residual"):
            multiplier = (
                2 if config.branch_projection.startswith("residual2x") else 3
            )
            use_relu = "relu" in config.branch_projection
            activation = nn.ReLU if use_relu else nn.GELU
            output_activation = activation
            learned_widths = [
                len(group) for group in self.group_indices
            ]
            self.branch_include_raw = False
            self.branch_residual = True

            def projection_factory(
                size: int,
                width: int,
            ) -> nn.Module:
                hidden_width = multiplier * size
                return nn.Sequential(
                    nn.Linear(size, hidden_width),
                    activation(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_width, size),
                )
        elif config.branch_projection in {
            "off",
            "off2_relu",
            "off2_gelu",
        }:
            learned_widths = [0] * len(self.group_indices)
            projection_factory = lambda size, width: nn.Linear(size, width)
            if config.branch_projection.startswith("off2_"):
                fusion_depth = 2
                output_activation = (
                    nn.ReLU
                    if config.branch_projection.endswith("_relu")
                    else nn.GELU
                )
        else:
            raise RuntimeError(
                "branch mode was validated but not constructed: "
                f"{config.branch_projection}"
            )

        self.learned_widths = tuple(learned_widths)
        self.projections = nn.ModuleList(
            [
                projection_factory(len(group), width)
                for group, width in zip(
                    self.group_indices,
                    learned_widths,
                )
                if width > 0
            ]
        )
        joined_dim = sum(
            (
                len(group) + width
                if self.branch_include_raw
                else width
            )
            for group, width in zip(
                self.group_indices,
                learned_widths,
            )
        )

        if config.use_query_size_film:
            self.condition_film: nn.Module | None = nn.Sequential(
                nn.Linear(1, condition_hidden),
                nn.GELU(),
                nn.Linear(condition_hidden, 2 * joined_dim),
            )
            final_layer = self.condition_film[-1]
            if not isinstance(final_layer, nn.Linear):
                raise RuntimeError("unexpected FiLM final layer")
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        else:
            self.condition_film = None

        output_layers: list[nn.Module] = [
            nn.Linear(joined_dim, hidden_dim),
            output_activation(),
            nn.Dropout(dropout),
        ]
        if fusion_depth == 2:
            output_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    output_activation(),
                    nn.Dropout(dropout),
                ]
            )
        output_layers.append(nn.Linear(hidden_dim, 1))
        self.output = nn.Sequential(*output_layers)

    def forward_logit(
        self,
        features: torch.Tensor,
        query_size_condition: torch.Tensor,
    ) -> torch.Tensor:
        branches = []
        projection_index = 0
        for group, learned_width in zip(
            self.group_indices,
            self.learned_widths,
        ):
            raw = features[:, list(group)]
            if learned_width > 0:
                learned = self.projections[projection_index](raw)
                projection_index += 1
                if self.branch_residual:
                    branch = raw + learned
                elif self.branch_include_raw:
                    branch = torch.cat([raw, learned], dim=1)
                else:
                    branch = learned
            else:
                branch = raw
            branches.append(branch)
        joined = torch.cat(branches, dim=1)

        if self.condition_film is not None:
            gamma_raw, beta_raw = self.condition_film(
                query_size_condition
            ).chunk(2, dim=1)
            joined = (
                joined * (1.0 + torch.tanh(gamma_raw))
                + torch.tanh(beta_raw)
            )

        return self.output(joined).squeeze(-1)

    def forward(
        self,
        features: torch.Tensor,
        query_size_condition: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward_logit(features, query_size_condition)
        )


class GatedResidualGELU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        channelwise: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        scale_shape = (output_dim,) if channelwise else ()
        self.nonlinearity_scale = nn.Parameter(torch.zeros(scale_shape))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.linear(features)
        scale = torch.tanh(self.nonlinearity_scale)
        return projected + scale * F.gelu(projected)


def build_configurable_model(
    group_indices: list[list[int]],
    use_query_size_film: bool,
    branch_projection: str,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    if branch_projection in {
        "monotonic_static",
        "monotonic_context",
    }:
        if use_query_size_film:
            raise ValueError(
                "monotonic pair-level models do not use query-size FiLM"
            )
        return ConservativeMonotonicMLP(
            contextual_slopes=(
                branch_projection == "monotonic_context"
            )
        )
    return ConfigurableClean3224MLP(
        group_indices,
        StructureConfig(
            use_query_size_film=use_query_size_film,
            branch_projection=branch_projection,
        ),
    )


CandidateTreeEstimator = ConfigurableClean3224MLP


def build_estimator(seed: int = 42) -> nn.Module:
    return build_configurable_model(
        [[0, 1, 2], [3, 4], [5, 6], [7, 8, 9, 10]],
        use_query_size_film=True,
        branch_projection="group5x_residual_gelu_raw",
        seed=seed,
    )
