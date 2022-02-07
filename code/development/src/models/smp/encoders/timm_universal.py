import timm
import torch
import torch.nn as nn


class TimmUniversalEncoder(nn.Module):
    def __init__(
        self, name: str, pretrained: bool = True, in_channels: int = 3, depth: int = 5, output_stride: int = 32
    ) -> None:
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # not all models support output stride argument, drop it by default
        if output_stride == 32:
            kwargs.pop("output_stride")

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self) -> int:
        return self._out_channels
