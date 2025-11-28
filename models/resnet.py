import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResNetBlock(nn.Module):
    """
    ResNet block as described in the paper:
    ResNetBlock(x) = x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))

    Args:
        d: Input and output dimension (must be the same for residual connection)
        d_hidden: Hidden dimension (typically d * d_hidden_factor)
        hidden_dropout: Dropout rate after activation
        residual_dropout: Dropout rate before residual connection
    """

    def __init__(
        self,
        d: int,
        d_hidden: int,
        hidden_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(d)
        self.linear1 = nn.Linear(d, d_hidden)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.linear2 = nn.Linear(d_hidden, d)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: Tensor) -> Tensor:
        # ResNetBlock(x) = x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))
        z = self.batch_norm(x)
        z = self.linear1(z)
        z = F.relu(z)
        z = self.hidden_dropout(z)
        z = self.linear2(z)
        z = self.residual_dropout(z)

        # Residual connection
        return x + z


class Prediction(nn.Module):
    """
    Prediction head as described in the paper:
    Prediction(x) = Linear(ReLU(BatchNorm(x)))

    Args:
        d_in: Input dimension
        d_out: Output dimension (number of classes or 1 for regression)
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(d_in)
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: Tensor) -> Tensor:
        # Prediction(x) = Linear(ReLU(BatchNorm(x)))
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.linear(x)
        return x


class TabularResNet(nn.Module):
    """
    ResNet for tabular data as described in the paper:
    ResNet(x) = Prediction(ResNetBlock(...(ResNetBlock(Linear(x)))))

    Architecture components:
    1. First linear layer to project to base dimension
    2. Stack of ResNet blocks
    3. Prediction head with BatchNorm + ReLU + Linear

    Args:
        d_in: Number of input features (after preprocessing)
        d: Base dimension for ResNet blocks
        d_hidden_factor: Factor to compute hidden dimension (d_hidden = d * d_hidden_factor)
        n_layers: Number of ResNet blocks
        hidden_dropout: Dropout rate after ReLU in blocks
        residual_dropout: Dropout rate before residual connection in blocks
        d_out: Number of output units (classes for classification, 1 for regression)
    """

    def __init__(
        self,
        d_in: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        # First linear layer: Linear(x)
        self.first_layer = nn.Linear(d_in, d)

        # Stack of ResNet blocks: ResNetBlock(...(ResNetBlock(...)))
        d_hidden = int(d * d_hidden_factor)
        self.blocks = nn.ModuleList(
            [
                ResNetBlock(
                    d=d,
                    d_hidden=d_hidden,
                    hidden_dropout=hidden_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Prediction head: Prediction(...)
        self.prediction = Prediction(d_in=d, d_out=d_out)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass implementing:
        ResNet(x) = Prediction(ResNetBlock(...(ResNetBlock(Linear(x)))))

        Args:
            x: Input tensor of shape (batch_size, d_in)

        Returns:
            Output tensor of shape (batch_size, d_out) or (batch_size,) if d_out=1
        """
        # Linear(x)
        x = self.first_layer(x)

        # ResNetBlock(...(ResNetBlock(...)))
        for block in self.blocks:
            x = block(x)

        # Prediction(...)
        x = self.prediction(x)

        return x
