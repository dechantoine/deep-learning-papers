from torch import nn, Tensor


class TwoLayerNet(nn.Module):
    """A two-layer neural network with ReLU activation functions and a softmax output layer."""

    def __init__(self, input_size: int = 28 * 28, hidden_size: int = 128, output_size: int = 10):
        """
        Initializes the two-layer neural network.

        Args:
            input_size (int): The size of the input layer.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer
        """
        super(TwoLayerNet, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the neural network.

        Args:
            x: The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x

    def mask_hidden_parameters(self, mask: Tensor):
        """
        Masks the hidden layer parameters with the given mask.

        Args:
            mask (Tensor): The mask to apply to the hidden layer parameters.
        """
        self.fc_hidden.weight.data = self.fc_hidden.weight.data * Tensor(mask)
