from torch import Tensor


def normalize_0_1(input: Tensor) -> Tensor:
    """Normalizes a given tensor to a range of [0, 1].

    Args:
        input (Tensor): Input tensor of any shape.

    Returns:
        output (Tensor): Normalized output tensor of the same shape as the input.
    """
    # Perform normalization
    output: Tensor = (input - input.min()) / (input.max() - input.min())
    return output
