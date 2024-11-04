import os

import imageio
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import colors


def letter_to_mask(letter, matrix_shape) -> np.ndarray:
    """
    Generates a binary mask in the shape of a letter to apply to a weight matrix.

    Args:
        letter (str): The letter to be converted into a mask.
        matrix_shape (int): The shape of the weight matrix (e.g., 28 for a 28x28 matrix).

    Returns:
        numpy.ndarray: A binary mask array (0s and 1s) with the letter shape.
    """
    # Create a blank image
    img = Image.new(mode="L", size=(matrix_shape, matrix_shape), color=0)
    draw = ImageDraw.Draw(img)

    text_width, text_height = 0, 0
    font_size = matrix_shape

    while (text_width < matrix_shape) and (text_height < matrix_shape):
        # Calculate text width and height
        bbox = draw.textbbox(xy=(0, 0), text=letter, font_size=font_size)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Increase font size
        font_size += 1

    # Calculate position to center the letter
    x = matrix_shape - bbox[2]
    y = matrix_shape - bbox[3]
    if text_width < matrix_shape:
        x //= 2
    if text_height < matrix_shape:
        y //= 2

    # Draw the letter on the image
    draw.text(xy=(x, y), text=letter, fill=255, font_size=font_size)

    # Convert to a NumPy array and create a binary mask
    mask = np.array(img)
    mask = (mask > 0).astype(int)

    return mask


def visualize_weights(weights: np.ndarray,
                      title: str = "Weights Visualization",
                      vmin: float = None,
                      vmax: float = None,
                      fig: plt.Figure = None,
                      ax: plt.Axes = None) -> plt.Axes:
    """
    Visualizes the weights of a neural network layer using a heatmap.

    Args:
        weights (numpy.ndarray): The weights to be visualized.
        title (str): The title of the plot.
        vmin (float): The minimum value of the color scale.
        vmax (float): The maximum value of the color scale.
        fig (matplotlib.figure.Figure): The plot figure.
        ax (matplotlib.figure.Axes): The plot axes.

    Returns:
        matplotlib.figure.Axes: The plot axes.
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    if vmin is None:
        vmin = np.min(weights)
    if vmax is None:
        vmax = np.max(weights)
    vabs = max(abs(vmin), abs(vmax))

    cax = ax.matshow(weights, cmap='RdBu', vmin=-vabs, vmax=vabs)
    fig.colorbar(cax, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Input features')
    ax.set_ylabel('Output features')

    return ax


def visualize_weights_distribution(weights_counts: np.ndarray,
                                   x_bins: np.ndarray,
                                   y_bins: np.ndarray,
                                   max_count: int,
                                   title: str = "Weights Distribution",
                                   fig: plt.Figure = None,
                                   ax: plt.Axes = None) -> plt.Axes:
    """
     Visualizes the distribution of weights in a neural network layer.

     Args:
          weights_counts (numpy.ndarray): The counts of weights in each bin.
          x_bins (numpy.ndarray): The bins for the x-axis.
          y_bins (numpy.ndarray): The bins for the y-axis.
          max_count (int): The maximum count value for the color scale.
          title (str): The title of the plot.
          fig (matplotlib.figure.Figure): The plot figure.
          ax (matplotlib.figure.Axes): The plot axes.

     Returns:
          matplotlib.figure.Axes: The plot axes.
     """
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    cax = ax.matshow(weights_counts,
                     cmap='BuPu',
                     norm=colors.LogNorm(
                         vmax=max_count,
                         clip=True),
                     aspect='auto')

    fig.colorbar(cax, ax=ax)

    sums = np.sum(weights_counts, axis=0)
    cumsums = np.cumsum(weights_counts, axis=0)
    argmeds = [np.min(np.argwhere(cumsums[:, i] >= sums[i] / 2)) for i in range(len(sums))]
    argquarts = [np.min(np.argwhere(cumsums[:, i] >= sums[i] / 4)) for i in range(len(sums))]
    argquarts3 = [np.min(np.argwhere(cumsums[:, i] >= 3 * sums[i] / 4)) for i in range(len(sums))]

    ax.plot(np.arange(len(argmeds)),
            argmeds,
            color='red',
            linewidth=1,
            linestyle='--')

    ax.plot(np.arange(len(argquarts)),
            argquarts,
            color='green',
            linewidth=1,
            linestyle='--')

    ax.plot(np.arange(len(argquarts3)),
            argquarts3,
            color='green',
            linewidth=1,
            linestyle='--')

    ax.set_xticks(np.arange(len(x_bins) - 1, step=2))
    ax.set_xticklabels(x_bins[:-1][::2])
    ax.set_xlabel('Weights at epoch 0')
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

    ax.set_yticks(np.arange(len(y_bins) - 1, step=4))
    ax.set_yticklabels(y_bins[:-1][::-4])
    ax.set_ylabel(f'Weights at epoch i')

    ax.set_title(title)

    return ax


def draw_learning_curves(train_losses: list,
                         val_losses: list,
                         min_loss: float = None,
                         max_loss: float = None,
                         n_epochs: int = None,
                         fig: plt.Figure = None,
                         ax: plt.Axes = None,
                         title: str = "Learning Curves") -> plt.Axes:
    """
    Draw the learning curves of a model.

    Args:
        train_losses (list): The training losses.
        val_losses (list): The validation losses.
        min_loss (float): The minimum loss value.
        max_loss (float): The maximum loss value.
        n_epochs (int): The number of total epochs.
        fig (plt.Figure): The plot figure.
        ax (plt.Axes): The plot axes.
        title (str): The title of the plot.

    Returns:
        matplotlib.figure.Axes: The plot axes.
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    if min_loss is None:
        min_loss = min(min(train_losses), min(val_losses))
    if max_loss is None:
        max_loss = max(max(train_losses), max(val_losses))

    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Test Loss')
    ax.size = (10, 5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, n_epochs)
    ax.set_ylim(min_loss * 0.9, max_loss * 1.1)
    ax.legend()
    ax.grid()
    ax.set_title(title)

    return ax


def create_gif_from_folder(folder_path: str,
                           output_path: str,
                           duration: float = 10.0):
    """
    Create a GIF from a folder of images.

    Args:
        folder_path (str): The path to the folder containing the images.
        output_path (str): The path to save the GIF file.
        duration (float): The duration of each frame in seconds.
    """
    fig_paths = [os.path.join(folder_path, fig_path) for fig_path in os.listdir(folder_path)]

    # Sort the figure paths by their index
    fig_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    with imageio.get_writer(output_path, mode='I', duration=duration, loop=0) as writer:
        for fig_path in fig_paths:
            image = imageio.v2.imread(fig_path)
            writer.append_data(image)

    # Clean up the saved figures
    for fig_path in fig_paths:
        os.remove(fig_path)
