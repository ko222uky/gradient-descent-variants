import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing import Iterable

#TODO: implement plain visualization for just the contour. Keep it as separate func.
def func_contour():
    pass

# primary func for generating the contours + historically travelled path.
def optim_contour(
    X: NDArray,
    Y: NDArray,
    Z: NDArray,
    start_point: NDArray,
    history: NDArray | Iterable[NDArray],
    known_optimum: NDArray = None,
    end_point: bool = True,
    figsize: tuple[int] = (10,7),
    num_contour_levels: int = 25,
    contourf_cmap: str = 'coolwarm',
    contourf_alpha: float = 0.7,
    contourf_label: str = 'Objective Value (Z)',

    contour_colors: str = 'black',
    contour_linewidths: float = 0.5,

    path_color: str = 'darkred',
    path_marker: str = 'o',
    path_markersize: int = 4,
    path_linestyle: str = '-',
    path_linewidth: float = 1.5,
    path_label = 'Optimization Path',

    start_color: str = 'red',
    start_marker: str = 'X',
    start_size: int = 150,
    start_zorder: int = 5,
    start_label: str = 'Start',

    end_color: str = 'black',
    end_marker: str = '*',
    end_size: int = 150,
    end_zorder: int = 5,
    end_label: str = 'End',

    optim_color: str = 'gold',
    optim_edgecolor: str = 'black',
    optim_marker: str = '*',
    optim_size: int = 200,
    optim_zorder: int = 5,  
    label_optim: str = 'Optimum',

    legend: bool = True

    ) -> Figure:
    """
    Creates a contour plot from a discrete grid of the search space
    and overlays the historical path taken by the optimizer.

    Requires an X, Y, Z 1D-array input where X,Y denote the coordinates in the search space,
    and Z is the objective function evaluated at (X, Y).

    Also requires a `history`, which is a 1D array of the updated :math:`x_i` values
    that were obtained during the optimizer algorithm.

    """
    # instantiate a new figure for the contour plot
    fig, ax = plt.subplots(figsize=figsize)

    # creates the color fill for contour levels
    contour_filled = ax.contourf(
        X, Y, Z, 
        cmap=contourf_cmap,  
        alpha=contourf_alpha,
        levels=num_contour_levels
    )

    # adds a colorbar to the contour plot with a label
    fig.colorbar(contour_filled, label=contourf_label)

    # draws countour level boundaries on top of the filled contours
    ax.contour(
        X, Y, Z,
        levels=num_contour_levels,
        colors=contour_colors,
        linewidths=contour_linewidths
    )


    if isinstance(history, Iterable):
        history = np.array(history)

    # plot the optimization path on top of the contour plot
    ax.plot(
        history[:, 0], # all rows of our X
        history[:, 1], # all rows of our Y
        color=path_color,
        marker=path_marker,
        markersize=path_markersize,
        linestyle=path_linestyle,
        linewidth=path_linewidth,
        label=path_label
    )



    # mark the starting point of the optimization path
    ax.scatter(
        start_point[0],
        start_point[1],
        color=start_color,
        marker=start_marker,
        s=start_size,
        zorder=start_zorder,
        label=start_label
    )

    if end_point:
        # mark the ending point
        if isinstance(history, Iterable):
            history = np.array(history)


        ax.scatter(
            history[-1, 0],  # last row of our X
            history[-1, 1],  # last row of our Y
            color=end_color,
            marker=end_marker,
            s=end_size,
            zorder=end_zorder,
            label=end_label
        )

    if legend:
        ax.legend()
    
    # if a known global optimum is provided, mark it, too.
    if known_optimum is not None:
        ax.scatter(
            known_optimum[0],
            known_optimum[1],
            color=optim_color,
            marker=optim_marker,
            s=optim_size,
            edgecolor=optim_edgecolor,
            zorder=optim_zorder,
            label=label_optim
        )

    return fig