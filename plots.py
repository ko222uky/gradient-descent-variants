"""
Simple script to get basic plots of the objective functions.
"""

from gdv.functions import (
    multimodal_nonconvex,
    rosenbrock,
    convex_bowl
)
from gdv.contour import func_contour
from gdv.functions import default_meshgrid


def main():
    funcs = {
        'multimodal_nonconvex': multimodal_nonconvex,
        'rosenbrock': rosenbrock,
        'convex_bowl': convex_bowl
    }

    for func_name, func in funcs.items():
        X, Y = default_meshgrid(x_range=(-15, 15), y_range=(-15, 15), num_points=5000)
        Z = func((X, Y))

        fig = func_contour(X, Y, Z, legend=False)
        fig.savefig(f"figures/{func_name}.png")

if __name__ == "__main__":
    main()