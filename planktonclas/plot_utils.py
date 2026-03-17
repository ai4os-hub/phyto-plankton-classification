""" "
Miscellaneous functions to plot.

Date: September 2018
Original Author: Ignacio Heredia (CSIC)
Maintainer: Wout Decrop (VLIZ)
Contact: wout.decrop@vliz.be
Github: woutdecrop / lifewatch
"""

import json
import os

import matplotlib.pylab as plt
import numpy as np
import seaborn

from planktonclas import paths


def create_pred_path(save_path, dir="", weighted=False, **kwargs):
    """
    Create the directory path for saving the plots based on the provided options.

    Args:
        save_path (str): Path where the plots will be saved.
        paths (object): Object with timestamped directory creation method.
        aimed (bool): Flag indicating whether the confusion matrices are aimed or not.
        weighted (bool): Flag indicating whether to compute weighted confusion matrices.

    Returns:
        str: Directory path for saving the plots.
    """
    value = next(iter(kwargs.values()))
    if weighted:
        pred_path = save_path or os.path.join(paths.get_results_dir(), dir,
                                              "confusion_weighted")
    else:
        pred_path = save_path or os.path.join(paths.get_results_dir(), dir,
                                              value)

    os.makedirs(pred_path, exist_ok=True)
    return pred_path
