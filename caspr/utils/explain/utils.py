import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def separate_pos_neg(attribution):
    """Separate out positive and negative attributes in the dataframe with attributes.

    Outputs two separated dataframes
    """
    attribution_pos_val = attribution*(attribution >= 0)
    attribution_neg_val = attribution*~(attribution >= 0)
    return attribution_pos_val, attribution_neg_val


def visualize(explanations: pd.DataFrame, separate_pos_neg_imp: bool = False,
              title="Average Feature Importances", axis_title="Features", save_fig: str = None):
    """Visualize explanations.

    Utility function used to create bar graph visualisations at a model level

    Args:
        explanations (pandas dataframe): Dataframe with feature attributions
        separate_pos_neg_imp (Boolean: Default = False): Determines if the positive and negative attributions are to be
            aggregated and plotted separately (two reverse sided bars) in the same plot
        title (String : Default = "Average Feature Importances") : Represents the title of the graph
        axis_title (String: Default = "Features") : Represents the title of the Y axis
        save_fig (String) : Contains the path where to save the image plot. If None : the module doesnt save the image

    """
    feature_names = explanations.columns
    imp_pos_df, imp_neg_df = separate_pos_neg(explanations)
    combine_importances = not separate_pos_neg_imp

    importances_pos = imp_pos_df.values
    importances_neg = imp_neg_df.values

    if importances_pos.ndim == 2:
        importances_pos = np.mean(importances_pos, axis=0)
        importances_neg = np.mean(importances_neg, axis=0)

    xlim_pos = np.max(importances_pos)*1.25
    xlim_neg = np.max(np.abs(importances_neg))*1.25

    if combine_importances:
        xlim_pos += xlim_neg
        xlim_neg = 0
        importances_pos += np.abs(importances_neg)

    else:
        xlim_pos = np.max([xlim_pos, xlim_neg])
        xlim_neg = -1 * xlim_pos

    x_pos = (np.arange(len(feature_names)))

    # Plotting begins
    plt.figure(figsize=(10, 10))
    width = 0.3

    if combine_importances:
        plt.barh(x_pos, importances_pos, width, align='center')
    else:
        plt.barh(x_pos, importances_pos, width, align='center')
        plt.barh(x_pos + width, importances_neg, width, align='center')

    plt.yticks(x_pos + width/2, feature_names, wrap=True)
    plt.ylabel(axis_title)
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([xlim_neg, xlim_pos])

    if save_fig is not None:
        plt.savefig(save_fig)
