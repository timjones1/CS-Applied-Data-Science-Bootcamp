import numpy as np
import pandas as pd


def nan_processor(df, replacement_str):
    """
    Author:Tim Jones

    :param df: Input data frame (pandas.DataFrame)
    :param replacement_str: string to find and replace by np.nan
    :returns: DataFrame where the occurences of replacement_str have been
        replaced by np.nan and subsequently all rows containing np.nan have
        been removed
    """

    def replace_string(x):
        if x == replacement_str:
            return np.nan
        return x
    df_rep = df.applymap(replace_string)
    df_rep.dropna(inplace=True)
    return df_rep


def feature_cleaner(df, low, high):
    """
    Author: Tim Jones

    :param df:      Input DataFrame (with numerical columns)
    :param low:     Lowest percentile  (0.0<low<1.0)
    :param high:    Highest percentile (low<high<1.0)
    :returns:       Scaled DataFrame where elements that are outside of the
                    desired percentile range have been removed
    """

    def nan_outside_quantiles(p):
        # set any values outside of low/high quantile range to zero
        low_q = p.quantile(low)
        high_q = p.quantile(high)
        p[~((p <= high_q) & (p >= low_q))] = np.nan
        return p

    def normalize(x):
        # normalizefeatures by subtracting the mean and dividing by std dev
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return (x - mean) / std

    df_quant = nan_outside_quantiles(df)
    df_quant.dropna(inplace=True)
    return normalize(df_quant)


def get_feature(df):
    """
    Author: Tim Jones

    :param df:  Input DataFrame (with numerical columns)
    :returns:   Name of the column with largest K
    """
    df0 = df[df["CLASS"] == 0]
    df1 = df[df["CLASS"] == 1]

    coef = pd.DataFrame([(d.max() - d.min())/d.var() for d in [df0, df1]])
    K_ratio = np.max(coef.apply(max, axis=0) / coef)
    return K_ratio.sort_values(ascending=False).index[0]


def one_hot_encode(label_to_encode, labels):
    """
    Author: Tim Jones

    :param label_to_encode: the label to encode
    :param labels: a list of all possible labels
    :return: a list of 0s and one 1
    """
    lbls = pd.Series(labels)
    if lbls.nunique() != len(lbls):
        return [0] * len(lbls)

    return list(lbls.str.match(label_to_encode).apply(int))
