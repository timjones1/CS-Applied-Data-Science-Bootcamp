import numpy as np
import pandas as pd


def nan_processor(df, replacement_str):
    """
    Take a DataFrame and return one where all occurrences
    of the replacement string have been replaced by `np.nan`
    and, subsequently, all rows containing np.nan
    have been removed.

    Example with replacement_str='blah'
         A       B      C                   A     B    C
    --------------------------         ------------------
    0 |  0.5 |  0.3   | 'blah'         1 | 0.2 | 0.1 | 5
    1 |  0.2 |  0.1   |   5     -->    3 | 0.7 | 0.2 | 1
    2 |  0.1 | 'blah' |   3
    3 |  0.7 |  0.2   |   1

    Note: keep the original index (not reset)

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
    # below converts all columns to numeric types but KATE objects
    # df_rep = df_rep.apply(pd.to_numeric, errors="ignore")
    return df_rep


def feature_cleaner(df, low, high):
    """
    Take a dataframe where columns are all numerical and non-constant.
    For each feature, mark the values that are not between the given
    percentiles (low-high) as np.nan. If a value is exactly on the high or low
    percentile, it should be marked as nan too.

    Then, remove all rows containing np.nan.
    Finally, the columns must be scaled to have zero mean and unit variance
    (do this without sklearn).

    Example testdf:
            0     1     2
    ---------------------
    A |   0.1   0.2   0.1
    B |   5.0  10.0  20.0
    C |   0.2   0.3   0.5
    D |   0.3   0.2   0.7
    E |  -0.1  -0.2  -0.4
    F |   0.1   0.4   0.3
    G |  -0.5   0.3  -0.2
    H | -10.0   0.3   1.0

    Output of feature_cleaner(testdf, 0.01, 0.99):

                0         1         2
    ---------------------------------
    A |  0.191663 -0.956183 -0.515339
    C |  0.511101  0.239046  0.629858
    D |  0.830540 -0.956183  1.202457
    F |  0.191663  1.434274  0.057260
    G | -1.724967  0.239046 -1.374236

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
    Take a dataframe where all columns are numerical (no NaNs) and not constant
    One of the column named "CLASS" is either 0 or 1.

    Within each class, for each feature compute the ratio (R) of the
    range over the variance (the range is the gap between the smallest
    and largest value).

    For each feature you now have two R; R_0 and R_1 where:
        R_0 = (max_class0 - min_class0) / variance_class0

    For each column, compute the ratio (say K) of the larger R to the smaller R
    Return the name of the column for which this last ratio K is largest.

    Test input
           A     B     C   CLASS
    ----------------------------
    0 |  0.1   0.2   0.1     0
    1 |  5.0  10.0  20.0     0
    2 |  0.2   0.3   0.5     1
    3 |  0.3   0.2   0.7     0
    4 |	-0.1  -0.2  -0.4     1
    5 |	 0.1   0.4   0.3     0
    6 |	-0.5   0.3  -0.2     0

    Expected output: 'C'

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
    Write a function that takes in a label to encode and a list of possible
    labels. It should return the label one-hot-encoded as a list of elements
    containing 0s and a unique 1 at the index corresponding to the matching
    label. Note that the input list of labels should contain unique elements.
    If the label does not appear in our known labels, return a list of 0s.

    Examples:
    one_hot_encode("pink", ["blue", "red", "pink", "yellow"]) -> [0, 0, 1, 0]
    one_hot_encode("b", ["a", "b", "c", "d", "e"]) -> [0, 1, 0, 0, 0]
    one_hot_encode("f", ["a", "b", "c", "d", "e"]) -> [0, 0, 0, 0, 0]

    :param label_to_encode: the label to encode
    :param labels: a list of all possible labels
    :return: a list of 0s and one 1
    """
    lbls = pd.Series(labels)
    if lbls.nunique() != len(lbls):
        return [0] * len(lbls)

    return list(lbls.str.match(label_to_encode).apply(int))
