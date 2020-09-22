import numpy as np


def build_sequences(min_value, max_value, sequence_number):
    """
    Write a function that can generate the following sequences:
        sequence #1: 2 * n + 1
        sequence #2: 50 - 5 * n
        sequence #3: 2 ** n

    Although this exercises can easily be done with list
    comprehensions, it can be more efficient to use numpy
    (the arange method can be handy here).

    Start by generating all 50 first values for the sequence that
    was selected by sequence_number and return a numpy array
    filtered so that it only contains values in
    [min_value, max_value] (min and max being included)

    :param min_value: minimum value to use to filter the arrays
    :param max_value: maximum value to use to filter the arrays
    :param sequence_number: number of the sequence to return
    :returns: the right sequence as a np.array
    """
    source = np.arange(50, dtype=np.int64)
    if sequence_number == 1:
        for index, value in enumerate(source):
            source[index] = 2 * value + 1
    elif sequence_number == 2:
        for index, value in enumerate(source):
            source[index] = 50 - 5 * value
    else:
        for index, value in enumerate(source):
            source[index] = 2 ** value
    res = source[(source >= min_value) & (source <= max_value)]
    return res


def moving_averages(x, k):
    """
    Given a numpy vector x of n > k, compute the moving averages
    of length k.  In other words, return a vector z of length
    m = n - k + 1 where z_i = mean([x_i, x_i-1, ..., x_i-k+1])

    Note that z_i refers to value of z computed from index i
    of x, but not z index i. z will be shifted compared to x
    since it cannot be computed for the first k-1 values of x.

    Example inputs:
    - x = [1, 2, 3, 4]
    - k = 3

    the moving average of 3 is only defined for the last 2
    values: [3, 4].
    And z = np.array([mean([1,2,3]), mean([2,3,4])])
        z = np.array([2.0, 3.0])

    :param x: numpy array of dimension n > k
    :param k: length of the moving average
    :returns: a numpy array z containing the moving averages.
    """
    res = []
    for index_x in range(len(x)):
        if k <= len(x[index_x:]):
            index_last = index_x + k
            sequence = x[index_x:index_last]
            res += [np.mean(sequence)]
    res = np.array(res)
    return res


def block_matrix(A, B):
    """
    Given two numpy matrices A and B of arbitrary dimensions,
    return a new numpy matrix of the following form:
        [A,0]
        [0,B]

    Example inputs:
        A = [1,2]    B = [5,6]
            [3,4]        [7,8]

    Expected output:
        [1,2,0,0]
        [3,4,0,0]
        [0,0,5,6]
        [0,0,7,8]

    :param A: numpy array
    :param B: numpy array
    :returns: a numpy array with A and B on the diagonal.
    """
    rows_a = A.shape[0]
    columns_a = A.shape[1]
    rows_b = B.shape[0]
    columns_b = B.shape[1]
    a_modified = np.append(A, np.zeros((rows_a, columns_b)), axis=1)
    b_modified = np.append(np.zeros((rows_b, columns_a)), B, axis=1)
    joined_matrix = np.append(a_modified, b_modified, axis=0)
    return joined_matrix
