import numpy as np

def build_sequences(min_value, max_value, sequence_number):
    if sequence_number == 1:
        seq = 2 * np.arange(50) + 1
    elif sequence_number == 2:
        seq = 50 - 5 * np.arange(50)
    elif sequence_number ==3:
        seq = 2 ** np.arange(50)
    else:
        return print('Invalid sequence')
    
    a = np.array(seq)
    idx = np.where((a >= min_value) & (a <= max_value))

    return a[idx]

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


def moving_averages(x, k):
    r = np.cumsum(x, dtype = float)
    r[k:] = r[k:] - r[:-k]
    m = r[k-1:] / k
    return m

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


def block_matrix(A, B):
    A = np.array(A)
    B = np.array(B)
    ar = np.size(A,0)
    ac = np.size(A,1)
    br = np.size(B,0)
    bc = np.size(B,1)
    
    C = np.block([
        [A,np.zeros((ar,bc))],
        [np.zeros((br,ac)), B]
        ])
    C = C.astype(int)
    
    return C

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
