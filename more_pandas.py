"""This exercise contains functions where you need to use pandas to build new columns from existing ones."""


def diff_in_days(df):
    """
    Write a function that takes a pandas DataFrame with two columns "time_1"
    and "time_2" of UNIX timestamps given in seconds (you will need to specify
    the unit if using pd.to_datetime).

    The function should return a new dataFrame with one single column
    "difference_days" consisting of the absolute difference in days between
    time_1 and time_2.

    Example input:

               time_1      time_2
        0  1456694829  1455845363

    Here we have a single row for which time_1 corresponds to 28/02/2016 and
    time_2 to 19/02/2016.

    Expected output:
           difference_days
        0                9

    Note:
    https://en.wikipedia.org/wiki/Unix_time,
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html

    Hint:
    Take special care on how negative timedeltas are treated in Python.
    Getting the number of days directly from a negative timedelta might
    not give you the result you expect.

    :param df: DataFrame with the two columns of timestamps
    :return: new dataframe with differences in days between timestamps
    """

    raise NotImplementedError


def return_location(df):
    """
    Write a function that takes a pandas DataFrame with one column, locations,
    containing information about a specific location. The info is stored in a string
    that can be loaded as a json object.
    The function should return a DataFrame with one column, "short_name" that contains the
    value associated with the key "short_name" for each row.

    Note: you can assume all strings are exactly in the format given below though
    possibly longer and with different keys.

    Example input:
                                              locations
        0  {"short_name": "Detroit, MI", "id": 2391585}
        1    {"short_name": "Tracy, CA", "id": 2507550}

    Where each value is a string such as:
       '{"short_name": "Detroit, MI", "id": 2391585}'

    Expected output:
            short_name
        0  Detroit, MI
        1    Tracy, CA

    Hint: you might want to use json.loads from the json library
    together with .apply from pandas to extract the correct key from
    the json object.

    :param df: DataFrame with the locations column
    :return: new DataFrame with the short_name column
    """

    raise NotImplementedError


def return_post_codes(df):
    """
    Write a function that takes a pandas DataFrame with one column, text, that
    contains an arbitrary text. The function should extract all post-codes that
    appear in that text and concatenate them together with " | ". The result is
    a new dataframe with a column "postcodes" that contains all concatenated
    postcodes.

    Example input:
                                                                            text
    0  Great Doddington, Wellingborough NN29 7TA, UK\nTaylor, Leeds LS14 6JA, UK
    1  This is some text, and here is a postcode CB4 9NE

    Expected output:

                postcodes
    0  NN29 7TA | LS14 6JA
    1              CB4 9NE

    Note: Postcodes, in the UK, are of one of the following form where `X` means
    a letter appears and `9` means a number appears:

    X9 9XX
    X9X 9XX
    X99 9XX
    XX9 9XX
    XX9X 9XX
    XX99 9XX

    Even though the standard layout is to include one single space
    in between the two halves of the post code, there are occasional formating
    errors where an arbitrary number of space is included (0, 1, or more). You
    should parse those codes as well.

    :param df: a DataFrame with the text column
    :return: new DataFrame with the postcodes column
    """

    raise NotImplementedError


