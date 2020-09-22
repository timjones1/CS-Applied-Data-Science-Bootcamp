"""
This exercise contains functions where you need to use pandas to
apply various data aggregations and transformations.
"""
import pandas as pd


def get_prices_for_heaviest_item(inventory):
    """
    Write a function that takes a pandas.DataFrame with four columns: "category", "price", "weight", "in_stock"
    and returns a pandas.Series containing the price of the heaviest weight per category of items still in stock.
    You can assume that no items in the same category have the same weight to make things simpler to reason about.
    The returned Series should not have an index name and the values should be sorted in descending order.
    You should return an empty Series if there are not items in stock for all categories.

    Example input:

        category      price weight  in_stock
    0   electronics   400   740     False
    1   health        5     100     False
    2   electronics   300   6000    True
    3   books         20    300     True


    Note: entries of in_stock are booleans.
    Expected output:

    electronics    300
    books          20
    dtype: int64


    :param inventory: pandas.DataFrame with four column "category", "price", "weight", "in_stock"
    :return: a pandas.Series with the category as index and the selected prices in descending order

    """
    if inventory[inventory['in_stock']].empty:
        res = pd.Series()
    else:
        index_heavy = inventory[inventory['in_stock']].groupby('category')['weight'].idxmax()
        price_heavy = inventory.iloc[index_heavy].groupby('category')['price'].mean().sort_values(ascending=False)
        res = pd.Series(price_heavy.values, index=price_heavy.index)
    return res


def reshape_temperature_data(measurements):
    """
    Write a function that takes a pandas.DataFrame with 7 columns:
     "location", 'Jan-2018', 'Feb-2018', 'Mar-2018', "April-2018", "May-2018", "June-2018".
    This DataFrame represents temperature measurements in the first two quarters of 2018 for a particular city.
    This function should return a new DataFrame containing three columns: "location", "Date", "Value"
    and where each row represents a measurement in particular location at a particular date.
    The returned pandas.DataFrame should sort the values by location first and then by temperature measurement.
    It should also drop any missing values and reset the index of the returned DataFrame.

    NOTE: If measurements is empty your function should return and empty dataframe:
        location       date   value


    Example input:

       location  Jan-2018  Feb-2018  Mar-2018  April-2018  May-2018  June-2018
    0  Brussels         2         3         8        12.0        14         17
    1     Paris         2         3         9         NaN        15         18

    Expected output:

        location        date  value
    0   Brussels    Jan-2018    2.0
    1   Brussels    Feb-2018    3.0
    2   Brussels    Mar-2018    8.0
    3   Brussels  April-2018   12.0
    4   Brussels    May-2018   14.0
    5   Brussels   June-2018   17.0
    6      Paris    Jan-2018    2.0
    7      Paris    Feb-2018    3.0
    8      Paris    Mar-2018    9.0
    9      Paris    May-2018   15.0
    10     Paris   June-2018   18.0


    :param measurements: pandas.DataFrame with seven columns:
    "location", 'Jan-2018', 'Feb-2018', 'Mar-2018', "April-2018", "May-2018", "June-2018"
    :return: a pandas.DataFrame containing three columns "location", "date", "value" with a row
    for each temperature measurement in a given location. There should be no missing values.
    """
    if measurements.empty:
        res = pd.DataFrame({'location': [],
                            'date': object(),
                            'value': []})
    else:
        df = measurements.set_index('location').transpose()
        df = pd.DataFrame(df.stack())
        res = pd.DataFrame({'location': df.index.get_level_values(1),
                            'date': df.index.get_level_values(0),
                            'value': df[0].values})
        res = res.sort_values(by=['location', 'value']).reset_index(drop=True)
    return res


def compute_events_matrix_count(events):
    """
    Write a function that takes a pandas.DataFrame containing 2 columns representing web events for a user:
    "user_id" and "event".
    This function should return a new DataFrame where each event value becomes a new column in the returned DataFrame.
    We expect the columns (events) to be in alphabetical order.

    For each event value, you need to calculate the count of that particular event for each userid.
    Missing values should be filled with 0.
    Effectively, this function calculates the number of occurrence for each event type (columns) for each user (rows).
    You should return an empty Series if the input DataFrame is empty.

    Example input:

        user_id	event
    0	1234	click
    1	4321	click
    2	1234	click
    3	1234	play
    4	4321	play
    5	3456	pause

    Expected output:

            click	pause	play
    1234	2.0	    0.0	    1.0
    3456	0.0	    1.0	    0.0
    4321	1.0	    0.0	    1.0


    :param events: pandas.DataFrame with two columns: "user_id" and "event"
    :return: a pandas.DataFrame returning the number of occurrence for each event type (columns) for each user (rows).
    """
    if events.empty:
        res = pd.Series()
    else:
        res = events.groupby(['user_id', 'event']).size().unstack().fillna(0)
    return res
