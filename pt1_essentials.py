# Part 1: Essentials


def business_ids_count():
    """
    Write a SQL query that finds the number of business ids in the businesses table
    :return: a string representing the SQL query
    :rtype: str
    """
    return "SELECT COUNT(business_id) FROM businesses"


def unique_business_names_count():
    """
    Write a SQL query that finds out how many unique business names are registered
    with San Francisco Food health investigation organization
    and name the column as unique restaurant name count.
    :return: a string representing the SQL query
    :rtype: str
    """
    return "SELECT count(DISTINCT name) AS unique_restaurant_name FROM businesses"


def first_and_last_investigation():
    """
    Write a SQL query that finds out what is the earliest and latest date
    a health investigation is recorded in this database.
    :return: a string representing the SQL query
    :rtype: str
    """
    return "SELECT MIN(date), MAX(date) FROM inspections"


def business_local_owner_count():
    """
    How many businesses are there in San Francisco where their owners live
    in the same area (postal code/ zip code) as the business is located?
    :return: a string representing the SQL query
    :rtype: str
    """
    return "SELECT Count(1) AS count_same_business_owner_address FROM businesses AS b1 INNER JOIN businesses AS b2 ON b1.business_id = b2.business_id WHERE b1.postal_code = b2.owner_zip"


def business_local_owner_reg_count():
    """
    Out of those businesses, how many of them has a registered business certificate?
    :return: a string representing the SQL query
    :rtype: str
    """
    return "SELECT count(b1.business_certificate) FROM businesses AS b1 INNER JOIN businesses AS b2 ON b1.business_id = b2.business_id WHERE b1.postal_code = b2.owner_zip AND b1.business_certificate IS NOT NULL"
