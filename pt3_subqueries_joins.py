# Part 3: Subqueries & Joins


def top_postcodes_for_chain_stores():
    """
    From the businesses table, select the top 10 most popular postal_code.
    They should be filtered to only count the restaurants owned by people/entities that own 5 or more restaurants.
    The result should:
    * return a row (postal_code, frequency) for each 10 selection
    * sort by descending order to get the most relevant zip codes
    :return: a string representing the SQL query
    :rtype: str
    """
    return "SELECT postal_code,count(1) AS count FROM (SELECT postal_code,count(1) AS count_owned FROM businesses GROUP BY owner_name HAVING count_owned > 5) GROUP BY postal_code ORDER BY count DESC LIMIT 10"


def inspection_scores_in_94103():
    """
    First let's get an idea about the inspection score our competition has.
    Based on multiple inspections, find out the minimum Score (as "min_score"),
    average Score (as "avg_score") and maximum Score (as "max_score") for all restaurant in post code "94103".
    The average score should be rounded to one decimal.
    :return: a string representing the SQL query
    :rtype: str
    """
    return "SELECT MIN(score) AS min_score,MAX(score) AS max_score, ROUND(AVG(score),1) AS avg_score FROM inspections INNER JOIN (SELECT business_id FROM businesses WHERE postal_code = '94103') AS businesses_in_postcode ON inspections.business_id = businesses_in_postcode.business_id"


def risk_categories_in_94103():
    """
    Now lets get more serious, and look at how many times restaurants with postal code 94103
    (that's Market street) has committed health violations and group them based on their risk category.
    The output should be (risk_category, count as frequency) and sorted in descending order by frequency
    :return: a string representing the SQL query
    :rtype: str
    """
    return "SELECT risk_category, count(1) FROM violations INNER JOIN (SELECT business_id FROM businesses  WHERE postal_code = '94103') AS businesses_in_postcode ON violations.business_id = businesses_in_postcode.business_id GROUP BY risk_category"
