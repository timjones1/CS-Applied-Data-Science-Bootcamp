from datetime import datetime as dt
import re


def count_elements_in_dataset(dataset):
    """
    Given a dataset loaded on Spark, return the
    number of elements.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: number of elements in the RDD
    """
    return dataset.count()


def get_first_element(dataset):
    """
    Given a dataset loaded on Spark, return the
    first element
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: the first element of the RDD
    """
    return dataset.take(1)[0]


def get_all_attributes(dataset):
    """
    Each element is a dictionary of attributes and their values for a post.
    Can you find the set of all attributes used throughout the RDD?
    The function dictionary.keys() gives you the list of attributes of a
    dictionary.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: all unique attributes collected in a list
    """

    dataset_keys = dataset.map(lambda x: (x.keys()))

    return dataset_keys.reduce(lambda a, b: list(set(a) | set(b)))


def get_elements_w_same_attributes(dataset):
    """
    We see that there are more attributes than just the one used in the first
     element.
    This function should return all elements that have the same attributes
    as the first element.

    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD containing only elements with same attributes as the
    first element
    """
    first_element_keys = set(dataset.take(1)[0].keys())
    return dataset.filter(lambda x: set(x.keys()) == first_element_keys)


def get_min_max_timestamps(dataset):
    """
    Find the minimum and maximum timestamp in the dataset
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: min and max timestamp in a tuple object
    :rtype: tuple
    """
    def extract_time(timestamp):
        return dt.utcfromtimestamp(timestamp)

    rdd_ex_time = dataset.map(lambda x: extract_time(x.get('created_at_i')))

    min_time = rdd_ex_time.reduce(lambda a, b: a if a < b else b)
    max_time = rdd_ex_time.reduce(lambda a, b: a if a > b else b)

    return (min_time, max_time)


def get_number_of_posts_per_bucket(dataset, min_time, max_time):
    """
    Using the `get_bucket` function defined in the notebook (redefine it in
     this file), this function should return a
    new RDD that contains the number of elements that fall within each bucket.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :param min_time: Minimum time to consider for buckets (datetime format)
    :param max_time: Maximum time to consider for buckets (datetime format)
    :return: an RDD with number of elements per bucket
    """

    def get_bucket(rec, min_timestamp, max_timestamp):
        """
        funtion provided to get the interval between min and max times
        """
        interval = (max_timestamp - min_timestamp + 1) / 200.0
        bucket_nr = int((rec['created_at_i'] - min_timestamp)/interval)

        return bucket_nr

    min_ts = dt.timestamp(min_time)
    max_ts = dt.timestamp(max_time)

    buckets_rdd = dataset.map(
        lambda x: (get_bucket(x, min_ts, max_ts), 1))

    return buckets_rdd


def get_hour(rec):
    """
    Helper function defined by cambridge spark used in multiple other
    functions
    """
    time = dt.utcfromtimestamp(rec['created_at_i'])
    return time.hour


def get_number_of_posts_per_hour(dataset):
    """
    Using the `get_hour` function defined in the notebook (redefine it in this
     file), this function should return a
    new RDD that contains the number of elements per hour.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with number of elements per hour
    """
    def get_hour(rec):
        time = dt.utcfromtimestamp(rec['created_at_i'])
        return time.hour

    hour_data = dataset.map(lambda x: (get_hour(x), 1))
    hours_buckets_rdd = hour_data.reduceByKey(lambda a, b: a+b)
    return hours_buckets_rdd


def get_score_per_hour(dataset):
    """
    The number of points scored by a post is under the attribute `points`.
    Use it to compute the average score received by submissions for each hour.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with average score per hour
    """
    hour_score_rdd = dataset.map(
        lambda x: (get_hour(x), (x.get("points"), 1)))

    score_tot_per_hour = hour_score_rdd.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1]))

    scores_per_hour_rdd = score_tot_per_hour.mapValues(
        lambda x: x[0] / x[1])

    return scores_per_hour_rdd


def get_proportion_of_scores(dataset):
    """
    It may be more useful to look at sucessful posts that get over 200 points.
    Find the proportion of posts that get above 200 points per hour.
    This will be the number of posts with points > 200 divided by the total
     number of posts at this hour.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with the proportion of scores over 200 per hour
    """
    success_posts_rdd = dataset.map(
        lambda x: (get_hour(x), (int(x.get("points") > 200), 1))
        )
    success_posts_per_hour_rdd = success_posts_rdd.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
        )
    prop_per_hour_rdd = success_posts_per_hour_rdd.mapValues(
        lambda x: x[0] / x[1]
        )
    return prop_per_hour_rdd


def get_proportion_of_success(dataset):
    """
    Using the `get_words` function defined in the notebook to count the
    number of words in the title of each post, look at the proportion
    of successful posts for each title length.

    Note: If an entry in the dataset does not have a title, it should
    be counted as a length of 0.

    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with the proportion of successful post per title length
    """

    def get_word_len(line):
        try:
            res = len(re.compile('\w+').findall(line))
        except TypeError:
            return 0
        else:
            return res

    words_success_rdd = dataset.map(
        lambda x: (
            get_word_len(x.get("title")),
            (int(x.get("points") > 200), 1)
        )
    )

    success_by_words_rdd = words_success_rdd.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )

    prop_per_title_length_rdd = success_by_words_rdd.mapValues(
        lambda x: x[0] / x[1]
    )

    return prop_per_title_length_rdd


def get_title_length_distribution(dataset):
    """
    Count for each title length the number of submissions with that length.

    Note: If an entry in the dataset does not have a title, it should
    be counted as a length of 0.

    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with the number of submissions per title length
    """
    title_lengths = dataset.map(lambda x: (get_word_len(x.get("title")),1))
    
    submissions_per_length_rdd = title_lengths.reduceByKey(lambda a,b: a+b)

    return submissions_per_length_rdd
