# Big Data

In this project, you will be asked to implement a set of functions defined in a file "spark_rdd.py". Each function takes a PySpark RDD called dataset and should return a specific object.

## Spark RDD: HackerNews data analysis challenge

You will analyse a dataset of (almost) all submitted HackerNews posts. 

There are two ways you can submit your code:

1) Using the WebIDE

Click on "Use the WebIDE" and write your solutions directly in the spark_rdd.py file. You can download the dataset here https://s3-eu-west-1.amazonaws.com/kate-datasets/hackernews/HNStories.zip to test locally

2) Cloning a repository and working with git

Click on `Clone Using HTTPS` (or SSH but that requires setup), that will copy a URL you can use to clone the repository locally. This repository has a script `get_data.py` allowing you to download the dataset, a notebook that helps you loading the dataset into an RDD and test your functions. You can then write your solutions in the `spark_rdd.py` file and push them via git