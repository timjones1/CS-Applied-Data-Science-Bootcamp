# HackerNews Data Analysis with PySpark

In this project, you will be asked to implement a set of functions to analyze a dataset containing (almost) all posts from HackerNews.

The functions you need to implement are defined in a file `spark_rdd.py`. Each function takes a PySpark RDD called dataset and should return a specific object.

## Getting started

There are two ways you can submit your code:

1) Using the WebIDE

Write your solutions directly in the `spark_rdd.py` file. 

For testing purposes, you can download the dataset here https://s3-eu-west-1.amazonaws.com/kate-datasets/hackernews/HNStories.zip and load it to a PySpark RDD.

After you make your first submission, you will also be able to test your code with the same samples used on KATE.


2) Cloning a repository and working with git

Click on `Clone Using HTTPS` (or SSH but that requires setup). That will copy a URL you can use to clone the repository locally. 

This repository has a script `get_data.py` allowing you to download the dataset, and notebook that helps you loading the dataset into an RDD and test your functions. You can then write your solutions in the `spark_rdd.py` file and push them via git