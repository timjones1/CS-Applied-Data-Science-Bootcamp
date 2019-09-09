# Predictions of the Outcome of Kickstarter Campaigns

Kickstarter is a crowdfunding platform with a community of more than 10 million people comprising of creative, tech enthusiasts who help in bringing new projects to life.

Until now, more than $3 billion dollars have been contributed by the members in fueling creative projects.
The projects can be literally anything – a device, a game, an app, a film etc.

Kickstarter works on all or nothing basis: a campaign is launched with a certain amount they want to raise, if it doesn’t meet its goal, the project owner gets nothing.
For example: if a projects’s goal is $5000. Even if it gets funded till $4999, the project won’t be a success.

If you have a project that you would like to post on Kickstarter now, can you predict whether it will be successfully funded or not? Looking into the dataset, what useful information can you extract from it, which variables are informative for your prediction and can you interpret the model?

The goal of this project is to build a classifier to predict whether a project will be successfully funded or not. You can use the algorithm of your choice.

**Notes on the dataset**:
* The target, `state` corresponds to a binary outcome: `0` for failed, `1` for successful.
* The variables `'deadline'', 'created_at', 'launched_at'` are stored in Unix time format.


## Get the data

You can download the dataset from [here](https://s3-eu-west-1.amazonaws.com/kate-datasets/kickstarter_basic/data.zip).

The data is given as a zipped csv to save space. *NO NEED TO UNZIP IT*, Pandas is able to work directly with zipped csv.

In a notebook, run:

```
import pandas as pd

df = pd.read_csv("data.zip")
```

to load the dataset.


## Get Started

You will need to implement three functions:

* `preprocess`

This takes a dataframe and should return three dataframes: `X`, `y` (your training data) as well as `X_eval`.

`X_eval` is the evaluation data, it needs to be preprocessed so KATE can generate prediction and evaluate the performance of your model. In the dataframe provided, there is a column `evaluation_set` that tells you whether this row is for evaluation or not.

To get all the rows that need to be used for evaluation only, you can use:

```
df.loc[df.evaluation_set]
```

* `train`

This takes the X and y you have processed previously and trains your model. It should return your trained model.


* `predict`

This takes the model you have trained as well as a test set (on KATE this will be `X_eval` that you processed above, but you can test this function locally with your own test set). 

This should return y_pred, predictions on the test set.


The recommended way of working on this project is to:
1) Download the data
2) Open it in a notebook and start prototyping
3) Break down your code into functions preprocess/train/predict and test it locally
4) When you're happy with your functions, copy/paste them in the WebIDE (in the file `model.py`)

*NOTE*: Since with this project your model will be trained directly on KATE, it is limited to models that can be trained under 1min. You will receive a `TimeoutError` if your model takes too long.

You can test that your functions work in a notebook with the following example:

```
import pandas as pd

df = pd.read_csv("data.zip")
X, y, X_eval = preprocess(df)
model = train(X, y)
y_pred = predict(model, X_eval)
print(y_pred)
```

## Baseline Model

Here is an example of a submission, building a simple logistic regression with only two features: `goal_usd` (adjusted goal) and `usa` (whether the campaign happened in the US)


```
from sklearn.linear_model import LogisticRegression

def preprocess(df):

    df["usa"] = df["country"] == "US"
    df["goal_usd"] = df["goal"] * df["static_usd_rate"]

    # save labels to know what rows are in evaluation set
    # evaluation_set is a boolean so we can use it as mask
    msk_eval = df.evaluation_set

    df = df[["goal_usd", "usa", "state"]]

    X = df[~msk_eval].drop(["state"], axis=1)
    y = df[~msk_eval]["state"]
    X_eval = df[msk_eval].drop(["state"], axis=1)

    return X, y, X_eval

def train(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred
```


Good luck!