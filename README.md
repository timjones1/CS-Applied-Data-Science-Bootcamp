# Instructions

In this exercise, you are working with a relational dataset that captures information
about food health investigations carried out in San Francisco and their outcomes.

Imagine you want to start a new restaurant in San Francisco.
You want to stay away from big restaurant owners who own multiple restaurants.
You also want to pick an area that is trending (Hence, alot of restaurants).

In this exercise, you will write SQL queries to find out the best area and answer business intelligence questions.

The database is stored under `data/sfscores.sqlite`

The database consists of *3 tables*. The schemas are shown below.

* `businesses`: information relating to restaurant businesses
* `inspections`: information about individual inspection events
* `violations`: information about violation events

The queries you need to implement have increased difficulty. By the end of this assignment, the focus will be on answering multipart business questions using multistep queries or multiple tables.


## businesses
```
CREATE TABLE businesses (
    business_id INTEGER NOT NULL,
    name VARCHAR(64),
    address VARCHAR(50),
    city VARCHAR(23),
    postal_code VARCHAR(9),
    latitude FLOAT,
    longitude FLOAT,
    phone_number BIGINT,
    "TaxCode" VARCHAR(4),
    business_certificate INTEGER,
    application_date DATE,
    owner_name VARCHAR(99),
    owner_address VARCHAR(74),
    owner_city VARCHAR(22),
    owner_state VARCHAR(14),
    owner_zip VARCHAR(15)
)
```

## violations
```
CREATE TABLE violations (
    business_id TEXT NOT NULL,
    date INTEGER NOT NULL,
    ViolationTypeID TEXT NOT NULL,
    risk_category TEXT NOT NULL,
    description TEXT NOT NULL
)
```

## inspections
```
CREATE TABLE inspections (
    business_id TEXT NOT NULL,
    Score INTEGER,
    date INTEGER NOT NULL,
    type VARCHAR (33) NOT NULL
)

```

# How to submit

You will need to complete the files `pt1_essentials.py`, `pt2_groupby.py` and `pt3_subqueries_joins` with your SQL queries.

You are welcome to use the associated notebook to play around with different queries.