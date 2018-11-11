# Toronto AirBnB Listings Analysis

This repo contains a quick analysis of the Toronto AirBnB listings.

### Objectives

The reason for creating this repository is to potentially build an application for helping Airbnb hosts price their listings in an easy and data-driven way.

Imagine being a new host on Airbnb, having a tool that does the required analysis for you in terms of the neighborhood that you are in as well as information about the property, would get you up and running quickly.

Another reason is curiosity about the effects of applications such as AirBnB on the city's rent prices but is outside the scope of this project.

We will be answer the following questions:

* Which neighborhoods have the most expensive price per bed?
* Which neighborhoods have the highest number of listings?
* Which property types are the most listed in AirBnB in Toronto?
* What’s the average listing price by property types?
* What’s the average price per bed price by property type?
* What does the occupancy rate look like by neighborhood?

Acknowledgements
Acknowledging sources of dataset, references and other resources.
Summary
Describing some of the key findings:

### Requirements & Libraries

* Python (3.5+)
* Pandas
* Matplotlib
* Numpy
* Scikit-learn
* XGBoost (optional)
* TPot Regressor (optional - recommended)


### Acknowledgements

All downloaded data in this project can be found on the [InsideAirbnb](http://insideairbnb.com/get-the-data.html) website.

For quick visualizations of the data, see [Toronto's map](http://insideairbnb.com/toronto/).


### Getting Started

First, you'll have to unzip the required csv files from 'data/toronto-october.zip' and place
all the csv files into the main '/data' folder.

Then run the [Jupyter (iPython) notebook](airbnb.ipynb) with the following command:

```
jupyter notebook ./airbnb.ipynb
```

The iPython notebook contains a highlevel walk-through of the code and visualizations
of the data.

If you would like to view the code details, then `lib.py` is what you are looking for.
This file contains the code used in the iPython notebook along with documentation
of the usage of each function.


### Article & Summary

The Medium article associated with this repository can be [found here](https://medium.com/@faisalaltameemi/airbnb-listings-analysis-in-toronto-october-2018-2a5358bae007).

### Results

Here is a summary of the modeling r2 scores:

| Model Name                              |  Training Score | 	Validation Score | 	Test Score |
| --------------------------------------- | --------------- | ----------------- | ----------- |
| NaiveModel (always predicts mean price) | 	0.000000       | 	-0.000172        | 	-0.000609  |
| RandomForestRegressor                   | 	0.711069       | 	0.487592         | 	0.489714   |
| PLSRegression                           | 	0.459397       | 	0.445175         | 	0.432827   |
| LinearRegression                        | 	0.534462       | 	0.488490         | 	0.487435   |

Key findings and results are detailed in the article linked above.