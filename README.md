# Toronto AirBnB Listings Analysis

This repo contains a quick analysis of the Toronto AirBnB listings.

The reason for creating this repository is to potentially build an application for helping Airbnb hosts price their listings in an easy and data-driven way.

Imagine being a new host on Airbnb, having a tool that does the required analysis for you in terms of the neighborhood that you are in as well as information about the property, would get you up and running quickly.

Another reason is curiosity about the effects of applications such as AirBnB on the city's rent prices but is outside the scope of this project.

### Requirements

* Python (3.5+)
* Pandas
* Matplotlib
* Numpy
* Scikit-learn
* XGBoost (optional)
* TPot Regressor (optional - recommended)


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


### Article

The Medium article associated with this repository can be [found here](https://medium.com/@faisalaltameemi/airbnb-listings-analysis-in-toronto-october-2018-2a5358bae007).
