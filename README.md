# Electric Vehicles (EVs) Transformation
---------
## Goal
--------
Predict future demand for Electric Vehicles (EVs) and public access EVs charging stations at zipcode level in NY state. 

## Current Problem 
-------
Even though the EVs market is increasing due to the tendency of green energy, the infrastructure to fully tranform to EVs is insufficient. This causes range anxiety meaning that people are worried about how far they can travel in an electrical car before its charge is out. 
This project will attempt to address the insufficiency of charging station by predicting how many charging stations required to be installed in each zipcode in NY. The project will provide following information,

* Current EVs on route
* Current EVs charging stations
* Prediction of EVs
* Prediction of EVs charging stations 

## Executive Summary
------
EVs Tranform is a project that predicts the EVs market and potential charging stations for users (such as state, car companies, and electric companies for installing EV infrastructe) to accelerate the use of EVs.

The main steps involve in the project were;
   1. Gathering several data
       * NY State EV dataset (2011-2020)
       * NY State EV charging stations dataset (2011-2020)
       * Census dataset (2011-2020)
   2. Data Wrangling (cleaning, merging datasets, etc.)
   3. Ploting the actual EVs vs predicted EVs and actual    charging stations vs predicted stations
   4. Creating Choropleth mapping at the zipcode level to show these actuals and predicted values for the year of 2020. 

## Technologies Used
--------
* Python
* Pandas
* Matplotlib
* Geocoder to get lats/longs of addresses
* SKLearn pipelines
* Altair -- Chropleth Mapping, plotting


