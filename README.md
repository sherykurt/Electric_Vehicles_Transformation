# Electric Vehicles (EVs) Transformation

## Goal

Predict future demand for Electric Vehicles (EVs) and public access EVs charging stations at zipcode level in NY state. 

## Current Problem 

Even though the EVs market is increasing due to the tendency of green energy, the infrastructure to fully tranform to EVs is insufficient. This causes range anxiety meaning that people are worried about how far they can travel in an electrical car before its charge is out. 
This project will attempt to address the insufficiency of charging station by predicting how many charging stations required to be installed in each zipcode in NY. In 2020, the total EVs The project provides following information,

> Current EVs on route
![Registered EV count each Year in NY state](https://user-images.githubusercontent.com/77508831/172192476-5099d33b-edd2-4b35-a7b4-3161953ec4fa.png)

> Current EVs charging stations
![EV Charging Station Count Each Year in NY](https://user-images.githubusercontent.com/77508831/172192207-a68f9848-d442-4ef7-ae4e-cf818c4e13ed.png)

> Actual EVs vs Prediction of EVs
![ActualEV_ _PredictedEV vs Zipcode in NY](https://user-images.githubusercontent.com/77508831/172192788-de5f0d42-c71d-40ff-99d4-97ae72977748.png)
<img width="706" alt="EV_predicted" src="https://user-images.githubusercontent.com/77508831/172193902-bbc778b0-c693-40ca-a386-fc7b521f7da0.png">

> Actual EVs charging stations vs Prediction of the stations
![ActualChargingStation   Predicted vs Zipcode in NY](https://user-images.githubusercontent.com/77508831/172193240-5ca2e81f-cfbe-4c56-8cdf-b9d67c2bfbfd.png) 

<img width="703" alt="EV_Charging_Station_Pred_NY" src="https://user-images.githubusercontent.com/77508831/172193917-b45b0c24-8dd7-4e42-8fc8-185ec2ac6cad.png">

## Executive Summary

EVs Tranform is a project that predicts the EVs market and potential charging stations for users (such as state, car companies, and electric companies for installing EV infrastructe) to accelerate the use of EVs.

The main steps involve in the project were;
   1. Gathering several data
       * NY State EV dataset (2011-2020)
       * NY State EV charging stations dataset (2011-2020)
       * Census dataset (2011-2020)
   2. Data Wrangling (cleaning, merging datasets, etc.)
   3. Ploting the actual EVs vs predicted EVs and actual charging stations vs predicted stations
   4. Creating Choropleth mapping at the zipcode level to show these actuals and predicted values for the year of 2020. 

## Technologies Used

* Python
* Pandas
* Matplotlib
* Geocoder to get lats/longs of addresses
* SKLearn pipelines
* Altair -- Chropleth Mapping, plotting


