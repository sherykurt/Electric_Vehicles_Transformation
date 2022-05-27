#!/usr/bin/env python
# coding: utf-8

# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import widgets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# reading the cleaned data
df = pd.read_csv("./ev_project_data_ready.csv", index_col=0)
df.head()

#Dropping unnecessary columns which are the same as Zip column. 
df = df.drop(['GEO_ID', 'NAME'], axis=1)
df.shape  #>>(16081, 57)


#creating an interactive plot to see the relationship of each feature(X) with y(Registered EV). 
features_dict = df.columns

X = df.drop('VIN', axis=1)
y = df['VIN']

def ev_plot(X, y):
    def plotter(column):
        valid_rows = X[column].notna()
        
        lr = LinearRegression()
        lr.fit(X[[column]], y) 
        y_pred = lr.predict(X[[column]])

        plt.plot(X.loc[valid_rows, column], y_pred, color='b', label='model prediction')
        plt.plot(X.loc[valid_rows, column], y[valid_rows], '.', color = 'k', label='training data')
        plt.ylabel('Registered Electric Vehicle (EV)')
        plt.legend()
    
    return plotter

widgets.interact(ev_plot(X, y), column=features_dict);


#creating an interactive plot to see the relationship of each feature(X) with y(EV charging stations). 
features_dict = df.columns

X = df.drop('stations', axis=1)
y = df['stations']

def evstations_plot(X, y):
    def plotter(column):
        valid_rows = X[column].notna()
        
        lr = LinearRegression()
        lr.fit(X[[column]], y) 
        y_pred = lr.predict(X[[column]])
        
        plt.plot(X.loc[valid_rows, column], y_pred, color='b', label='model prediction')
        plt.plot(X.loc[valid_rows, column], y[valid_rows], '.', color='k', label='training data')
        
        plt.ylabel('EV charging stations')
        plt.legend()
    
    return plotter

widgets.interact(evstations_plot(X, y), column=features_dict);


#MACHINE LEARNING

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

df['Year'].describe()

# converting Year and Zip into categorical values
df['Year'] = df['Year'].astype('category')
df['Zip'] = df['Zip'].astype('category')
type(df['Zip'][1])

#selecting Features and label for modelling
X = df.iloc[:,:-1]
y = df['stations']


#Separating the last year (2020) from the df
#plotting the EV station counts over the years(2011-2020)

df.groupby('Year')['stations'].aggregate('sum').plot(kind='bar', color='b', rot=0)

plt.xlabel('Year')
plt.ylabel('EV Charging Stations')
plt.title('EV Charging Stations Count Each Year in NY')
plt.show()
#to save the plot
plt.savefig('EV Charging Satation Count Each Year in NY.png', dpi=600)



#Data before 2020:
df_pre2020 = df[df['Year'] != 2020]
df[df['Year'] != 2020].shape #-->(14474, 57)

#Data for 2020:
df_2020 = df[df['Year'] == 2020]
df[df['Year'] == 2020].shape #-->(1607, 57)

#Remove year variable from year before modelling
df_pre2020 = df_pre2020.loc[:, df_pre2020.columns != 'Year']
print(f'the shape of df before 2020: {df_pre2020.shape}')

df_2020 = df_2020.loc[:, df_2020.columns !='Year']
print(f'the shape of df for 2020: {df_2020.shape}')


# select numeric and categorical columns:
# X-> features without target variable('stations')

X = df_pre2020.iloc[:,:-1]
y = df_pre2020.iloc[:,-1]

numeric_columns = list(X.select_dtypes(include=[np.number]).columns.values)


# XGBoost Regression for model prediction
# XGBoost version:
import xgboost; print(xgboost.__version__)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


# cross-validation via GridSearchCV
XGBreg = XGBRegressor(
            tree_method="hist",
            eval_metric=mean_absolute_error
        )

params = { 'regressor__max_depth': [4,6,8],
           'regressor__learning_rate': [0.03, 0.05, 0.07],
           'regressor__colsample_bytree': [0.1, 0.3]}


features = ColumnTransformer([
#                             ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                            ('numeric', StandardScaler(), numeric_columns)
                        ])

evStation_xgboost_pipe = Pipeline([
                        ('features', features),
                        ('regressor', XGBreg)
                    ])

model_gs = GridSearchCV(estimator=evStation_xgboost_pipe, 
                     param_grid=params, 
                     cv=10, 
                     n_jobs=2,
                     verbose=1)


model_gs.fit(X, y)
print(model_gs.best_params_) #Fitting 10 folds for each of 18 candidates, totalling 180 fits
                             #{'regressor__colsample_bytree': 0.3, 'regressor__learning_rate': 0.05,
                             # 'regressor__max_depth': 6}

model_gs.predict(X)
model_gs.score(X, y) #-> 0.7244642179982401


# prepare 2020 data for prediction
X_2020 = df_2020.iloc[:,:-1]
y_2020 = df_2020.iloc[:, -1]


# predict stations for 2020 data:
stations_pred_2020 = model_gs.predict(X_2020)
stations_pred_2020


# predictions from XGboost cv
stations_XGCVpred = model_gs.predict(X_2020)
stations_XGCVpred


# Model Evaluation 


from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Model Evaluation
print(f'explained_variance_score: {explained_variance_score(y_2020, stations_pred_2020)}')
print(f'max_error: {max_error(y_2020, stations_XGCVpred)}')
print(f'mean_absolute_error: {mean_absolute_error(y_2020, stations_XGCVpred)}')
print(f'root_mean_squared_error: {mean_squared_error(y_2020, stations_XGCVpred, squared=False)}')
print(f'R^2_score: {r2_score(y_2020, stations_XGCVpred)}')

rmse = mean_squared_error(y_2020, stations_XGCVpred, squared=False)
print (rmse)

scatter_index= rmse/(df['stations'].mean()*100)
print(scatter_index)


# stations in 2020 actual and predicted
stations_20 = pd.concat([y_2020.reset_index(),pd.DataFrame({'stations_predicted':stations_XGCVpred})], axis=1)
stations_20

# concat zip codes to this
stations_all_2020 = pd.concat([stations_20,pd.DataFrame(X_2020.Zip).reset_index().Zip], axis=1)
stations_all_2020


# convert Zip to string
stations_all_2020 = stations_all_2020.astype({'Zip':str})
stations_all_2020.info()


# Loading geoJSON for all zipcodes in the state of NY 


#load geoJSON
import json
with open('./ny_new_york_zip_codes_geo.min.json','r') as jsonFile:
    data = json.load(jsonFile)
tmp = data
len(tmp)

#number of zip codes
len(tmp['features']) 

#filter zip codes in my data:
geozips = []
for i in range(len(tmp['features'])):
    if tmp['features'][i]['properties']['ZCTA5CE10'] in list(stations_all_2020.Zip):
        geozips.append(tmp['features'][i])

new_json = dict.fromkeys(['type', 'features'])
new_json['type'] = 'FeatureCollection'
new_json['features'] = geozips

#save new json as new file
#open('updated_zips.json', 'w').write(json.dumps(new_json, sort_keys = True, indent = 4, separators=(',',':')))


ziplist = list(stations_all_2020.Zip)
ziplist.sort()

len(new_json['features'])

ny_zips_included = 'updated_zips.json'

stations_all_2020_dict = stations_all_2020.set_index('Zip')['stations']
stations_all_2020_dict


# Altair Interactive Plot and ChoroPleth Mapping 
# pip install altair vega_datasets ## if it is required

import altair as alt
import geopandas as gpd
from vega_datasets import data

#load geoJSON saved our folder before (to avoid running each cell in notebook)
import json
with open('./updated_zips.json','r') as jsonFile:
    ny_json = json.load(jsonFile)

gdf_ny = gpd.GeoDataFrame.from_features(ny_json)

stations_all_2020.stations_predicted = stations_all_2020.stations_predicted.round()
stations_all_2020.head()

gdf_ny = gdf_ny.merge(stations_all_2020, left_on='ZCTA5CE10', right_on='Zip', how='inner')
gdf_ny.head()

# Layered bar(EV stations actual count for 2020) and line(EV stations predicted count for 2020) graphs
# for each Zipcode(count# 1607 in this data) in NY state

source = gdf_ny
base = alt.Chart(source).encode(x='Zip:O')
bar = base.mark_bar().encode(y='stations:Q')
line =  base.mark_line(color='red').encode(
    y='stations_predicted:Q'
)


tooltips = [
    alt.Tooltip('Zip', title = 'Zip', format='0i'),
    alt.Tooltip('stations', title = 'stations_actual', format='0i'),
    alt.Tooltip('stations_predicted', title='stations_predicted', format='0i')
]

bar=bar.encode(tooltip=tooltips)

(bar + line)


# Altair Choropleth 

choro_json = json.loads(gdf_ny.to_json())
choro_data = alt.Data(values=choro_json['features'])


def gen_map(geodata, color_column, title, tooltip, color_scheme):
    '''
    Generates NY Zip map with stations choropleth
    '''
    
    # Add Base Layer
    base = alt.Chart(geodata, title = title).mark_geoshape(
        stroke='black',
        strokeWidth=1
    ).encode(
    ).properties(
        width=800,
        height=800
    )
    # Add Choropleth Layer
    choro = alt.Chart(geodata).mark_geoshape(
        #fill='lightgray',
        stroke='black'
    ).encode(
        alt.Color(color_column, 
                  type='quantitative', 
                  scale=alt.Scale(scheme=color_scheme),
                  title = "Stations"),
         tooltip=tooltip
    )
    
    return base + choro


tooltips = [
    alt.Tooltip('properties.Zip:O', title = 'Zip', format='0i'),
    alt.Tooltip('properties.stations:Q', title = 'stations_actual', format='0i'),
    alt.Tooltip('properties.stations_predicted:Q', title='stations_predicted', format='0i')
    ]

ny_map = gen_map(geodata=choro_data, color_column='properties.stations_predicted', 
                 title=f'Predicted Charging Station counts in NY', 
                  tooltip=tooltips,
                 color_scheme='yelloworangered')

ny_map


#Machine Learning For Registered EV in the state of NY

# Recalling the original dataframe df
df.info()

#plotting the registered EV counts over the years(2011-2020)
df.groupby('Year')['VIN'].aggregate('sum').plot(kind='bar', rot=0, color='b')

plt.xlabel('Year')
plt.ylabel('Registered EV')
plt.title('Registered EV count each Year')
plt.show()
#to save the figure
plt.savefig('Registered EV count each Year in NY state.png', dpi = 600)

#Data before 2020:
df_pre2020 = df[df['Year'] != 2020]
df[df['Year'] != 2020].shape

#Data for 2020:
df_2020 = df[df['Year'] == 2020]
df[df['Year'] == 2020].shape


#Remove year variable from year before modelling
df_pre2020 = df_pre2020.loc[:, df_pre2020.columns != 'Year']
print(f'the shape of df before 2020: {df_pre2020.shape}')

df_2020 = df_2020.loc[:, df_2020.columns !='Year']
print(f'the shape of df for 2020: {df_2020.shape}')

# select numeric and categorical columns for model XGBoost
Xev = df_pre2020.drop('VIN', axis=1, inplace=False)
yev = df_pre2020['VIN']

numeric_columns = list(Xev.select_dtypes(include=[np.number]).columns.values)
#categorical_columns = ['Zip']


#XGBoost Regression for model prediction

#GridSearchCV cross-validation
XGB_regression = XGBRegressor(
            tree_method="hist",
            eval_metric=mean_absolute_error
        )

params = { 'regressor__max_depth': [3,6,10,12],
           'regressor__learning_rate': [0.03, 0.05, 0.07, 0.1],
           'regressor__colsample_bytree': [0.3, 0.5, 0.7]}


features = ColumnTransformer([
                            #('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                            ('numeric', StandardScaler(), numeric_columns)
                        ])

evRegistered_xgboost_pipe = Pipeline([
                        ('features', features),
                        ('regressor', XGB_regression)
                    ])

model_gs_ev = GridSearchCV(estimator=evRegistered_xgboost_pipe, 
                     param_grid=params, 
                     cv=10,  
                     n_jobs=2,
                     verbose=1)

model_gs_ev.fit(Xev, yev)
print(model_gs_ev.best_params_) #--> {'regressor__colsample_bytree': 0.3, 'regressor__learning_rate': 0.03,
                                #     'regressor__max_depth': 12}


model_gs_ev.predict(Xev)
model_gs_ev.score(Xev, yev) #-->0.9709030722574984


# prepare 2020 data for prediction
X_2020 = df_2020.drop('VIN', axis=1, inplace=False)
y_2020 = df_2020['VIN']

# predict stations for 2020 data:
evRegistered_pred_2020 = model_gs_ev.predict(X_2020)
evRegistered_pred_2020

# predictions from XGboost cv
evRegistered_XGCVpred = model_gs_ev.predict(X_2020)
evRegistered_XGCVpred


# Model Evaluation 

print(f'explained_variance_score: {explained_variance_score(y_2020, evRegistered_pred_2020)}')
print(f'max_error: {max_error(y_2020, evRegistered_XGCVpred)}')
print(f'mean_absolute_error: {mean_absolute_error(y_2020, evRegistered_XGCVpred)}')
print(f'root_mean_squared_error: {mean_squared_error(y_2020, evRegistered_XGCVpred, squared=False)}')
print(f'R^2_score: {r2_score(y_2020, evRegistered_XGCVpred)}')


rmse = mean_squared_error(y_2020, evRegistered_XGCVpred, squared=False)
print (rmse)

scatter_index= rmse/(df['VIN'].mean()*100)
print(scatter_index)


# Registered EV in 2020 actual and predicted
evRegistered_20 = pd.concat([y_2020.reset_index(),pd.DataFrame({'RegisteredEV_predicted':evRegistered_XGCVpred})], axis=1)
evRegistered_20['RegisteredEV_actual'] = evRegistered_20['VIN']
evRegistered_20.drop(columns='VIN', inplace=True)
evRegistered_20

# concat zip codes to this
evRegistered_all_2020 = pd.concat([evRegistered_20,pd.DataFrame(X_2020.Zip).reset_index().Zip], axis=1)
evRegistered_all_2020

# convert Zip to string
evRegistered_all_2020 = evRegistered_all_2020.astype({'Zip':str})
evRegistered_all_2020.info()


#filter zip codes in my data:
geozips = []
for i in range(len(tmp['features'])):
    if tmp['features'][i]['properties']['ZCTA5CE10'] in list(evRegistered_all_2020.Zip):
        geozips.append(tmp['features'][i])

new_json = dict.fromkeys(['type', 'features'])
new_json['type'] = 'FeatureCollection'
new_json['features'] = geozips


ziplist1 = list(evRegistered_all_2020.Zip)
ziplist1.sort()


evRegistered_all_2020_dict = evRegistered_all_2020.set_index('Zip')['RegisteredEV_actual']
evRegistered_all_2020_dict


# Altair Interactive Plot and ChoroPleth Mapping for Electric Vehicle prediction

evRegistered_all_2020.RegisteredEV_predicted = evRegistered_all_2020.RegisteredEV_predicted.round()
evRegistered_all_2020.head()


gdf_ny_evRegistered = gdf_ny.merge(evRegistered_all_2020, left_on='ZCTA5CE10', right_on='Zip', how='inner')
gdf_ny_evRegistered.head()


import altair as alt
from vega_datasets import data

source = gdf_ny_evRegistered

base = alt.Chart(source).encode(x='Zip_y:O')

bar = base.mark_bar().encode(y='RegisteredEV_actual:Q')

line =  base.mark_line(color='red').encode(
    y='RegisteredEV_predicted:Q'
)



tooltips = [
    alt.Tooltip('Zip_y', title = 'Zip', format='0i'),
    alt.Tooltip('RegisteredEV_actual', title = 'EV_actual', format='0i'),
    alt.Tooltip('RegisteredEV_predicted', title='EV_predicted', format='0i')
]

bar=bar.encode(tooltip=tooltips)

(bar + line) 

import altair as alt
choro_json = json.loads(gdf_ny_evRegistered.to_json())
choro_data = alt.Data(values=choro_json['features'])


def gen_map_ev(geodata, color_column, title, tooltip, color_scheme):
    '''
    Generates NY Zip map with stations choropleth
    '''
    
    # Add Base Layer
    base = alt.Chart(geodata, title = title).mark_geoshape(
        stroke='black',
        strokeWidth=1
    ).encode(
    ).properties(
        width=800,
        height=800
    )
    # Add Choropleth Layer

    choro = alt.Chart(geodata).mark_geoshape(
        #fill='lightgray',
        stroke='black'
    ).encode(
        alt.Color(color_column, 
                  type='quantitative', 
                  scale=alt.Scale(scheme=color_scheme),
                  title = "EV counts"),
         tooltip=tooltip
    )
    
    return base + choro


tooltips_1 = [
    alt.Tooltip('properties.Zip_y:O', title = 'Zip', format='0i'),
    alt.Tooltip('properties.RegisteredEV_actual:Q', title = 'EV_actual', format='0i'),
    alt.Tooltip('properties.RegisteredEV_predicted:Q', title='EV_predicted', format='0i')
    ]


ny_ev_map = gen_map_ev(geodata=choro_data, color_column='properties.RegisteredEV_predicted', 
                 title=f'Predicted EV counts in NY', 
                 tooltip=tooltips_1, 
                 color_scheme='yelloworangered')
""
ny_ev_map

