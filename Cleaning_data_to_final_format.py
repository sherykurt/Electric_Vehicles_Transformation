#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#reading the data 
census_demographic = pd.read_excel("./census_bureau.xlsx", sheet_name=2)
census_demographic

census_employment = pd.read_excel("./census_bureau.xlsx", sheet_name=3)
census_employment.head()

ev_charging_stationsNY= pd.read_csv("./Electric_Vehicle_Charging_Stations_in_New_York.csv")
ev_charging_stationsNY.head()

ny_ev_registrations= pd.read_csv("/Users/serifeakkurt/Desktop/EV Capstone Project /ny_ev_registrations.csv")
ny_ev_registrations.info()

ny_ev_registrations['VIN'].nunique()

ev_regist_original = ny_ev_registrations[ny_ev_registrations['Registration']=='Original']
ev_regist_original.info()

ev_regist_original['Registration'].describe()

ev_regist_original['VIN'].nunique() #number of unique vehicle ID numbers in ev_regist_original file

ev_regist_original['VIN'].nunique()

ev_regist_original.groupby('ZIP Code').count()['VIN']

# as seen below, one vehicle still has multiple rows of different DMV records in the original registration dataset
ev_regist_original[ev_regist_original.VIN=='YV4BR0PM7J1376582']

# without loss of critical information we can select only rows with unique VIN values from the original registration data
ev_regist_original_uniqueVIN = ev_regist_original.drop_duplicates('VIN')

ev_regist_original_uniqueVIN.info()

ev_regist_original_uniqueVIN['ZIP Code'].nunique()

ev_regist_original_uniqueVIN[ev_regist_original_uniqueVIN['ZIP Code']==99999].count()


# which zip codes in the registration data is not included in the census zip codes
# filter the data as such
ev_regist_original_uniqueVIN[~ev_regist_original_uniqueVIN['ZIP Code'].isin(census_demographic['ZIP Code'].unique())]


# 884 zip codes in the registration data is not included in the census zip codes

ev_regist_original_uniqueVIN[~ev_regist_original_uniqueVIN['ZIP Code'].isin(census_demographic['ZIP Code'].unique())]['ZIP Code'].nunique()


# which zip codes in the registration data is not included in the census zip codes
ev_regist_original_uniqueVIN[~ev_regist_original_uniqueVIN['ZIP Code'].isin(census_demographic['ZIP Code'].unique())]['ZIP Code'].unique()

census_demographic['ZIP Code'].unique()

# filter the data to include only zip codes included in the census data
ev_regist_ZIPinCensus = ev_regist_original_uniqueVIN[ev_regist_original_uniqueVIN['ZIP Code'].isin(census_demographic['ZIP Code'].unique())]
ev_regist_ZIPinCensus

ev_regist_ZIPinCensus.shape


ev_regist_ZIPinCensus.info()

# create a RegDate field as datetime object
ev_regist_ZIPinCensus['RegDate'] = pd.to_datetime(ev_regist_ZIPinCensus['Registration Valid Date'])
ev_regist_ZIPinCensus['Year'] = ev_regist_ZIPinCensus['RegDate'].dt.year
ev_regist_ZIPinCensus

# create a dataframe which has number of registrations by zip code and years
ev_regist_byZIP = ev_regist_ZIPinCensus.groupby(['ZIP Code','Year']).count()['VIN'].reset_index()
ev_regist_byZIP

# EV cars are registered in 1587 different Zip codes
ev_regist_byZIP['ZIP Code'].nunique()

# TOP SELLERS: which car models sold the most
ny_ev_registrations.drop_duplicates('VIN')['Vehicle Name'].value_counts()

ny_ev_registrations.info()

census_demographic.info()

census_demographic["End Date"].max()

census_employment.info()

census_demographic['ZIP Code'].nunique()

census_employment['ZIP Code'].nunique()

ny_ev_registrations['ZIP Code'].nunique()

ev_charging_stationsNY['ZIP'].nunique()

ev_charging_stationsNY['Open Date'].min()

ev_charging_stationsNY['Open Date'].max()

ny_ev_registrations['Registration Valid Date'].nunique()

ny_ev_registrations['Registration Valid Date'].min()

ny_ev_registrations['Registration Valid Date'].max()

pd.to_datetime(ny_ev_registrations['Registration Valid Date']).dt.year.max()


# Some next steps for data wrangling and merge.
# - Convert all dates into date format, and add months or years variables to all datasets
# - NOT NEEDED ANYMORE! <- drop unnecessary variables from demography data (Household income in Past12 Months)
# - Apply many to one merge using zipcode and year information
# - DONE! for the recent years 2019, 2020, 2021 either download new data or merge using 2018 data for all
# - DONE! there are missing census data for some zipcodes, either lose them or download data!
# - convert variable types as appropriate
# - create time evolution type graphs
# - DONE! notice that there are many invalid Zip code values in the registration file

# Reading census data

# importing the module
import pandas as pd

import os
os.listdir('censuses')

# read specific columns of 2011 DP02 csv file using Pandas
dp02_2011 = pd.read_csv("censuses/ACSDP5Y2011.DP02_data.csv", usecols = ['GEO_ID','NAME',
            'DP02_0058E','DP02_0059E','DP02_0060E','DP02_0061E','DP02_0062E','DP02_0063E','DP02_0064E','DP02_0065E',])
dp02_2011.head()

# read all DP02 files using a loop: Create filenames
for year in range(2011,2021):
        print('ACSDP5Y'+str(year)+'.DP02_data.csv')

# create some dynamic variable names in the global environment for dp02 files
for i in range(2011, 2021):
    globals()[f"dp02_{i}"] = i
print(dp02_2020)

# years from 2011 to 2018 DP02 files have same varnames:
for year in range(2011,2019):
        # read csv file by selecting specific columns
    globals()[f"dp02_{year}"] = pd.read_csv("censuses/ACSDP5Y"+str(year)+".DP02_data.csv", 
                                            usecols = ['GEO_ID','NAME',
                                                       'DP02_0058E','DP02_0059E',
                                                       'DP02_0060E','DP02_0061E','DP02_0062E',
                                                       'DP02_0063E','DP02_0064E','DP02_0065E'], skiprows = [1])
       # adding year column 
    globals()[f"dp02_{year}"]['Year'] = year
        # change variable names
    globals()[f"dp02_{year}"].rename(columns={'DP02_0058E':'edu_pop_over25', 
                                              "DP02_0059E":"edu_less_than9grade",
                                              "DP02_0060E":"edu_from9to12_nodiploma", 
                                              "DP02_0061E":"edu_high_school_grad",
                                              "DP02_0062E":"edu_college_nodegree", 
                                              "DP02_0063E":"edu_associate_degree",
                                              "DP02_0064E":"edu_bs_degree", 
                                              "DP02_0065E":"edu_grad_degree"}, inplace = True)
dp02_2011.head()

# years from 2019 to 2020 DP02 files have same varnames:
for year in range(2019,2021):
        globals()[f"dp02_{year}"] = pd.read_csv("censuses/ACSDP5Y"+str(year)+".DP02_data.csv", usecols = ['GEO_ID','NAME',
            'DP02_0059E','DP02_0060E','DP02_0061E','DP02_0062E','DP02_0063E','DP02_0064E','DP02_0065E','DP02_0066E'], skiprows = [1])
        # adding year column 
        globals()[f"dp02_{year}"]['Year'] = year
        # renaming columns
        globals()[f"dp02_{year}"].rename(columns= {'DP02_0059E':'edu_pop_over25',
                                                   'DP02_0060E':'edu_less_than9grade',
                                                   'DP02_0061E':'edu_from9to12_nodiploma',
                                                   'DP02_0062E':'edu_high_school_grad',
                                                   'DP02_0063E':'edu_college_nodegree',
                                                   'DP02_0064E':'edu_associate_degree',
                                                   'DP02_0065E':'edu_bs_degree',
                                                   'DP02_0066E':'edu_grad_degree'}, inplace=True)
dp02_2020.head()

## READ DP03 files
# years from 2011 to 2020 all DP03 files have same varnames:
for year in range(2011,2021):
    globals()[f"dp03_{year}"] = pd.read_csv("censuses/ACSDP5Y"+str(year)+".DP03_data.csv", 
                                            usecols = ['GEO_ID','NAME','DP03_0004E','DP03_0009PE','DP03_0018E',
                                                       'DP03_0019E','DP03_0020E','DP03_0021E','DP03_0022E',
                                                       'DP03_0023E','DP03_0024E','DP03_0025E','DP03_0039E',
                                                       'DP03_0040E','DP03_0041E','DP03_0051E','DP03_0057E',
                                                       'DP03_0058E','DP03_0059E','DP03_0060E','DP03_0061E',
                                                       'DP03_0062E'], skiprows = [1])
    # adding year column
    globals()[f"dp03_{year}"]['Year'] = year
    # rename variables
    globals()[f"dp03_{year}"].rename(columns= {'DP03_0004E':'emp_employed',
                                               'DP03_0009PE':'emp_precent_unemployed',
                                               'DP03_0018E':'commute_workers_over16years',
                                               'DP03_0019E':'commute_drovealone',
                                               'DP03_0020E':'commute_carpooled',
                                               'DP03_0021E':'commute_public_transp',
                                               'DP03_0022E':'commute_walked',
                                               'DP03_0023E':'commute_other_means',
                                               'DP03_0024E':'commute_worked_athome',
                                               'DP03_0025E':'commute_travel_time',
                                               'DP03_0039E':'industry_IT',
                                               'DP03_0040E':'industry_financial_realest',
                                               'DP03_0041E':'industry_prof_management',
                                               'DP03_0051E':'total_households',
                                               'DP03_0057E':'househld_inc_50_75',
                                               'DP03_0058E':'househld_inc_75_100',
                                               'DP03_0059E':'househld_inc_100_150',
                                               'DP03_0060E':'househld_inc_150_200',
                                               'DP03_0061E':'househld_inc_200_more',
                                               'DP03_0062E':'househld_median_income'}, inplace = True)

                                                
dp03_2011.head()

## READ DP04 files
# years from 2011 to 2014 all DP03 files have same varnames:
for year in range(2011, 2015):
    globals()[f"dp04_{year}"] = pd.read_csv("censuses/ACSDP5Y"+str(year)+".DP04_data.csv", 
                                            usecols= ['GEO_ID','NAME', 
                                                      'DP04_0001E','DP04_0007E','DP04_0007PE',
                                                      'DP04_0008E','DP04_0008PE','DP04_0013E',
                                                      'DP04_0013PE','DP04_0056E','DP04_0057E',
                                                      'DP04_0058E','DP04_0059E','DP04_0060E',
                                                      'DP04_0079E','DP04_0080E','DP04_0081E',
                                                      'DP04_0082E','DP04_0083E','DP04_0084E',
                                                      'DP04_0085E','DP04_0086E','DP04_0087E','DP04_0088E'], 
                                            skiprows = [1])
    # adding year column
    globals()[f"dp04_{year}"]['Year'] = year
    #rename variables
    globals()[f"dp04_{year}"].rename(columns={'DP04_0001E':'housing_total_units',
                                              'DP04_0007E':'housing_1unit_detached',
                                              'DP04_0007PE':'housing_1unit_detached_percent',
                                              'DP04_0008E':'housing_1unit_attached',
                                              'DP04_0008PE':'housing_1unit_attached_percent',
                                              'DP04_0013E':'housing_20moreunits',
                                              'DP04_0013PE':'housing_20moreunits_percent',
                                              'DP04_0056E':'housing_occupied_units',
                                              'DP04_0057E':'househld_no_vehicles',
                                              'DP04_0058E':'househld_vehicles_1',
                                              'DP04_0059E':'househld_vehicles_2',
                                              'DP04_0060E':'househld_vehicles_3more',
                                              'DP04_0079E':'housing_owner_occupied_units',
                                              'DP04_0080E':'house_value_less50',
                                              'DP04_0081E':'house_value_50_100',
                                              'DP04_0082E':'house_value_100_150',
                                              'DP04_0083E':'house_value_150_200',
                                              'DP04_0084E':'house_value_200_300',
                                              'DP04_0085E':'house_value_300_500',
                                              'DP04_0086E':'house_value_500_1mil',
                                              'DP04_0087E':'house_value_1mil_more',
                                              'DP04_0088E':'house_value_median'}, inplace=True)

## READ DP04 files
# years from 2015 to 2020 all DP03 files have same varnames:
for year in range(2015, 2021):
    globals()[f"dp04_{year}"] = pd.read_csv("censuses/ACSDP5Y"+str(year)+".DP04_data.csv", 
                                            usecols= ['GEO_ID','NAME',
                                                      'DP04_0001E','DP04_0007E','DP04_0007PE','DP04_0008E',
                                                      'DP04_0008PE','DP04_0013E','DP04_0013PE','DP04_0057E',
                                                      'DP04_0058E','DP04_0059E','DP04_0060E','DP04_0061E',
                                                      'DP04_0080E','DP04_0081E','DP04_0082E','DP04_0083E',
                                                      'DP04_0084E','DP04_0085E','DP04_0086E','DP04_0087E',
                                                      'DP04_0088E','DP04_0089E'], skiprows = [1])
    #adding year column
    globals()[f"dp04_{year}"]['Year'] = year
    #rename variables
    globals()[f"dp04_{year}"].rename(columns={'DP04_0001E':'housing_total_units',
                                              'DP04_0007E':'housing_1unit_detached',
                                              'DP04_0007PE':'housing_1unit_detached_percent',
                                              'DP04_0008E':'housing_1unit_attached',
                                              'DP04_0008PE':'housing_1unit_attached_percent',
                                              'DP04_0013E':'housing_20moreunits',
                                              'DP04_0013PE':'housing_20moreunits_percent',
                                              'DP04_0057E':'housing_occupied_units',
                                              'DP04_0058E':'househld_no_vehicles',
                                              'DP04_0059E':'househld_vehicles_1',
                                              'DP04_0060E':'househld_vehicles_2',
                                              'DP04_0061E':'househld_vehicles_3more',
                                              'DP04_0080E':'housing_owner_occupied_units',
                                              'DP04_0081E':'house_value_less50',
                                              'DP04_0082E':'house_value_50_100',
                                              'DP04_0083E':'house_value_100_150',
                                              'DP04_0084E':'house_value_150_200',
                                              'DP04_0085E':'house_value_200_300',
                                              'DP04_0086E':'house_value_300_500',
                                              'DP04_0087E':'house_value_500_1mil',
                                              'DP04_0088E':'house_value_1mil_more',
                                              'DP04_0089E':'house_value_median'}, inplace=True)

dp04_2019.head()


# ## Renaming columns
# renaming dp02 files
dp02_dict_2011_2018 = {'DP02_0058E':'edu_pop_over25',
"DP02_0059E":"edu_less_than9grade",
"DP02_0060E":"edu_from9to12_nodiploma",
"DP02_0061E":"edu_high_school_grad",
"DP02_0062E":"edu_college_nodegree",
"DP02_0063E":"edu_associate_degree",
"DP02_0064E":"edu_bs_degree",
"DP02_0065E":"edu_grad_degree"}

dp02_2011.rename(columns=dp02_dict_2011_2018,
          inplace=True)
dp02_2011.head()

dp02_2011.shape

# years from 2019 to 2020 DP02 files have same varnames:
for year in range(2019,2021):
        globals()[f"dp02_{year}"].rename(columns= {'DP02_0059E':'edu_pop_over25','DP02_0060E':'edu_less_than9grade',
                                                    'DP02_0061E':'edu_from9to12_nodiploma','DP02_0062E':'edu_high_school_grad',
                                                   'DP02_0063E':'edu_college_nodegree','DP02_0064E':'edu_associate_degree',
                                                   'DP02_0065E':'edu_bs_degree','DP02_0066E':'edu_grad_degree'}, inplace=True)
dp02_2019.head()

dp02_all=pd.concat([dp02_2011, dp02_2012,dp02_2013,dp02_2014,dp02_2015,dp02_2016,dp02_2017,
                    dp02_2018,dp02_2019,dp02_2020])
dp02_all.head()

dp03_all=pd.concat([dp03_2011, dp03_2012,dp03_2013,dp03_2014,dp03_2015,dp03_2016,dp03_2017,
                    dp03_2018,dp03_2019,dp03_2020])
dp03_all.head()

dp04_all=pd.concat([dp04_2011, dp04_2012,dp04_2013,dp04_2014,dp04_2015,dp04_2016,dp04_2017,
                    dp04_2018,dp04_2019,dp04_2020])
dp04_all.head()

# merging dp02_all, and dp03_all on GEO_ID, Year, NAME
dp02_dp03 = dp02_all.merge(dp03_all, on = ['GEO_ID', 'Year', 'NAME'])

# merge dp02_dp03 and dp04_all on GEO_ID, Year, NAME
dp_all= dp02_dp03.merge(dp04_all, on = ['GEO_ID', 'Year', 'NAME'])

dp_all

print(dp02_all.shape)
print(dp03_all.shape)
print(dp04_all.shape)
print(dp_all.shape)

dp_all['NAME'].nunique()


# ## Charging station data

ev_charging_stationsNY.info()

ev_charging_stationsNY['Open Date'] = pd.to_datetime(ev_charging_stationsNY['Open Date'])

ev_charging_stationsNY['Year'] = ev_charging_stationsNY['Open Date'].dt.year

ev_charging_by_Zip_year=ev_charging_stationsNY.groupby(['ZIP', 'Year'])['ID'].count().reset_index()
ev_charging_by_Zip_year['stations'] = ev_charging_by_Zip_year['ID']
ev_charging_by_Zip_year


# ## Merging demographic data with employment and EV data

#energy_cost_saving['domain'] = energy_cost_saving['company email'].apply(lambda email: email.split('@')[1]                                                                        # if pd.notna(email) and '@'in email else email)
dp_all['Zip'] = dp_all['NAME'].apply(lambda x: int(x.split(' ')[1]))
dp_all

ev_regist_byZIP

census_employment=census_employment.assign(business_size = census_employment['Employment Size of Establishment'].apply(lambda x:
    {
        'Establishments with 1 to 4 employees': 'small_business',
        'Establishments with 5 to 9 employees': 'small_business',
        'Establishments with 10 to 19 employees': 'small_business',
        'Establishments with 20 to 49 employees': 'small_business',
        'Establishments with 50 to 99 employees': 'medium_business',
        'Establishments with 100 to 249 employees': 'medium_business',
        'Establishments with 250 to 499 employees': 'large_business',
        'Establishments with 500 to 999 employees': 'large_business',
        'Establishments with 1,000 employees or more': 'large_business' 
    }[x]))

census_employment_groupbyZip_size=census_employment.groupby(['ZIP Code', 'business_size'])['Total Establishments'].sum().reset_index()


census_employment_groupbyZip_size

#print df.pivot(index='Salesman',columns='idx')[['product','price']]
ZIP_business_by_size=census_employment_groupbyZip_size.pivot(index='ZIP Code', columns='business_size')['Total Establishments']
ZIP_business_by_size = pd.DataFrame(ZIP_business_by_size.reset_index())
ZIP_business_by_size

dp_all.info()

# merge demographic dp_all with employment data
dp_all_employment = dp_all.merge(ZIP_business_by_size, how = 'left', left_on = ['Zip'], right_on='ZIP Code', validate="m:1")
dp_all_employment.info()


dp_all_business_ev_regist =dp_all_employment.merge(ev_regist_byZIP, how='left', 
                         left_on=['Zip', 'Year'], right_on=['ZIP Code', 'Year'])
dp_all_business_ev_regist.info()

all_in_one=dp_all_business_ev_regist.merge(ev_charging_by_Zip_year, how='left', 
                                 left_on=['Zip','Year'], right_on=['ZIP','Year'])
all_in_one

ev_charging_by_Zip_year.info()


ZIP_business_by_size.info()


ev_regist_byZIP.info()

# to save merged and cleaned data
all_in_one.to_csv('ev_project_data_all_in_one.csv')

