# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
import os
import math
from scipy import stats

os.chdir("C:\\Users\\hzhu\\Documents\\Ames House")

data = pd.read_csv('train.csv')
data_copy = pd.read_csv('train.csv')
###############################################################################################################
######################### Resolve Data Quality Issues #########################################################
###############################################################################################################

# Adjust screen output size
#pd.util.terminal.get_terminal_size() # get current size
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)

# There are a total of 1,460 

# Find all missing values
data.isnull().sum()
# Below are all variables with missing values
#LotFrontage       259
#Electrical          1
#MasVnrType          8
#MasVnrArea          8
#Alley            1369
#BsmtQual           37
#BsmtCond           37
#BsmtExposure       38
#BsmtFinType1       37
#BsmtFinType2       38
#FireplaceQu       690
#GarageType         81
#GarageYrBlt        81
#GarageFinish       81
#GarageQual         81
#GarageCond         81
#PoolQC           1453
#Fence            1179
#MiscFeature      1406

# Some variables have so many missing values that they are not useful for the study and need to be dropped
data = data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

###############################################################################################################
#Electrical: Electrical system
#Electrical has the following values
#       SBrkr	Standard Circuit Breakers & Romex
#       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#       Mix	Mixed

# Electrical only has 1 missing data point. For the row with missing electrical value, are there any other missing
# values?
data[data['Electrical'].isnull()==True].isnull().any(axis=0).tolist()
# The results show that there are no other missing data points in this row.

# There are 3 choices with regard to dealing with missing data points: 
# 1. discard the row
# 2. Replace with the median value or 
# 3. take the best guess with the value. 
# It is difficult to guess the value (may need to regress Electrical against rest of values to determine the closest value
# and replacing with the median has the risk of not representing the true value, although this might not affect results that
# much as this is only one value.)

# Since it is just 1 row of data, we will drop the row
data = data[data['Electrical'].isnull()==False]
#data = data.drop(data.index[1379])

###############################################################################################################
# LotFrontage: Linear feet of street connected to property
# For LotFrontage, we will assume that if it is NaN, then there is 0 feet of property connected to the street
data['LotFrontage'].fillna(0, inplace=True)

###############################################################################################################
# MasVnrType: Masonry veneer type
# Has 8 missing values. For rows where veneer type is NaN, is Veneer Area missing also? If so, then we can 
# Fill in missing values for Masonry veneer type as None.

# When MasVnrType is missing, the masonry veneer area is also missing 
data[data['MasVnrType'].isnull()==True]['MasVnrArea']

# Check that for instances where MasVnrType is none, the MasVnrArea is 0.
data[data['MasVnrType']=='None']['MasVnrArea'].value_counts(dropna=False)
#0.0      858
#1.0        2
#312.0      1
#344.0      1
#288.0      1
# This seems to be true most of the time, but there are a few exceptions

# Correct data quality issue: Replace MasVnrArea with 0 when MasVnrType is None
data.loc[(data['MasVnrType'] == "None") & (data['MasVnrArea']!=0), 'MasVnrArea'] = 0

# Fill missing values for MasVnrType with "none"
data['MasVnrType'].fillna("None", inplace=True)

# For MasVnrArea, fill that with 0
data['MasVnrArea'].fillna(0, inplace=True)


###############################################################################################################
# Basement:
# Here are all the basement-related variables: BsmtCond, BsmtExposure, BsmtFinSF1, BsmtFinSF2, 
# BsmtFinType1, BsmtFinType2, BsmtFullBath, BsmtHalfBath, BsmtQual, BsmtUnfSF

# Examine Bsmt data - some houses clearly don't have basements - why are there not the same number of NA's for 
# each basement variable?

# Check that for houses without basements, all basement var's should have missing values
data[data['BsmtCond'].isnull()][['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 
     'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF']].isnull().sum()

#BsmtCond        37
#BsmtExposure    37
#BsmtFinSF1       0
#BsmtFinSF2       0
#BsmtFinType1    37
#BsmtFinType2    37
#BsmtFullBath     0
#BsmtHalfBath     0
#BsmtQual        37
#BsmtUnfSF        0

# It seems that not all basement variables are null when Basement Condition is null
# Replace "No Basement" with "NoBase"
data.loc[data['BsmtCond'].isnull(), 'BsmtCond'] = "NoBase"
data.loc[data['BsmtExposure'].isnull(), 'BsmtExposure'] = "NoBase"           
data.loc[data['BsmtFinType2'].isnull(), 'BsmtFinType2'] = "NoBase" 
data.loc[data['BsmtFinType1'].isnull(), 'BsmtFinType1'] = "NoBase"
data.loc[data['BsmtQual'].isnull(), 'BsmtQual'] = "NoBase"

# For basements that are unfinished (BsmtUnfSF is not 0), what do the other basement data points look like?
data[data['BsmtUnfSF']>0][['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 
     'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF']].isnull().sum()

#BsmtCond        0
#BsmtExposure    1
#BsmtFinSF1      0
#BsmtFinSF2      0
#BsmtFinType1    0
#BsmtFinType2    1
#BsmtFullBath    0
#BsmtHalfBath    0
#BsmtQual        0
#BsmtUnfSF       0

# If a basement is unfinished, most basement variables still do not have missing values.
# Drop rows where basement exposure and basement Fin Type 2 variables are null

data = data.drop(data.index[np.where((data['BsmtUnfSF']>0) & (data['BsmtExposure'].isnull() | data['BsmtFinType2'].isnull()))])

# Create variable indicating whether the house has a basement or not.
# If BsmtCond is null, then house does not have a basement
data['HasBsmt'] = 1
# If the basement condition is missing, then we assume that the house does not have a basement:
data.loc[data['BsmtCond'].isnull(), 'HasBsmt'] = 0

# How many houses have basements vs. not?
data['HasBsmt'].value_counts(dropna=False)
#1    1420
#0      37


###############################################################################################################
# Garage:
# It appears that 81 houses do not have garages
data[data['GarageCond'].isnull()][['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt']].isnull().sum()
# For these 81 houses, all of the garage variables are null
#GarageCond      81
#GarageFinish    81
#GarageQual      81
#GarageType      81
#GarageYrBlt     81


# Create variable indicating whether house has garages or not
data['HasGarage'] = 1
data.loc[data['GarageCond'].isnull(), 'HasGarage'] = 0

#Replace NA with NoGarage for Garage-Related Variables
data.loc[data['GarageCond'].isnull(), 'GarageCond'] = "NoGarage"
data.loc[data['GarageFinish'].isnull(), 'GarageFinish'] = "NoGarage"  
data.loc[data['GarageQual'].isnull(), 'GarageQual'] = "NoGarage"         
data.loc[data['GarageType'].isnull(), 'GarageType'] = "NoGarage"         
data.loc[data['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 0       
         
# Do we need both Garage Cars and Garage Area?
fig = plt.figure(figsize=(5, 5))

plt.scatter(data['GarageCars'], data['GarageArea'])
plt.show()


fig = plt.figure(figsize=(5, 10))
axplot1 = fig.add_subplot(2, 1, 1)
axplot1.scatter(data['GarageCars'], data['SalePrice'])
axplot1.set_xlabel('Cars')
axplot1.set_ylabel('Sale Price')

axplot2 = fig.add_subplot(2, 1, 2)
axplot2.scatter(data['GarageArea'], data['SalePrice'])
axplot2.set_xlabel('Area')
axplot2.set_ylabel('Sale Price')

# It seems that Garage Area and Garage Cars do not have a 1-1 correlation, and that both have a positive corr with SalePrice.
# So we'll keep both vars.

# For houses without garages, how can we fill in missing values for Garage Year Built?
fig = plt.figure(figsize=(5, 5))

plt.scatter(data['GarageYrBlt'], data['SalePrice'])
plt.show()

# It seems from the below graph that most garage are built in the same year as the house, which makes sense
# out of 1,459 properties, 1089 garages were built in the same year as the house
plt.figure(figsize=(5, 5))
plt.scatter(data_org['YearBuilt'], data_org['GarageYrBlt'])
plt.show()

# Impute GarageYrBlt
data.loc[(data['GarageYrBlt']==0), 'GarageYrBlt'] = data['YearBuilt']
# Although it may be illogical to give houses that do not have a garage a year where there is a garage,
# the number is small enough that this may work

plt.figure(figsize=(5,5)) 
plt.hist(data['GarageYrBlt'])
plt.show()

###############################################################################################################
# Living Area:

# Check that 1st Flr Living Area + 2nd Flr Living Area = GrLivingArea

data['GrLivArea2'] = data['1stFlrSF']+data['2ndFlrSF']
data[data['GrLivArea2']!=data['GrLivArea']][['GrLivArea2', 'GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 
    'MSSubClass']]

data[data['GrLivArea2']!=data['GrLivArea']].shape
# 26 observations have the total General Living Area not equal to 1st Flr Area + 2nd Flr Area.
# Upon examination of these data rows, the General Living Area in the data is greater than the living area calculated.
# The GrLivArea does not seem to include the basement area, because adding that would produce a sum
# greater than the general living area in the data. 
# There may be some data quality issues, so we'll use solely GrLivArea to get rid of the inconsistency 

data = data.drop(['1stFlrSF', '2ndFlrSF'], axis=1)

###############################################################################################################
# Baths: Add bathrooms together
data['BsmtBaths'] = data['BsmtFullBath']+data['BsmtHalfBath']
data['Baths'] = data['FullBath']+data['HalfBath']

data = data.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)

###############################################################################################################
# Porch Area - how do each of the porch areas vary with Sale Price?
# From the graphs, it looks like the more proches of each type below that the property has, 
# the higher the Sale Price
fig = plt.figure(figsize=(5, 25))

axplot1 = fig.add_subplot(5, 1, 1)
axplot1.scatter(data['WoodDeckSF'], data['SalePrice'])
axplot1.set_xlabel('Wood Deck')
axplot1.set_ylabel('Sale Price')

axplot2 = fig.add_subplot(5, 1, 2)
axplot2.scatter(data['OpenPorchSF'], data['SalePrice'])
axplot2.set_xlabel('OpenPorchSF')
axplot2.set_ylabel('Sale Price')

axplot3 = fig.add_subplot(5, 1, 3)
axplot3.scatter(data['3SsnPorch'], data['SalePrice'])
axplot3.set_xlabel('3SsnPorch')
axplot3.set_ylabel('Sale Price')

axplot4 = fig.add_subplot(5, 1, 4)
axplot4.scatter(data['ScreenPorch'], data['SalePrice'])
axplot4.set_xlabel('ScreenPorch')
axplot4.set_ylabel('Sale Price')

axplot5 = fig.add_subplot(5, 1, 5)
axplot5.scatter(data['EnclosedPorch'], data['SalePrice'])
axplot5.set_xlabel('EnclosedPorch')
axplot5.set_ylabel('Sale Price')

# Create variable just indicating whether the property has a proch or not
# Create variable 
data['zeros'] = 0
data['HasPorch'] = 0
data.loc[((data['WoodDeckSF']>data['zeros']) | (data['OpenPorchSF']>data['zeros']) | (data['3SsnPorch']>data['zeros']) | 
        (data['ScreenPorch']>data['zeros']) | (data['EnclosedPorch']>data['zeros'])), 'HasPorch'] = 1  

data['TotalPorchArea'] = data['WoodDeckSF'] + data['OpenPorchSF'] + data['3SsnPorch'] + data['ScreenPorch'] + data['EnclosedPorch']

###############################################################################################################
# Drop unneeded variables

data = data.drop(['BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea2', 'zeros', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'EnclosedPorch'], 
    axis = 1)

# Write out cleaned data
data.to_csv('data_cleaned.csv')








