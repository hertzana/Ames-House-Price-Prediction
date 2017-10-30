# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 18:08:37 2017

@author: hanzhu
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
import os
import math
from scipy import stats
#os.chdir("C:\Users\hanzhu\Documents\DAT210x-master\Ames House Prices")
os.chdir("C:\\Users\\hanzhu\\Documents\\DAT210x-master\\Ames House Prices")

from sklearn.metrics import r2_score

# Adjust screen output size
#pd.util.terminal.get_terminal_size() # get current size
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)


data = pd.read_csv('train.csv')

data_org = pd.DataFrame(data)

data = data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)


###############################################################################################################
### Dealing with Missing Data #################################################################################
###############################################################################################################

# Find other missing values - can we do anything about them without dropping the rows of data?
MissingValues = {}

for i in data.columns:
    MissingValues[i]=len(data[data[i].isnull()]==True)

###############################################################################################################
# Electrical:
#Electrical: Electrical system
#
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
# much as this is only one value. We may not want to discard the row of data since only "Electrical" is missing.)

# Drop data row with missing electrical value
#data[data['Electrical'].isnull()]
data = data.drop(data.index[1379])
#data = data[data['Electrical'].isnull()==False]

###############################################################################################################
# LotFrontage: Linear feet of street connected to property
# For LotFrontage, we will assume that if it is NaN, then there is 0 feet of property connected to the street
data['LotFrontage'].fillna(0, inplace=True)

###############################################################################################################
# MasVnrType: Masonry veneer type
# Has 8 missing values. For rows where veneer type is NaN, is Veneer Area missing also? If so, then we can 
# Fill in missing values for Masonry veneer type as None. 
data[data['MasVnrType'].isnull()==True]['MasVnrArea']
data[data['MasVnrType']=='None']['MasVnrArea'].value_counts(dropna=False)
# Reveals that there are data quality issues
#0.0      859
#1.0        2
#312.0      1
#344.0      1
#288.0      1

#Since "Make assumption that it is none for NaN
data['MasVnrType'].fillna("None", inplace=True)

###############################################################################################################
# MasVnrType: Masonry veneer type
# Has 8 missing values
data['MasVnrArea'].fillna(0, inplace=True)


###############################################################################################################
# Basement:
# Here are all the basement-related variables: BsmtCond, BsmtExposure, BsmtFinSF1, BsmtFinSF2, 
# BsmtFinType1, BsmtFinType2, BsmtFullBath, BsmtHalfBath, BsmtQual, BsmtUnfSF

# Examine Bsmt data - some houses clearly don't have basements - why are there not the same number of NA's for 
# each basement variable?

data[data['BsmtExposure'].isnull()][['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 
     'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF']]

# Results make sense - # of baths is 0 and sq ft is 0 for houses without basements

# For basements that are unfinished (BsmtUnfSF is not 0), what do the other basement data points look like?
data[data['BsmtUnfSF']>0][['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 
     'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF']]

# Create variable indicating whether the house has a basement or not.
# If BsmtCond is null, then house does not have a basement
data['HasBsmt'] = 1
    
data.loc[data['BsmtCond'].isnull(), 'HasBsmt'] = 0
         
# Row 948 has an unfinished basement with missing data for 'BsmtExposure' --> drop this row, since we can't guess
# how much exposure the data has
#data.loc[948, 'BsmtExposure'] = "No"
data[data['BsmtExposure'].isnull()][['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 
     'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF']]
data.loc[948, ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 
     'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF']]
# Give BsmtExposure the most common value
data.loc[948, 'BsmtExposure'] = "No"

data[data['BsmtFinType2'].isnull()][['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 
     'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF']]
data.loc[332, ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 
     'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF']]
# Make BsmtFinType2 Unfinished
data.loc[332, 'BsmtFinType2'] = "Unf"

# Replace "No Basement" with "NoBase"
data.loc[data['BsmtCond'].isnull(), 'BsmtCond'] = "NoBase"
data.loc[data['BsmtExposure'].isnull(), 'BsmtExposure'] = "NoBase"           
data.loc[data['BsmtFinType2'].isnull(), 'BsmtFinType2'] = "NoBase" 
data.loc[data['BsmtFinType1'].isnull(), 'BsmtFinType1'] = "NoBase"
data.loc[data['BsmtQual'].isnull(), 'BsmtQual'] = "NoBase"


###############################################################################################################
# Garage:
# It appears that 81 houses do not have garages
data[data['GarageCond'].isnull()][['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt']]
# For these 81 houses, all of the garage variables are null



# Create variable indicating whether house has garages or not
data['HasGarage'] = 1
data.loc[data['GarageCond'].isnull(), 'HasGarage'] = 0

#Replace NA with NoGarage for Garage-Related Variables
data.loc[data['GarageCond'].isnull(), 'GarageCond'] = "NoGarage"
data.loc[data['GarageFinish'].isnull(), 'GarageFinish'] = "NoGarage"  
data.loc[data['GarageQual'].isnull(), 'GarageQual'] = "NoGarage"         
data.loc[data['GarageType'].isnull(), 'GarageType'] = "NoGarage"         
data.loc[data['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 0       
         
data1 = pd.DataFrame(data) # Data with missing values taken care of

###############################################################################################################
### Convert to Categorical Variables ##########################################################################
###############################################################################################################
data = pd.get_dummies(data, columns=['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 
                                            'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                                            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                                            'Heating', 'Functional', 'GarageType', 
                                            'GarageFinish', 'PavedDrive', 'SaleType', 'MoSold', 'Electrical', 'SaleCondition'])


overallqual = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
data.ExterQual= data.ExterQual.astype("category", ordered=True, categories=overallqual).cat.codes
data.ExterCond = data.ExterCond.astype("category", ordered=True, categories=overallqual).cat.codes
data.HeatingQC = data.HeatingQC.astype("category", ordered=True, categories=overallqual).cat.codes
data.KitchenQual = data.KitchenQual.astype("category", ordered=True, categories=overallqual).cat.codes

overallqual2 = ['NoBase', 'Po', 'Fa', 'TA', 'Gd', 'Ex'] 
data.BsmtQual = data.BsmtQual.astype("category", ordered=True, categories=overallqual2).cat.codes
data.BsmtCond = data.BsmtCond.astype("category", ordered=True, categories=overallqual2).cat.codes
                                     
overallqual3 = ['NoGarage', 'Po', 'Fa', 'TA', 'Gd', 'Ex']                                      
data.GarageQual = data.GarageQual.astype("category", ordered=True, categories=overallqual3).cat.codes
data.GarageCond = data.GarageCond.astype("category", ordered=True, categories=overallqual3).cat.codes
                                                                                
    
slope = ['Gtl', 'Mod', 'Sev']
data.LandSlope = data.LandSlope.astype("category", ordered=True, categories=slope).cat.codes

exposure = ['NoBase', 'No', 'Mn', 'Av', 'Gd']
data.BsmtExposure = data.BsmtExposure.astype('category', ordered=True, categories = exposure).cat.codes
                                              
# Central Air
data['CentralAir'] = data['CentralAir'].replace(['N', 'Y'], [0, 1])
                                              
# For BasmtFinType, both Unfinished and No Basement get 0's
data['BsmtFinType1'] = data['BsmtFinType1'].replace(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NoBase'], [3, 2, 1, 2, 1, 0, 0])
data['BsmtFinType2'] = data['BsmtFinType2'].replace(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NoBase'], [3, 2, 1, 2, 1, 0, 0])

data.loc[data['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 0       

data2 = pd.DataFrame(data)
###############################################################################################################
### Creating/Adjusting Variables ##############################################################################
###############################################################################################################

###############################################################################################################
# Basement:

# Combine basement area into 1 variable
data['BsmtFinType'] = data['BsmtFinType1']+data['BsmtFinType2']

###############################################################################################################
# Living Area:
# Check that 1st Flr Living Area + 2nd Flr Living Area = GrLivingArea
#MSSubClass = [col for col in data if col.startswith('MSSubClass')]

data['GrLivArea2'] = data['1stFlrSF']+data['2ndFlrSF']
data[data['GrLivArea2']!=data['GrLivArea']][['GrLivArea2', 'GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 
    'MSSubClass_20', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 'MSSubClass_50', 'MSSubClass_120']]

data[data['GrLivArea2']!=data['GrLivArea']].shape
# 26 observations have the total General Living Area not equal to 1st Flr Area + 2nd Flr Area.
# Upon examination of these data rows, the General Living Area in the data is greater than the living area calculated.
# The GrLivArea does not seem to include the basement area, because adding that would produce a sum
# greater than the general living area in the data. 
# There may be some data quality issues, so we'll use solely GrLivArea to get rid of the inconsistency 

###############################################################################################################
# Baths:
data['BsmtBaths'] = data['BsmtFullBath']+data['BsmtHalfBath']
data['Baths'] = data['FullBath']+data['HalfBath']

###############################################################################################################
# Porch Area - how do each of the porch areas vary with Sale Price?
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

# Create variable 
data['zeros'] = 0
data['HasPorch'] = 0
data.loc[((data['WoodDeckSF']>data['zeros']) | (data['OpenPorchSF']>data['zeros']) | (data['3SsnPorch']>data['zeros']) | 
        (data['ScreenPorch']>data['zeros']) | (data['EnclosedPorch']>data['zeros'])), 'HasPorch'] = 1    
    
#zero_data = np.zeros(shape=(len(data['WoodDeckSF']), 1))
#zero_data = pd.Series([0 for x in range(len(data['WoodDeckSF']))])

data['TotalPorchArea'] = data['WoodDeckSF'] + data['OpenPorchSF'] + data['3SsnPorch'] + data['ScreenPorch'] + data['EnclosedPorch']

###############################################################################################################
# Garage
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


###############################################################################################################
# Drop these vars: 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea2', '1stFlrSF', '2ndFlrSF', 
# 'BsmtFullBath', 'BsmtHalfBath', FullBath, HalfBath, zeros, WoodDeckSF, OpenPorchSF, 3SsnPorch, ScreenPorch, EnclosedPorch

data = data.drop(['BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea2', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 
           'BsmtHalfBath', 'FullBath', 'HalfBath', 'zeros', 'WoodDeckSF', 'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'EnclosedPorch'], 
    axis = 1)

data3 = pd.DataFrame(data)
##########################################################################################################
# What to do with houses that don't have garages, porches, basements
# Garage Vars
data[data['HasGarage']==0][['HasGarage', 'GarageYrBlt',
 'GarageCars',
 'GarageArea',
 'GarageQual',
 'GarageCond',
'GarageType_2Types',
 'GarageType_Attchd',
 'GarageType_Basment',
 'GarageType_BuiltIn',
 'GarageType_CarPort',
 'GarageType_Detchd',
 'GarageType_NoGarage',
 'GarageFinish_Fin',
 'GarageFinish_NoGarage',
 'GarageFinish_RFn',
 'GarageFinish_Unf']]

# Basement Related Vars
data[data['HasBsmt']==0][['BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtUnfSF',
 'TotalBsmtSF',
 'HasBsmt',
 'BsmtFinType',
 'BsmtBaths']]

 
# Porch Related Vars
data[data['HasPorch']==0][['HasPorch',
'TotalPorchArea']]

# How can we impute GarageYrBlt? 
fig = plt.figure(figsize=(5, 5))

plt.scatter(data_org['GarageYrBlt'], data_org['SalePrice'])
plt.show()


fig = plt.figure(figsize=(5, 5))

plt.scatter(data_org['GarageYrBlt'], data_org['SalePrice'])
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
### Dealing with Skewed Data ##############################################################################
###############################################################################################################
# Drop variables that are not meaningful:
# Drop PoolArea - only 7 properties have pools
# Drop LandCountour - vast majority of properties are level
# Drop "MiscVal"
for var in data_org.columns:
    print(data_org[var].value_counts(dropna=False))



#############################################################################################################
# Create several sets of data
# One set of data just has basement, garage, and porch indicators, but not any of the porch vars
data_droppedvars = data.drop('GarageCars', axis=1)

# Second has data where all properties have a basement, garage, and porch
data_complete = data[(data['HasGarage']==1) & (data['HasBsmt']==1) & (data['HasPorch']==1)]
# THis has 1149 observations, down from 1459

# Try dropping Year_Built for garage
data_noyrblt = data.drop('GarageYrBlt', axis=1)

# What about imputing Year_Built 
data_imp = data.copy()


###############################################################################################################
### Data Standardization and Normalization ####################################################################
###############################################################################################################

######################################
#### Part 1: Normalization ###########
######################################

data_imp = data_imp.drop("Id", axis=1)
data_imp['SalePrice'] = np.log1p(data_imp['SalePrice'])

# Skewness of Sale Price Data
plt.figure(figsize=(5, 5))
plt.hist(data_imp['SalePrice'])
plt.show()

# Get all other continuous variables & calculate skew
cont_var = ['LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'MasVnrArea',
 'ExterQual',
 'ExterCond',
 'BsmtQual', # gets worse with transform
 'BsmtCond', # gets worse with transform
 'BsmtExposure', # sqrt
 'BsmtUnfSF',
 'TotalBsmtSF',
 'HeatingQC', # best not to do anything
 'LowQualFinSF', # most fall into one value
 'GrLivArea', 
 'BedroomAbvGr', # no need for transforming
 'KitchenAbvGr', # not differentiating var
 'KitchenQual',
 'TotRmsAbvGrd', # log works
 'Fireplaces',
 'GarageYrBlt',
 'GarageCars',
 'GarageArea',
 'GarageQual', # leave as is, gets worse with transform
 'GarageCond', # leave as is, gets worse with transform
 'PoolArea', # most have no pools - no transform needed
 'MiscVal', # most have no miscval
 'TotalPorchArea', 
 'BsmtBaths', # leave as is
 'Baths' # leave as is
  ]


### 2nd Pass #############################################
cont_var2 = ['LotFrontage', # unchanged
 'LotArea', # log works
 'MasVnrArea', # Log works
 'BsmtUnfSF', # Sqrt
 'TotalBsmtSF', # Sqrt
 'GrLivArea', # Log works
 'KitchenQual', # sqrt
 'BsmtExposure', # sqrt
 'TotRmsAbvGrd', #log
 'Fireplaces', #sqrt
 'GarageQual', 
 'GarageCond',
 'TotalPorchArea', # sqrt
 'ExterCond', # use log
]

# Find skew of each numerical variable
data_cont = data_imp[cont_var]
skewness = data_cont.apply(lambda x: stats.skew(x)) 
skewed_vars = skewness[abs(skewness)>0.5]
# Log Transform skewed vars
skewed_ind = skewed_vars.index
skewness_ind = skewness.index
data_trans = data_imp.copy()
data_trans[skewed_ind] = np.log1p(data_imp[skewed_ind])

# Check skewness of resulting vars
data_cont2 = data_trans[cont_var]
skewness2 = data_cont2.apply(lambda x: stats.skew(x))
#skew_vars2 = skewness2[abs(skewness2)>0.5] 
skewed_ind_vars2 = skewness2.index
# Even after taking log, more than half of the variables are skewed (19 out of 32)

# Look at distribution before and after log transform
fig = plt.figure(figsize=(10, 120))
for i in range(len(skewness_ind)):
    axplot1 = fig.add_subplot(len(skewness_ind), 2, 2*i+1)
    axplot1.hist(data_imp[skewness_ind[i]])
    axplot1.set_title(str(skewness_ind[i])+ " Original: "+ str(skewness[i]))
    axplot2 = fig.add_subplot(len(skewness_ind), 2, 2*i+2)
    axplot2.hist(data_trans[skewed_ind_vars2[i]])
    axplot2.set_title(str(skewed_ind_vars2[i])+ " Log Transformed: " + str(skewness2[i]))


    
#LotFrontage        0.268362
#LotArea           -0.136932
#LandSlope          4.291050
#OverallQual        0.215497
#OverallCond       -0.254602
#YearBuilt         -0.639784
#YearRemodAdd      -0.509545
#MasVnrArea         0.501880
#ExterQual          0.465278
#ExterCond         -0.240732
#BsmtQual          -3.608799
#BsmtCond          -5.143676
#BsmtExposure       0.573356
#BsmtUnfSF         -2.183234
#TotalBsmtSF       -5.151375
#HeatingQC         -0.882179
#CentralAir        -3.525272
#LowQualFinSF       7.449961
#GrLivArea         -0.006303
#BedroomAbvGr       0.211839
#KitchenAbvGr       3.863864
#KitchenQual        0.387691
#TotRmsAbvGrd      -0.057415
#Fireplaces         0.180655
#GarageYrBlt       -0.719173
#GarageCars        -0.341494
#GarageArea         0.179081
#GarageQual        -3.637779
#GarageCond        -3.643433
#PoolArea          14.343356
#MiscVal            5.163427
#TotalPorchArea    -1.310267
#dtype: float64

#####################################################################
#### Determine how to transform data to make normal
# BsmtUnfSF
plt.figure(figsize=(5,5))
plt.hist(data_imp['BsmtUnfSF'])
plt.show()

a = np.log10(data_imp['BsmtUnfSF']+1)
stats.skew(a)
plt.figure(figsize=(5,5))
plt.hist(a)
plt.show()

b = np.log(data_imp['BsmtUnfSF']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['BsmtUnfSF'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

# BsmtUnfSF needs to use sqrt for transformation. This results in skew of -0.24653389698892683 from 0.9

## TotalBsmtSF - take sqrt. Doesn't solve problem completely, but helps somewhat
plt.figure(figsize=(5,5))
stats.skew(data_imp['TotalBsmtSF'])
plt.hist(data_imp['TotalBsmtSF'])
plt.show()

a = np.log10(data_imp['TotalBsmtSF']+1)
stats.skew(a)
plt.figure(figsize=(5,5))
plt.hist(a)
plt.show()

b = np.log(data_imp['TotalBsmtSF']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['TotalBsmtSF'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

data_imp['ones'] = 1
d = data_imp['ones']/np.square(data_imp['TotalBsmtSF'])
stats.skew(d)
plt.figure(figsize=(5,5))
plt.hist(d)
plt.show()

stats.skew(a)
stats.skew(b)
stats.skew(c)


## LowQualFinSF - Just leave as it is. Vast majority of properties do not have low quality finished areas.
plt.figure(figsize=(5,5))
stats.skew(data_imp['LowQualFinSF'])
plt.hist(data_imp['TotalBsmtSF'])
plt.show()

a = np.log10(data_imp['LowQualFinSF']+1)
stats.skew(a)
plt.figure(figsize=(5,5))
plt.hist(a)
plt.show()


# MasVnrArea
stats.skew(data_imp['MasVnrArea'])
plt.hist(data_imp['MasVnrArea'])
plt.show()

a = np.log10(data_imp['MasVnrArea']+1)
stats.skew(a)
plt.figure(figsize=(5,5))
plt.hist(a)
plt.show()

b = np.log(data_imp['MasVnrArea']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['MasVnrArea'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(a)
stats.skew(b)
stats.skew(c)

# BedroomAbvGrd
stats.skew(data_imp['BedroomAbvGr'])
plt.hist(data_imp['BedroomAbvGr'])
plt.show()

a = np.log10(data_imp['BedroomAbvGr']+1)
stats.skew(a)
plt.figure(figsize=(5,5))
plt.hist(a)
plt.show()

b = np.log(data_imp['BedroomAbvGr']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['BedroomAbvGr'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(a)
stats.skew(b)
stats.skew(c)

# BedroomAbvGrd
stats.skew(data_imp['KitchenAbvGr'])
plt.hist(data_imp['KitchenAbvGr'])
plt.show()

b = np.log(data_imp['KitchenAbvGr']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['KitchenAbvGr'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)


# TotRmsAbvGrd
stats.skew(data_imp['TotRmsAbvGrd'])
plt.hist(data_imp['TotRmsAbvGrd'])
plt.show()

b = np.log(data_imp['TotRmsAbvGrd']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['TotRmsAbvGrd'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)


# Fireplaces
stats.skew(data_imp['Fireplaces'])
plt.hist(data_imp['Fireplaces'])
plt.show()

b = np.log(data_imp['Fireplaces']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['Fireplaces'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)

# Porch Area
stats.skew(data_imp['TotalPorchArea'])
plt.hist(data_imp['TotalPorchArea'])
plt.show()

b = np.log(data_imp['TotalPorchArea']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['TotalPorchArea'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)


# Misc Val
stats.skew(data_imp['MiscVal'])
plt.hist(data_imp['MiscVal'])
plt.show()

b = np.log(data_imp['MiscVal']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['MiscVal'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)

# BsmtQual
stats.skew(data_imp['GarageQual'])
plt.hist(data_imp['BsmtQual'])
plt.show()

b = np.log(data_imp['BsmtQual']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['BsmtQual'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)

# Exter Cond
stats.skew(data_imp['ExterCond'])
plt.hist(data_imp['ExterCond'])
plt.show()

a = np.log1p(data_imp['ExterCond'])
stats.skew(a)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

b = np.log(data_imp['ExterCond']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['ExterCond'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)

# BsmtExposure
stats.skew(data_imp['BsmtExposure'])
plt.hist(data_imp['BsmtExposure'])
plt.show()


b = np.log(data_imp['BsmtExposure']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['BsmtExposure'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)

# Heating QC
stats.skew(data_imp['HeatingQC'])
plt.hist(data_imp['HeatingQC'])
plt.show()

b = np.log(data_imp['HeatingQC']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['HeatingQC'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)


# KitchenQual
stats.skew(data_imp['KitchenQual'])
plt.hist(data_imp['KitchenQual'])
plt.show()

b = np.log(data_imp['KitchenQual']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['KitchenQual'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)

# BsmtBaths
stats.skew(data_imp['BsmtBaths'])
plt.hist(data_imp['BsmtBaths'])
plt.show()

b = np.log(data_imp['BsmtBaths']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['BsmtBaths'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)

# Baths
stats.skew(data_imp['Baths'])
plt.hist(data_imp['Baths'])
plt.show()

b = np.log(data_imp['Baths']+1)
stats.skew(b)
plt.figure(figsize=(5,5))
plt.hist(b)
plt.show()

c = np.sqrt(data_imp['Baths'])
stats.skew(c)
plt.figure(figsize=(5,5))
plt.hist(c)
plt.show()

stats.skew(b)
stats.skew(c)

cont_var2 = ['LotFrontage', # unchanged
 'LotArea', # log works
 'MasVnrArea', # Log works
 'BsmtUnfSF', # Sqrt
 'TotalBsmtSF', # Sqrt
 'GrLivArea', # Log works
 'KitchenQual', # sqrt
 'BsmtExposure', # sqrt
 'TotRmsAbvGrd', #log
 'Fireplaces', #sqrt
 'GarageQual', 
 'GarageCond',
 'TotalPorchArea', # sqrt
 'ExterCond', # use log]

log_transform = ['LotArea', 'MasVnrArea', 'GrLivArea', 'TotRmsAbvGrd', 'ExterCond']
sqrt = ['BsmtUnfSF', 'TotalBsmtSF', 'KitchenQual', 'BsmtExposure', 'Fireplaces', 'TotalPorchArea']

# Find skew of each numerical variable
#log = data_imp[log_transform]
#skewness = log.apply(lambda x: stats.skew(x))
#skew_log_ind = skewness.index
#data_trans[skew_log_ind] = np.log1p(data_trans[skew_log_ind])
#
#sqrt_vars = data_imp[sqrt]
#skew_sqrt = log.apply(lambda x: stats.skew(x))
#skew_sqrt_ind = skew_sqrt.index
#data_trans[skew_sqrt_ind] = np.sqrt(data_trans[skew_sqrt_ind])

# try looping
data_trans = data_imp.copy()
skew_loop = []
skew_trans = []
for i in data_imp[log_transform]:
    skew_loop.append(stats.skew(data_imp[i]))
    data_trans[i] = np.log1p(data_imp[i])
    skew_trans.append(stats.skew(data_trans[i]))
    
for x in data_imp[sqrt]:
    skew_loop.append(stats.skew(data_imp[x]))
    data_trans[x] = np.sqrt(data_imp[x])
    skew_trans.append(stats.skew(data_trans[x]))

# Check skew of all transformed variables
skewed_vars = log_transform + sqrt
data_skew_check = data_trans[skewed_vars]
skew_check = data_skew_check.apply(lambda x: stats.skew(x))

# Plot out graph of all cont vars
# Look at distribution before and after log transform
fig = plt.figure(figsize=(10, 120))
for i in range(len(cont_var)):
    axplot1 = fig.add_subplot(len(cont_var), 2, 2*i+1)
    axplot1.hist(data_imp[cont_var[i]])
    axplot1.set_title(str(cont_var[i])+ " Original: "+ str(stats.skew(data_imp[cont_var[i]])))
    axplot2 = fig.add_subplot(len(cont_var), 2, 2*i+2)
    axplot2.hist(data_trans[cont_var[i]])
    axplot2.set_title(str(cont_var[i])+ " Log Transformed: " + str(stats.skew(data_trans[cont_var[i]])))
    
    
# Write out data set
data_trans.to_csv("Ames_Normalized.csv", index=False)
###############################################################################################################
### Perform Regression ########################################################################################
###############################################################################################################
# import prepared dataset
data_trans = pd.read_csv('Ames_Normalized.csv')

# import methods needed for modeling
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer

# Isolate outcome variable
saleprice = data_trans['SalePrice']

# Get predictor variables
data_trans = data_trans.drop('SalePrice', axis=1)

# Split the data into training sets and test sets
x_train, x_test, y_train, y_test = train_test_split(data_trans, saleprice, test_size = 0.25, random_state = 0)

# Make RMSE the scorer
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_train(model):
    return np.sqrt(-cross_val_score(model, x_train, y_train, scoring=scorer, cv=10))

def rmse_test(model):
    return np.sqrt(-cross_val_score(model, x_test, y_test, scoring=scorer, cv=10))


#############################################################################
# Run Linear Regression without regularization

# Initialize Linear Reg
lr = LinearRegression()
lr.fit(x_train, y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

# What is RMSE for both training and test set?
print(rmse_train(lr).mean()) #0.142
print(rmse_test(lr).mean()) #0.188

# Calculate R-squared of model to see how much variance it explains
from sklearn.metrics import r2_score

r2_test = r2_score(y_test, y_test_pred) # Model explains 88% of variance within testing data
r2_train = r2_score(y_train, y_train_pred) # Model explains 95% of variance within training data

# Calculate adjusted R-squared
def adj_rsquared(rsquared, data):
    n = data.shape[0]
    p = data.shape[1]
    a = ((1-rsquared)*(n-1))/(n-p-1)
    return 1 - a

adj_rsquared(r2_test, data_trans) # 86%
adj_rsquared(r2_train, data_trans) #94%  


# Plot residuals
plt.scatter(y_train, (y_train_pred - y_train), c="blue", marker = "o", label="Training Data")
plt.scatter(y_test, (y_test_pred - y_test), c="green", marker = "o", label = "Testing Data")
plt.title("Linear Regression")
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [0, 0], c = "red")
plt.show()

# Plot predictions - how closely did predictions match actual results?
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

# Which coefficients were most important in predicting Sales Price?
coef = pd.Series(lr.coef_,index=x_test.columns).sort_values()
coef.plot(kind="bar", title = "Linear Reg Coeffs", figsize=(50, 50))



##########################################################################################
# Introducing Ridge Regularization
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(x_train, y_train)
alpha = ridge.alpha_

y_rdg_pred_train = ridge.predict(x_train)
y_rdg_pred_test = ridge.predict(x_test)

# What is error with ridge?
rmse_train(ridge).mean() #0.134
rmse_test(ridge).mean() #0.125
# So, this is a slight improvement compared to linear regression with no regularization

# What is the r2 score?
r2_rdg_train = r2_score(y_train, y_rdg_pred_train) #0.939
r2_rdg_test = r2_score(y_test, y_rdg_pred_test) #0.901
# R-2 goes down, but this is expected. 

# What is mean squared error without cross validation?
mean_squared_error(y_train, y_rdg_pred_train) # 0.0099
mean_squared_error(y_test, y_rdg_pred_test) #0.015


# Calculate adjusted rsquared
adj_rsquared(r2_rdg_train, data_trans) #92%
adj_rsquared(r2_rdg_test, data_trans) #88%
# When comparing adjusted r-squared, we see that using Ridge improves the r2 score on the test
# set

# What are the important coefficients?
coefs = pd.Series(ridge.coef_, index=x_train.columns)
import_coefs = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])
import_coefs.plot(kind='barh')

# What do residuals look like?
plt.scatter(y_rdg_pred_train, (y_rdg_pred_train - y_train), c="blue", marker="o", label = "Training data")
plt.scatter(y_rdg_pred_test, (y_rdg_pred_test - y_test), c="lightgreen", marker="o", label = "Testing data")
plt.title("Linear Regression with Ridge")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# How does prediction compare to actual data?
plt.scatter(y_rdg_pred_train, y_train, c="blue", marker="o", label="Training Data")
plt.scatter(y_rdg_pred_test, (y_test), c="lightgreen", marker="o", label = "Testing data")
plt.title("Linear Regression with Ridge")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], color = "red")
plt.show()

print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
# Picked 237 features and eliminated 8 features

#Which features were eliminated?
coefs[coefs == 0]
#Utilities_AllPub       0.0
#Utilities_NoSeWa       0.0
#Condition2_PosA        0.0
#Exterior1st_AsphShn    0.0
#Exterior1st_ImStucc    0.0
#Exterior2nd_AsphShn    0.0
#Heating_Floor          0.0
#SaleType_Con           0.0


##########################################################################################
# Introducing Lasso Regularization
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1, 3, 6, 10, 30, 60], 
                max_iter = 50000, cv = 10)

lasso.fit(x_train, y_train)
alpha_lasso = lasso.alpha_

lasso2 = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)

lasso2.fit(x_train, y_train)
alpha_lasso2 = lasso2.alpha_

y_train_lass = lasso.predict(x_train)
y_test_lass = lasso.predict(x_test)

y_train_lass2 = lasso2.predict(x_train)
y_test_lass2 = lasso2.predict(x_test)

# What is rmse for lasso?
rmse_train(lasso).mean() #0.1302
rmse_test(lasso).mean() #0.126

rmse_train(lasso2).mean() #0.13047
rmse_test(lasso2).mean() #0.1254

# So, the two lasso's do not make too much of a difference

# What is r-squared
r2_score(y_train, y_train_lass2) #94%
r2_score(y_test, y_test_lass2) #91%

# What is adjusted r-squared
adj_rsquared(r2_score(y_train, y_train_lass2), data_trans) #93%
adj_rsquared(r2_score(y_test, y_test_lass2), data_trans) #89%

# What is mean squared error?
mean_squared_error(y_train, y_train_lass2) #0.0096611083640537158
mean_squared_error(y_test, y_test_lass2) #0.014127081905170362

# What are the important coefficients?
coefs_lasso = pd.Series(lasso2.coef_, index=x_train.columns)
import_coefs_lasso = pd.concat([coefs_lasso.sort_values().head(20), coefs_lasso.sort_values().tail(20)])
import_coefs_lasso.plot(kind='barh')

# What do residuals look like?
plt.scatter(y_train_lass2, (y_train_lass2 - y_train), c="blue", marker="o", label = "Training data")
plt.scatter(y_test_lass2, (y_test_lass2 - y_test), c="lightgreen", marker="o", label = "Testing data")
plt.title("Linear Regression with Lasso")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# How does prediction compare to actual data?
plt.scatter(y_train_lass2, y_train, c="blue", marker="o", label="Training Data")
plt.scatter(y_test_lass2, y_test, c="lightgreen", marker="o", label = "Testing data")
plt.title("Linear Regression with Lasso")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], color = "red")
plt.show()

print("Lasso picked " + str(sum(coefs_lasso != 0)) + " features and eliminated the other " +  \
      str(sum(coefs_lasso == 0)) + " features")
# Picked 113 features and eliminated 132 features


# What were the 113 features picked by Lasso?
coefs_lasso_all[coefs_lasso_all != 0]

coefs_lasso_all = pd.concat([coefs_lasso.sort_values()])
coefs_lasso_all.plot(kind='barh')

# What were the features that were discarded?
coefs_lasso[coefs_lasso == 0]


#Which features were eliminated?
coefs[coefs == 0]
#Utilities_AllPub       0.0
#Utilities_NoSeWa       0.0
#Condition2_PosA        0.0
#Exterior1st_AsphShn    0.0
#Exterior1st_ImStucc    0.0
#Exterior2nd_AsphShn    0.0
#Heating_Floor          0.0
#SaleType_Con           0.0


##########################################################################################
# ElasticNet

elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)

elasticNet.fit(x_train, y_train)

alpha_elas = elasticNet.alpha_

y_train_elas = elasticNet.predict(x_train)
y_test_elas = elasticNet.predict(x_test)

# What is rmse for ElasticNet?
rmse_train(elasticNet).mean() #0.13177596290056032
rmse_test(elasticNet).mean() #0.12498927947029641


# What is r-squared
r2_score(y_train, y_train_elas) #0.93839899554839956
r2_score(y_test, y_test_elas) #0.90807240452615567

# What is adjusted r-squared
adj_rsquared(r2_score(y_train, y_train_elas), data_trans) #0.92595691303344319
adj_rsquared(r2_score(y_test, y_test_elas), data_trans) #0.88950500065880878

# What is mean squared error?
mean_squared_error(y_train, y_train_elas) #0.00992923072420707
mean_squared_error(y_test, y_test_elas) #0.014195675347252202

# What are the important coefficients?
coefs_elas = pd.Series(elasticNet.coef_, index=x_train.columns)
import_coefs_elas = pd.concat([coefs_elas.sort_values().head(20), coefs_elas.sort_values().tail(20)])
import_coefs_elas.plot(kind='barh')

# What do residuals look like?
plt.scatter(y_train_elas, (y_train_elas - y_train), c="blue", marker="o", label = "Training data")
plt.scatter(y_test_elas, (y_test_elas - y_test), c="lightgreen", marker="o", label = "Testing data")
plt.title("Linear Regression with Elastic Net")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# How does prediction compare to actual data?
plt.scatter(y_train_elas, y_train, c="blue", marker="o", label="Training Data")
plt.scatter(y_test_elas, y_test, c="lightgreen", marker="o", label = "Testing data")
plt.title("Linear Regression with Elastic Net")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], color = "red")
plt.show()

print("Elastic Net picked " + str(sum(coefs_elas != 0)) + " features and eliminated the other " +  \
      str(sum(coefs_elas == 0)) + " features")
# Picked 119 features and eliminated 126 features


# What were the 113 features picked by Lasso?
coefs_elas[coefs_elas != 0]

coefs_lasso_all = pd.concat([coefs_lasso.sort_values()])
coefs_lasso_all.plot(kind='barh')

# What were the features that were discarded?
coefs_lasso[coefs_lasso == 0]


#Which features were eliminated?
coefs[coefs == 0]
#Utilities_AllPub       0.0
#Utilities_NoSeWa       0.0
#Condition2_PosA        0.0
#Exterior1st_AsphShn    0.0
#Exterior1st_ImStucc    0.0
#Exterior2nd_AsphShn    0.0
#Heating_Floor          0.0
#SaleType_Con           0.0












