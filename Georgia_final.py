#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:13:47 2023

@author: parimagphanthong
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy as pt
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
import seaborn as sns

# Set up python to display all columns when print data frame
pd.set_option('display.max_columns', None)

# IMPORT DATASET
ga = pd.read_stata('/Users/parimagphanthong/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/EconS 525 Econometrics/Project/Georgia_final')

# Clean up data

## Drop unnecessary specifications from data
specs = ['multyear','sample','serial','cbserial','hhwt','cluster','stateicp','strata','gq','pernum','perwt']
ga = ga.drop(columns=specs)

## Drop same-sex parents for simplicity
### Drop rows where any of the specified columns is NOT NaN
same_sex = ['educd_mom2','educd_pop2']
ga = ga[~ga[same_sex].notna().any(axis=1)]
### Drop same-sex column
ga = ga.drop(columns=same_sex)

## Drop sample whose educational attainment data is not available
ga = ga[ga['educ'] != 'n/a or no schooling']

## Drop sample who are not 18
ga = ga[(ga['age'] == '18')]

## Drop sample who has higher than high school,
    ##we want to exclude those who graduated early or take college classes
    ##while in hs and earned an AD with their hs diploma
ga = ga[(ga['educd'] != 'associate\'s degree, type not specified')]

## Create treatment dummy
print(ga['birthyr'].value_counts().sort_index())
ga['treatment'] = np.where(ga['birthyr'].isin([1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]), 1, 0)
### Print data to check
print(pd.crosstab(ga['year'], ga['treatment'], margins=True, margins_name='Total'))

## Create high school dummy
ga['hs'] = np.where(ga['educd'].isin
                           (['high school graduate or ged', 
                             'regular high school diploma', 
                             'some college, but less than 1 year',
                             '1 or more years of college credit, no degree',
                             '12th grade, no diploma', 
                             'ged or alternative credential']), 1, 0)
print(pd.crosstab(ga['educd'], ga['hs'], margins=True, margins_name='Total'))

## Create parents' high school dummy
ga['higher_hs_mom'] = np.where(ga['educd_mom'].isin
                           (['high school graduate or ged', 
                             'regular high school diploma', 
                             'some college, but less than 1 year',
                             '1 or more years of college credit, no degree',
                             'bachelor\'s degree', 
                             'ged or alternative credential',
                             'master\'s degree',
                             'professional degree beyond a bachelor\'s degree',
                             'doctoral degree']),1, 0)
ga['higher_hs_mom'] = ga['higher_hs_mom'].astype('category')
ga['higher_hs_pop'] = np.where(ga['educd_pop'].isin
                           (['high school graduate or ged', 
                             'regular high school diploma', 
                             'some college, but less than 1 year',
                             '1 or more years of college credit, no degree',
                             'bachelor\'s degree', 
                             'ged or alternative credential',
                             'master\'s degree',
                             'professional degree beyond a bachelor\'s degree',
                             'doctoral degree']),1, 0)
ga['higher_hs_pop'] = ga['higher_hs_pop'].astype('category')

## Create dummy for food stamp
ga['foodstmp'] = ga['foodstmp'].replace({'yes': 1, 'no': 0})
ga['foodstmp'] = ga['foodstmp'].astype('category')
### Print to check
print(ga['foodstmp'].value_counts())

## Create dummy for sex
ga['female'] = (ga['sex'] == 'female').astype(int)
ga['female'] = ga['female'].astype('category')
### Print cross tab to check
print(pd.crosstab(ga['sex'],ga['female'], margins = True, margins_name='Total'))

## Create dummy for race
ga['black'] = (ga['race'] == 'black/african american').astype(int)
ga['black'] = ga['black'].astype('category')
### Print cross tab to check
print(pd.crosstab(ga['black'],ga['race'], margins = True, margins_name='Total'))
print(pd.crosstab(ga['hispan'],ga['hisp'], margins = True, margins_name='Total'))

## Creae dummy for hispanic
print(ga['hispan'].value_counts())
ga['hisp'] = np.where(ga['hispan'].isin(['mexican','other','puerto rican','cuban']),1,0)
ga['hisp'] = ga['hisp'].astype('category')

## Replace sibling numbers
ga['nsibs'] = ga['nsibs'].replace({'0 siblings': 0, 
                                   '1 sibling': 1,
                                   '2 siblings': 2,
                                   '3 siblings': 3,
                                   '4 siblings': 4,
                                   '5 siblings': 5,
                                   '6 siblings': 6,
                                   '7 siblings': 7,
                                   '8 siblings': 9,
                                   '9 or more siblings': 9,})
ga['nsibs'] = ga['nsibs'].astype(int)
### Print to check
print(ga['nsibs'].value_counts())

## Print data to check
print(f'ga.head(): \n{ga.head()}\n')

# Create a new data frame for regression
selected_columns = ['year','foodstmp','nsibs','age','birthyr','treatment','hs','higher_hs_mom','higher_hs_pop','female','black','hisp']
ga_cleaned = ga[selected_columns]

## Find summary statistics
data_summary = ga[['year','foodstmp','nsibs','age','birthyr','treatment','hs','higher_hs_mom','higher_hs_pop','female','black','hisp']].describe()
print(f'Data Summary:\n{data_summary}\n')

## Frequency tables
print((ga_cleaned['black'] == 1).mean() * 100)
print((ga_cleaned['female'] == 1).mean() * 100)
print((ga_cleaned['hisp'] == 1).mean() * 100)
print((ga_cleaned['foodstmp'] == 1).mean() * 100)
print((ga_cleaned['higher_hs_mom'] == 1).mean() * 100)
print((ga_cleaned['higher_hs_pop'] == 1).mean() * 100)
print((ga_cleaned['treatment'] == 1).mean() * 100)

## Find the correlation coeficients for all data
corr_coef = pd.DataFrame(ga_cleaned.corr())
print(f'Correlation Coefficients:\n{corr_coef}\n')
### Export to excel
corr_coef.to_excel('/Users/parimagphanthong/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/EconS 525 Econometrics/Project/Book2.xlsx', index=False, engine='openpyxl')

## Check for Heteroskedasticity
### Estimate probit model
reg_probit = smf.probit(formula = 'hs ~ treatment + nsibs + foodstmp + higher_hs_mom + higher_hs_pop + female + black + hisp', data = ga_cleaned)
results_probit = reg_probit.fit()
print(f'probit Results:\n{results_probit.summary()}\n')

### BP test
y, X = pt.dmatrices('hs ~ treatment + nsibs + foodstmp + higher_hs_mom + higher_hs_pop + female + black + hisp', data = ga_cleaned, return_type = 'dataframe')
result_bp = sm.stats.diagnostic.het_breuschpagan(results_probit.resid_dev,results_probit.model.exog)
bp_statistics = result_bp
bp_pval = result_bp
print(bp_statistics)
print(bp_pval)

# Linear Regression
reg_lin = smf.ols(formula = 'hs ~ treatment + nsibs + foodstmp + higher_hs_mom + higher_hs_pop + female + black + hisp', data = ga_cleaned, cov_type = 'HC3')
results_lin = reg_lin.fit(cov_type='HC3')
print(f'Linear Regression:\n{results_lin.summary()}\n')

# Probit Regression
reg_probit_robust = smf.probit(formula = 'hs ~ treatment + nsibs + foodstmp + higher_hs_mom + higher_hs_pop + female + black + hisp', data = ga_cleaned, cov_type = 'HC3')
results_probit_robust = reg_probit_robust.fit(cov_type='HC3')
print(f'probit Results Robust:\n{results_probit_robust.summary()}\n')
## Obtain resid
residuals_with_instruments = results_probit_robust.resid_dev

## Confusion table
print('Confusion Matrix for the probit Model [true neg 0 false pos1] \[false neg 0 true pos 1]')
confusion_probit = results_probit_robust.pred_table(threshold=0.5)
print(f'confusion_probit:\n {confusion_probit}\n')
## APE
### automatic average partial effects:
coef_names = np.array(results_lin.model.exog_names)
coef_names = np.delete(coef_names, 0) # drop Intercept
APE_probit_autom = results_probit.get_margeff().margeff
table_auto = pd.DataFrame({'coef_names': coef_names,'APE_probit_autom':np.round(APE_probit_autom, 4)})
print(f'table_auto: \n{table_auto}\n')
table_auto.to_excel('/Users/parimagphanthong/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/EconS 525 Econometrics/Project/Book2.xlsx', index=False, engine='openpyxl')

## Check for endogeneity
### Model without instruments
reg_probit_robust_no = smf.probit(formula = 'hs ~ treatment + nsibs + foodstmp + female + black + hisp', data = ga_cleaned, cov_type = 'HC3')
results_probit_robust_no = reg_probit_robust_no.fit(cov_type='HC3')
print(f'probit Results Robust:\n{results_probit_robust_no.summary()}\n')
#### Obtain resid
residuals_without_instruments = results_probit_robust_no.resid_dev

# Perform Durbin-Wu-Hausman test
test_statistic = np.dot(residuals_with_instruments, residuals_with_instruments) / np.dot(residuals_without_instruments, residuals_without_instruments)

# Calculate the p-value using the chi-squared distribution
degrees_of_freedom = X.shape[1]
p_value = 1 - chi2.cdf(test_statistic, degrees_of_freedom)

# Print test results
print(f"Durbin-Wu-Hausman Test Statistic: {test_statistic}")
print(f"P-value: {p_value}")

# Robustness Test
## No parents
reg_probit_no_parents = smf.probit(formula = 'hs ~ treatment + nsibs + foodstmp + female + black + hisp', data = ga_cleaned, cov_type = 'HC3')
results_probit_no_parents = reg_probit_no_parents.fit(cov_type='HC3')
print(f'probit Results Robust No Parents:\n{results_probit_no_parents.summary()}\n')
## No Siblings
reg_probit_no_sibs = smf.probit(formula = 'hs ~ treatment + foodstmp + higher_hs_mom + higher_hs_pop + female + black + hisp', data = ga_cleaned, cov_type = 'HC3')
results_probit_no_sibs = reg_probit_no_sibs.fit(cov_type='HC3')
print(f'probit Results Robust No Siblings:\n{results_probit_no_sibs.summary()}\n')
## No parents no sibs
reg_probit_none = smf.probit(formula = 'hs ~ treatment + foodstmp + female + black + hisp', data = ga_cleaned, cov_type = 'HC3')
results_probit_none = reg_probit_none.fit(cov_type='HC3')
print(f'probit Results Robust No Siblings:\n{results_probit_none.summary()}\n')
