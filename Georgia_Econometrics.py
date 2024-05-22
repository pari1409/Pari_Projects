#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:35:51 2023

@author: parimagphanthong
"""

import pandas as pd
import numpy as np

# IMPORT DATASET
ga_test = pd.read_stata('/Users/p.magphanthong/OneDrive - Washington State University (email.wsu.edu)/EconS 525 Econometrics/Project/Georgia.dta copy')
ga = pd.read_stata('/Users/p.magphanthong/OneDrive - Washington State University (email.wsu.edu)/EconS 525 Econometrics/Project/Georgia.dta copy')
## Explore Dataset
# look at imported data
print(f'ga.head(): \n{ga.head()}\n')
print(f'ga.tail(): \n{ga.tail()}\n')

# CLEAN UP DATA
## General clean up of unnecessary data
clean_up = ['multyear','sample','serial','cbserial','hhwt','cluster','stateicp','countyicp','strata','gq','pernum','perwt','birthqtr','educ','occ2010','presgl','race_mom','race_pop','hispan_mom','hispan_pop','educ_mom','educ_pop','empstat_mom','empstat_pop','empstatd_mom','empstatd_pop']
ga = ga.drop(columns=clean_up)

## Drop same-sex parents for simplicity
### Drop rows where any of the specified columns is NOT NaN
same_sex = ['educd_mom2', 'educ_mom2', 'educd_pop2', 'educ_pop2', 'incwage_mom2', 'incwage_pop2',
                    'poverty_mom2', 'poverty_pop2', 'race_mom2', 'race_pop2', 'empstat_mom2', 'empstat_pop2',
                    'empstatd_mom2', 'empstatd_pop2', 'hispan_mom2', 'hispan_pop2']
ga = ga[~ga[same_sex].notna().any(axis=1)]
### Drop same-sex column
ga = ga.drop(columns=same_sex)
### Drop age is not numerical
ga['age'] = pd.to_numeric(ga['age'], errors='coerce')
ga = ga.dropna(subset=['age'])



## Make dummies for educational attainment detail
### Create dummies for 'educd' column
educd_dummies = pd.get_dummies(ga['educd'], prefix='educd_dummy')
educd_dummies.columns = [f'educd_dummy_{i}' for i in range(1, educd_dummies.shape[1] + 1)]
### Create a mapping for renaming
dummy_rename_map = {
    f'educd_dummy_{i}': label
    for i, label in enumerate(['na', 'no_schooling', 'nurs_4', 'nurs_pre', 'kin', 'grade1', 'grade2', 'grade3', 'grade4', 'grade5_6',
                               'grade5', 'grade6', 'grade7_8', 'grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12_nodip',
                               'dip_ged', 'reg_high', 'ged', 'some_college', 'college_nodeg', 'associate', 'bach', 'master', 'prof', 'doct'], start=1)
}
#### Rename the dummy columns
educd_dummies.rename(columns=dummy_rename_map, inplace=True)
### Concatenate the dummy columns with the original DataFrame
ga = pd.concat([ga, educd_dummies], axis=1)
### Drop the original 'educd' column
ga.drop(columns=['educd'], inplace=True)

## Make dummies for parents education
### Create dummies for 'educd_mom'
#### show all category for educd_mom
print(ga['educd_mom'].value_counts())
#### Make dummy for mom no high school
ga['mom_nohs'] = 'False'
nohs_conditions_mom = ga['educd_mom'].isin(['grade 11','grade 10','grade 9','12th grade, no diploma','grade 6','grade 8','grade 7 or 8','grade 7','grade 5 or 6','grade 5','grade 3','nursery school to grade 4','grade 4','grade 2','grade 1','kindergarten','nursery school, preschool','no schooling completed'])
ga.loc[nohs_conditions_mom, 'mom_nohs'] = 'True'
#### Make dummy for mom high school
ga['mom_hs'] = 'False'
hs_conditions_mom = ga['educd_mom'].isin(['regular high school deploma','high school graduate or ged','ged or alternative credential'])
ga.loc[hs_conditions_mom, 'mom_hs'] = 'True'
#### Make dummy for some college (including Associate's degree)
ga['mom_some_college'] = 'False'
some_college_conditions_mom = ga['educd_mom'].isin(['1 or more years of college credit, no degree','associate\'s degree, type not specified','some college, but less than 1 year'])
ga.loc[some_college_conditions_mom, 'mom_highered'] = 'True'
#### Make dummy for mom bachelor's degree
ga['mom_bach'] = 'False'
ga.loc[ga['educd_mom'] == 'bachelor\'s degree', 'mom_bach'] = 'True'
#### Make dummy for mom advanced degree (beyond bachelor)
ga['mom_highered'] = 'False'
higher_edu_conditions_mom = ga['educd_mom'].isin(['professional degree beyond a bachelor\'s degree', 'doctoral degree', 'master\'s degree'])
ga.loc[higher_edu_conditions_mom, 'mom_highered'] = 'True'

### Create dummies for educs_pop     
#### Make dummy for pop no high school
ga['pop_nohs'] = 'False'
nohs_conditions_pop = ga['educd_pop'].isin(['grade 11','grade 10','grade 9','12th grade, no diploma','grade 6','grade 8','grade 7 or 8','grade 7','grade 5 or 6','grade 5','grade 3','nursery school to grade 4','grade 4','grade 2','grade 1','kindergarten','nursery school, preschool','no schooling completed'])
ga.loc[nohs_conditions_pop, 'pop_nohs'] = 'True'
#### Make dummy for pop high school
ga['pop_hs'] = 'False'
hs_conditions_pop = ga['educd_pop'].isin(['regular high school deploma','high school graduate or ged','ged or alternative credential'])
ga.loc[hs_conditions_pop, 'pop_hs'] = 'True'
#### Make dummy for some college (including Associate's degree)
ga['pop_some_college'] = 'False'
some_college_conditions_pop = ga['educd_pop'].isin(['1 or more years of college credit, no degree','associate\'s degree, type not specified','some college, but less than 1 year'])
ga.loc[some_college_conditions_pop, 'pop_highered'] = 'True'
#### Make dummy for pop bachelor's degree
ga['pop_bach'] = 'False'
ga.loc[ga['educd_pop'] == 'bachelor\'s degree', 'pop_bach'] = 'True'
#### Make dummy for pop advanced degree (beyond bachelor)
ga['pop_highered'] = 'False'
higher_edu_conditions_pop = ga['educd_pop'].isin(['professional degree beyond a bachelor\'s degree', 'doctoral degree', 'master\'s degree'])
ga.loc[higher_edu_conditions_pop, 'pop_highered'] = 'True'

## Make dummies for demographics and socioeconomics data
# Make dummy variable for 'female'
ga['female'] = 'False'
ga.loc[ga['sex'] == 'female', 'female'] = 'True'

# Make dummy variable for 'white'
ga['white'] = 'False'
ga.loc[ga['race'] == 'white', 'white'] = 'True'

# Make dummy variable for 'black'
ga['black'] = 'False'
ga.loc[ga['race'] == 'black/african american', 'black'] = 'True'


# Make dummy variable for 'hisp'
print(ga['hispan'].value_counts())
ga['hisp'] = 'False'
ga.loc[ga['hispand'] != 'not hispanic' , 'hisp'] = 'True'

# Make dummy variable for 'unemployed'
print(ga['empstat'].value_counts())
ga['unemployed'] = pd.NA
ga.loc[ga['empstat'] == 'employed', 'unemployed'] = 'False'
ga.loc[ga['empstat'] == 'unemployed', 'unemployed'] = 'True'

# Make dummy variable for 'poor'
ga['poor'] = pd.NA
ga.loc[ga['poverty'] <= 100, 'poor'] = 'True'
ga.loc[ga['poverty'] > 100, 'poor'] = 'False'

# Make dummy variable for 'metro_dummy'
print(ga['metro'].value_counts())
ga['metro_dummy'] = 'False'
ga.loc[ga['metro'] == 'in metropolitan area: in central/principal city', 'metro_dummy'] = 'True'




# Assuming 'ga' is your DataFrame

# Make dummies for attend pre-k
ga['upk95'] = 0
ga.loc[(ga['year'] == 2013) & (ga['age'] == 18), 'upk95'] = 1
ga.loc[(ga['year'] == 2018) & (ga['age'] == 23), 'upk95'] = 1
ga.loc[(ga['year'] == 2020) & (ga['age'] == 25), 'upk95'] = 1

ga['upk96'] = 0
ga.loc[(ga['year'] == 2014) & (ga['age'] == 18), 'upk96'] = 1
ga.loc[(ga['year'] == 2019) & (ga['age'] == 23), 'upk96'] = 1
ga.loc[(ga['year'] == 2021) & (ga['age'] == 25), 'upk96'] = 1

ga['upk97'] = 0
ga.loc[(ga['year'] == 2015) & (ga['age'] == 18), 'upk97'] = 1
ga.loc[(ga['year'] == 2020) & (ga['age'] == 23), 'upk97'] = 1

ga['upk98'] = 0
ga.loc[(ga['year'] == 2016) & (ga['age'] == 18), 'upk98'] = 1
ga.loc[(ga['year'] == 2021) & (ga['age'] == 23), 'upk98'] = 1

ga['upk99'] = 0
ga.loc[(ga['year'] == 2017) & (ga['age'] == 18), 'upk99'] = 1

ga['upk00'] = 0
ga.loc[(ga['year'] == 2018) & (ga['age'] == 18), 'upk00'] = 1

ga['upk01'] = 0
ga.loc[(ga['year'] == 2019) & (ga['age'] == 18), 'upk01'] = 1

ga['upk02'] = 0
ga.loc[(ga['year'] == 2020) & (ga['age'] == 18), 'upk02'] = 1

ga['upk03'] = 0
ga.loc[(ga['year'] == 2021) & (ga['age'] == 18), 'upk03'] = 1

# Make dummy for attended pre-k and age for 18 years old
ga['upk95_18'] = 0
ga.loc[(ga['year'] == 2013) & (ga['age'] == 18), 'upk95_18'] = 1

ga['upk96_18'] = 0
ga.loc[(ga['year'] == 2014) & (ga['age'] == 18), 'upk96_18'] = 1

ga['upk97_18'] = 0
ga.loc[(ga['year'] == 2015) & (ga['age'] == 18), 'upk97_18'] = 1

ga['upk98_18'] = 0
ga.loc[(ga['year'] == 2016) & (ga['age'] == 18), 'upk98_18'] = 1

ga['upk99_18'] = 0
ga.loc[(ga['year'] == 2017) & (ga['age'] == 18), 'upk99_18'] = 1

ga['upk00_18'] = 0
ga.loc[(ga['year'] == 2018) & (ga['age'] == 18), 'upk00_18'] = 1

ga['upk01_18'] = 0
ga.loc[(ga['year'] == 2019) & (ga['age'] == 18), 'upk01_18'] = 1

ga['upk02_18'] = 0
ga.loc[(ga['year'] == 2020) & (ga['age'] == 18), 'upk02_18'] = 1

ga['upk03_18'] = 0
ga.loc[(ga['year'] == 2021) & (ga['age'] == 18), 'upk03_18'] = 1

# Make dummy for 23 years old
ga['upk95_23'] = 0
ga.loc[(ga['year'] == 2018) & (ga['age'] == 23), 'upk95_23'] = 1

ga['upk96_23'] = 0
ga.loc[(ga['year'] == 2019) & (ga['age'] == 23), 'upk96_23'] = 1

ga['upk97_23'] = 0
ga.loc[(ga['year'] == 2020) & (ga['age'] == 23), 'upk97_23'] = 1

ga['upk98_23'] = 0
ga.loc[(ga['year'] == 2021) & (ga['age'] == 23), 'upk98_23'] = 1

# Make dummy for 25 years old
ga['upk95_25'] = 0
ga.loc[(ga['year'] == 2020) & (ga['age'] == 25), 'upk95_25'] = 1

ga['upk96_25'] = 0
ga.loc[(ga['year'] == 2021) & (ga['age'] == 25), 'upk96_25'] = 1

# Make a upk eligibility year
ga['upk_year'] = ga[['upk95', 'upk96', 'upk97', 'upk98', 'upk99', 'upk00', 'upk01', 'upk02', 'upk03']].idxmax(axis=1).str.extract(r'(\d+)').astype(float)

# Make 18 y/o treatment and not treatment
ga['treatment_18'] = 0
ga.loc[(ga['year'].astype(int) >= 2008) & (ga['year'].astype(int) <= 2012) & (ga['age'] == 18), 'treatment_18'] = 1

# Make 23 y/o treatment
ga['treatment_23'] = 0
ga.loc[(ga['year'].astype(int) >= 2013) & (ga['year'].astype(int) <= 2017) & (ga['age'] == 23), 'treatment_23'] = 1

# Make 25 y/o treatment
ga['treatment_25'] = 0
ga.loc[(ga['year'].astype(int) >= 2018) & (ga['year'].astype(int) <= 2019) & (ga['age'] == 25), 'treatment_25'] = 1


# Display the DataFrame
print(ga['year'].value_counts())

pd.set_option('display.max_rows', None)
birth_year_counts = ga['birthyr'].value_counts()
birth_year_counts_sorted = birth_year_counts.sort_index()
print(birth_year_counts_sorted)
