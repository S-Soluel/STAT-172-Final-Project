# STAT 172
# Final Project Codespace
# Team Members: Joel Aguirre, Madelyn Osten, Sam Solheim

# This is the main coding environment that will be utilized during the course
# of this project. Whenever the group commits changes to our code, any changes
# made will be implemented here for our final results. 

rm(list=ls())

library(rpart) # for fitting classification trees
library(rpart.plot) # for plotting trees
library(ggplot2) # for professional exploratory graphics
library(pROC) # for ROC curves
library(RColorBrewer)
library(randomForest)
library(glmnet) # for fitting lasso, ridge regressions (GLMs)

# READING IN OUR DATA --------------
cats <- read.csv("aac_shelter_cat_outcome_eng.csv", stringsAsFactors = TRUE)

# Read in the data set and then look into the summary and 'strength' of 
# variables in the data set. Make sure that variables appear to have been written 
# in correctly
summary(cats)
str(cats)

# CREATING BINARY Y VARIABLE --------------

table(cats$outcome_type)
#            Adoption     Died             Disposal      Euthanasia 
# 3           12732       403              16            1452 
# Missing       Return to Owner   Rto-Adopt    Transfer 
# 28            1431              33           13323 

# For our research, there are three separate cases that would
# go into our positive outcome of a cat going home with someone, 
# and these are "Adoption", "Return to Owner", and
# "Rto-Adopt". Additionally, the 3 observations for "" will be put into the 
# "Transfer" outcome, so when we create the binary variable this category can simply
# go into the "No" category. 

# Convert Y variable from factor to character var
cats$outcome_type <- as.character(cats$outcome_type)

# modify "" to be "Transfer"
cats$outcome_type[cats$outcome_type == ""] <- "Transfer"
table(cats$outcome_type)

# convert to a factor
cats$outcome_type <- as.factor(cats$outcome_type)
str(cats)
table(cats$outcome_type)

# Turning outcome_type back into a factor, order will not matter, as
# this will be turned into a binary outcome for our use-case

# The following creates the binary version of our y variable
# In order to accomplish this, we will use several ifelse statements, separated
# by '|' which is the equivalent of the OR operator in R. 

cats$outcome_bin <- as.factor(ifelse(cats$outcome_type 
                                     %in% c("Adoption", "Return to Owner",
                                            "Rto-Adopt"), "Yes", "No"))
# Verify that the binary y was properly created
# Should expect (12732+1431+33) = 14196 for "Yes",
# also expect (403+16+1452+28+13326) = 15225 for "No"

summary(cats$outcome_bin)
# The number that was calculated matched the number of 1's in cats$outcome_bin. 
# Binary Y variable has been created, now we can go into the next steps.

# CREATION OF ADDITIONAL X VARIABLES --------------

# creating a season variable using the month the cat's outcome occurred
# going by the generally agreed upon distribution of the seasons by month
cats$season_outcome <- as.character(ifelse(cats$outcome_month %in% c(12, 1, 2), "Winter", 
                            ifelse(cats$outcome_month %in% c(3, 4, 5), "Spring",
                            ifelse(cats$outcome_month %in% c(6, 7, 8), "Summer",
                            ifelse(cats$outcome_month %in% c(9, 10, 11), "Fall",
                            "N/A")))))
                                   
# check if it worked
unique(cats$season_outcome)

# change to an ordered factor
cats$season_outcome <- factor(cats$season_outcome, 
                      levels = c("Spring", "Summer", "Fall", "Winter"))
# check if it worked
table(cats$season_outcome)
str(cats)
# season_outcome was correctly marked as a factor

# creating a variable counting the number of characters in each name to 
# explore the impact of during x variable selection
str(cats$name)

#convert to a character
cats$name <- as.character(cats$name)

# create the numeric variable
cats$name_length <- as.numeric(nchar(cats$name))
# check if it worked as anticipated
unique(cats$name_length)
str(cats)

# Appears to have worked, noticed there are some cats that apparently
# don't come in with names, as there is a length 0 name. Look into this further:
table(cats$name_length)
# It appears the vast majority of cats (12774) don't have names when they come in,
# we can't really consolidate this into other groups, so it will be interesting to
# see how this affects a cat's outcome. It could possibly mean that cat wasn't anyone's pet
# before arriving at the shelter. 

# create a consolidated age variable that says what age group a cat
# is in when their outcome is reported
unique(cats$age_upon_outcome)

#convert to a character
cats$age_upon_outcome <- as.character(cats$age_upon_outcome)
# Note: Since we will be creating an age group variable, when we go ahead to x variable selection the age_upon_outcome variable
#       will be left out by utilizing a subset of our data. 


cats$outcome_age_group[cats$age_upon_outcome %in% c("2 weeks", "1 month", "3 weeks", "4 weeks", "3 days", "6 days", "1 week", "2 days", "5 days", "1 day", "1 weeks", "4 days", "0 years")] <- "0-1 months"
# we decided to make the assumption that "0 years" should go into "0-1 months", primarily because that will be our most-common group
cats$outcome_age_group[cats$age_upon_outcome %in% c("3 months", "2 months", "5 weeks")] <- "1-3 months"
cats$outcome_age_group[cats$age_upon_outcome %in% c("5 months", "4 months", "6 months")] <- "3-6 months"
cats$outcome_age_group[cats$age_upon_outcome %in% c("8 months", "10 months", "11 months", "9 months", "7 months", "1 year")] <- "6-12 months"
cats$outcome_age_group[cats$age_upon_outcome %in% c("3 years", "4 years", "2 years", "7 years", "6 years", "8 years", "10 years", "5 years", "9 years")] <- "1-10 years"
cats$outcome_age_group[cats$age_upon_outcome %in% c("15 years", "16 years", "11 years", "14 years", "13 years", "12 years", "17 years", "20 years", "18 years", "19 years", "22 years")] <- "10+ years"
table(cats$outcome_age_group)
# 6914 + 6135 + 8416 + 634 + 2686 + 4636 = 29421, all cats covered!

# convert to an ordered factor
cats$outcome_age_group <- factor(cats$outcome_age_group, 
                      levels = c("0-1 months", "1-3 months", "3-6 months", "6-12 months", "1-10 years", "10+ years"))
# Check to make sure ordering worked
table(cats$outcome_age_group)
str(cats$outcome_age_group)
# Looks all good! We can go ahead and proceed to the next stage



