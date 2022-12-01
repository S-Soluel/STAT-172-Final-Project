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






