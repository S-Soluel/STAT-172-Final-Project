## Sam Solheim, Madelyn Osten, Joel Aguirre
## STAT 172 Final Project

rm(list=ls())

library(rpart) # for fitting classification trees
library(rpart.plot) # for plotting trees
library(ggplot2) # for professional exploratory graphics
# there's also 'plotly' but we won't be using this
library(pROC) # for ROC curves
library(RColorBrewer)
# install.packages("randomForest")
library(randomForest)

cats <- read.csv("aac_shelter_cat_outcome_eng.csv", stringsAsFactors = TRUE)

# Read in the data set and then look into the summary and 'strength' of 
# variables in the data set. Make sure that variables appear to have been written 
# in correctly
summary(cats)
str(cats)

summary(cats$outcome_type)
#            Adoption     Died             Disposal      Euthanasia 
# 3           12732       403              16            1452 
# Missing       Return to Owner   Rto-Adopt    Transfer 
# 28            1431              33           13323 

# The following creates a binary version of our Y variable of interest, 
# outcome_type. For our research, there are three separate cases that would
# go into our positive outcome, and these are "Adoption", "Return to Owner", and
# "Rto-Adopt". 
# In order to accomplish this, we will use several ifelse statements, separated
# by '|' which is the equivalent of the OR operator in R. 

cats$outcome_bin <- as.factor(ifelse(cats$outcome_type == "Adoption" |
                                       cats$outcome_type == "Return to Owner" | 
                                       cats$outcome_type == "Rto-Adopt", 1, 0))
summary(cats$outcome_bin)
# Calculate the total number of cats in dataset that were adopted, returned to
# owner, or Rto-adopt. The number that was calculated matched the number
# of 1's in cats$outcome_bin. 
(12732+1431+33)

# Binary Y variable has been created, now we can go into the next steps. 

