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

###### DATA PREPARATION FOR RANDOM FOREST ---------
# Before fitting a random forest, we want to create a subset of our data that does not
# include variables such as "age_upon_outcome" because we do not want to utilize
# variables that went into the creation of some of our newer X variables. 
# The data given from these two variables will be very highly correlated, which 
# could mess with our x variable selection. 

# Additionally, the variable "animal_type" is useless in the case of this database, 
# as ALL animals observed were cats. The "age_group" variable provided by the 
# dataset will be replaced as well, as there is no context of the scale, the age groups given additionally do not make sense,
# and we also created our own variable for age groups. 

# Additionally removed: outcome_age_.days. and outcome_age_.years.
#                       do not fully make sense in this context, and we have a
#                       variable that already deals with this. 
# The count variable has "1" noted for every single value, therefore this will not be
# useful in creating a model, and will be removed prior to fitting of a random forest. 
summary(cats)
str(cats) # Prior creating a subset, we have 41 variables. 
cats <- subset(cats, select = -c(age_upon_outcome, animal_id, animal_type, 
                                 date_of_birth, datetime, monthyear, 
                                 outcome_subtype, outcome_type, sex_upon_outcome, 
                                 count, sex_age_outcome, age_group, dob_monthyear,
                                 outcome_age_.days., outcome_age_.years.))
# Check that variables subset out of dataset are correctly removed, and 
# that no issues have happened. 
summary(cats)
str(cats)
# We took out 15 variables, and now it shows the data set has 26 variables. 
# This checks out.

# TROUBLESHOOTING IN TREE FITTING

# In attempting to fit our base forest, we encountered an error that randomForest.default()
# isn't able to handle categorical predictors with more than 53 categories, 
# so several additional variables will be removed before attempting again. 
cats$name <- factor(cats$name)
str(cats$name) # There are 7410 levels in name, this will need to be removed for certain
# Variables to be removed are name, color, and breed as they all have 53+ levels
cats <- subset(cats, select = -c(name, color, breed))
str(cats) # Checking that everything dropped as expected, dropping 3 additional
# variables leaves us at 23 which checks out

# Attempting to fit our base forest using this still gives us an error about 
# not being able to create this with 53+ categorical levels, must mean more than
# 53 total. Going to create one last subset which will include our binary y variable
# and 9 possible x variables we are interested in looking into. 

cats <- subset(cats, select = c(outcome_bin, sex, Spay.Neuter, outcome_weekday,
                                  outcome_hour, cfa_breed, domestic_breed, 
                                  season_outcome, name_length, outcome_age_group))
summary(cats)
str(cats)
# Double checking that everything worked properly in subsetting, then
# can finally get back to tree fitting and interpretations

## START OF ADDITIONAL CODE FROM 12/3/2022 ##

### TREE FITTING AND INTERPRETATION --------------

# Set the seed and in order to create training and testing data frames
RNGkind(sample.kind = "default")
set.seed(14038450) # picked arbitrary number

train.idx <- sample(x = 1:nrow(cats), size = .8*nrow(cats))
# create training data
train.df <- cats[train.idx, ]

# create testing data
test.df <- cats[-train.idx, ]
str(train.df) # 23536 observations in training data set
str(test.df) # # 5885 observations in testing data set
# totals up to 29421 observations -- all accounted for!

# set seed again before fitting forest
set.seed(14038450)

# fitting our baseline forest
baseforest <- randomForest(outcome_bin ~., 
                           # including all variables left in our subset of cats
                           # to see which are important to our interpretations
                         data = train.df, #TRAINING DATA
                         ntree = 1000, # this is our B
                         mtry = 3, # number of x variables to sample
                         # choose mtry -> sqrt(9) = 3
                         importance = TRUE)

baseforest
# Looking at the results of our base random forest
# Call:
#   randomForest(formula = outcome_bin ~ ., data = train.df, ntree = 1000,      mtry = 3, importance = TRUE) 
# Type of random forest: classification
# Number of trees: 1000
# No. of variables tried at each split: 3
# 
# OOB estimate of  error rate: 12.11%
# Confusion matrix:
#   No   Yes class.error
# No  10454  1712    0.140720
# Yes  1138 10232    0.100088
# 
# Calculating the OOB error rate in forest
mean(predict(baseforest) != train.df$outcome_bin)
# 0.1210911
# The OOB error for our base forest is about 12.11%


# TUNING FOREST ------------
# Now go ahead and tune forest
plot(baseforest) # can plot base forest to see general trend of error rate,
                # by the number of trees
# create sequence of m values we want to try, 
# ranging this from 1 to 9 possible variables being put in each tree in the forest.

mtry <- c(1:9)

# make room for m and oob error (empty data frame)
keeps <- data.frame(m=rep(NA, length(mtry), OOB_error_rate = rep(NA, length(mtry))))

for(idx in 1:length(mtry)){
  print(paste0("Fitting m = ",mtry[idx]))
  tempforest <- randomForest(outcome_bin ~., 
                             data = train.df,
                             ntree = 1000,
                             mtry = mtry[idx]) # mtry is varying
  
  # record OOB error, corresponding to mtry for each forest fit
  keeps[idx, "m"] <- mtry[idx]
  keeps[idx, "OOB_error_rate"] <- mean(predict(tempforest) != train.df$outcome_bin)
  
}
keeps # Show results for each value of m and the corresponding OOB error rate
# m OOB_error_rate
# 1 1      0.1403807
# 2 2      0.1269120
# 3 3      0.1214735
# 4 4      0.1238953
# 5 5      0.1293763
# 6 6      0.1326054
# 7 7      0.1343899
# 8 8      0.1358345
# 9 9      0.1362593

# Create a plot showing these values of m compared to OOB error rates
# plot the OOB error rate vs m
ggplot(data = keeps) +
  geom_line(aes(x = m, y = OOB_error_rate)) +
  labs(x = "'M' Variables Sampled per Forest", y = "Estimated OOB_error_rate") +
  ggtitle("Estimated OOB Error Rate by Number of Sampled Variables")

# Looking both at the chart and the graphical charting of
# our possible values of the OOB error rate compared to the number of variables sampled per 
# forest, the lowest OOB error rate we can get occurs when we use
# m = 3, and that error rate is around 12.14% (0.1214735)

# Looking at this, it appears that our initial base model will also be our
# final model. 

final_forest <- baseforest # storing our base forest in a new variable for the 
# forest we decided to go with
final_forest
# Call:
#   randomForest(formula = outcome_bin ~ ., data = train.df, ntree = 1000,      mtry = 3, importance = TRUE) 
# Type of random forest: classification
# Number of trees: 1000
# No. of variables tried at each split: 3
# 
# OOB estimate of  error rate: 12.11%
# Confusion matrix:
#        No   Yes class.error
# No  10454  1712    0.140720
# Yes  1138 10232    0.100088

# PREDICTIONS BASED ON FINAL FOREST --------
# Create ROC Curve
# Assume positive event is "Yes" as we are wanting to determine what
# leads to a cat being adopted or returned to their owner!
pi_hat <- predict(final_forest, test.df, type = "prob")[, "Yes"]

rocCurve <- roc(response = test.df$outcome_bin, 
                predictor = pi_hat, 
                levels = c("No", "Yes")) # Order for levels matters
# First is negative event, then positive event
plot(rocCurve, print.thres = TRUE, print.auc = TRUE)

# Interpretations of pi*, specificity, and sensitivity
# If we set pi* = 0.587, then we can achieve a specificity of 0.889
# and a sensitivity of 0.872. 

# In other words, when a cat is adopted or returned to their owner, we correctly
# predict this outcome about 87.2% of the time. (specificity interpretation)
# Additionally, when we predict that a cat will not be adopted or returned to their owner, 
# we correctly predict this scenario around 88.9% of the time. (specificity interpretation)

# The AUC for our rocCurve is 0.949, which we think is good since
# a better AUC will be closer to 1. 

# Make predictions on our test data
# extract pi* - need for good predictions
# (that maximize sensitivity and specificity)
pi_star <- coords(rocCurve, "best", ret = "threshold")$threshold[1]
# Interpretation of pi_star
# If the forest predicts an Adoption / Return to Owner probability 
# greater than 0.5865 (pi_star), then we predict that a cat will be
# going home with someone. Otherwise, we predict they will not adopted or 
# returned to their owner. 

# The following line creates predictions that are consistent with the 
# above statements about AUC, pi_hat, specificity, and sensitivity
test.df$forest_pred <- as.factor(ifelse(pi_hat > pi_star, "Yes", "No"))


# INTERPRETATIONS UTILIZING FINAL FOREST -----------
# utilizing random forest to get a ranked list of variable importance
# to be used in our model creation

varImpPlot(final_forest, type = 1)
# all of our "MeanDecreaseAccuracy" values are positive, 
# that means each variable helped the model, though to very different degrees
# Somewhat interestingly, we can see that the length of a cat's name (name_length)
# appears to be the most important variable for prediction in our data set
# Other important variables are ranked as follows:
# Grouped closely to each other: 
  # (2) Spay.Neuter and (3) outcome_age_group
# By itself: (4) outcome_hour
# By itself: (5) outcome_weekday
# Grouped together: (6) sex and (7) season_outcome
# Last group of variables: (8) domestic_breed and (9) cfa_breed

# After gaining further context about which variables appear most important,
# we can utilize the results to create an appropriate model using logistic regression. 
# We do this because logistic regression one of its benefits over random forests is
# that we can easily make interpretations based on our results.  
# The reason we didn't skip creating a random forest is that it 
# allowed us to create predictions and determine which 
# variables are most important for a logistic regression model. 

## MODEL CREATION THROUGH LOGISTIC REGRESSION
# We chose to create a model with the Bernoulli random variable component and 
# the logit link to fit our binary response variable. 

m1 <- glm(outcome_bin ~ name_length, data = cats,
          family = binomial(link = "logit"))
AIC(m1) # 33440.12
BIC(m1) # 33456.69

# next two important are Spay.Neuter & outcome_age_group
m2 <- glm(outcome_bin ~ name_length + Spay.Neuter + outcome_age_group, 
          data = cats, family = binomial(link = "logit"))
AIC(m2) # 23206.27
BIC(m2) # 23272.59
# vast improvement from the first model, both AIC and BIC have decreased

# next important is outcome_hour
m3 <- glm(outcome_bin ~ name_length + Spay.Neuter + outcome_age_group +
            outcome_hour,
          data = cats, family = binomial(link = "logit"))
AIC(m3) # 21489.87
BIC(m3) # 21564.48

# next important is outcome_weekday
m4 <- glm(outcome_bin ~ name_length + Spay.Neuter + outcome_age_group +
            outcome_hour + outcome_weekday, 
          data = cats, family = binomial(link = "logit"))
AIC(m4) # 20974.08
BIC(m4) # 21098.42

# next two important are sex & season_outcome
m5 <- glm(outcome_bin ~ name_length + Spay.Neuter + outcome_age_group +
            outcome_hour + outcome_weekday + sex + season_outcome, 
          data = cats, family = binomial(link = "logit"))
AIC(m5) # 20883.27
BIC(m5) # 21040.77

# next two important are domestic_breed & cfa_breed
m6 <- glm(outcome_bin ~ name_length + Spay.Neuter + outcome_age_group +
            outcome_hour + outcome_weekday + sex + season_outcome +
            domestic_breed + cfa_breed, 
          data = cats, family = binomial(link = "logit"))
AIC(m6) # 20822.03
BIC(m6) # 20996.11

# our sixth and final model is our best performing, with the lowest AIC & BIC
# values, we would use this model to make our interpretations

final_model <- glm(outcome_bin ~ name_length + Spay.Neuter + outcome_age_group +
            outcome_hour + outcome_weekday + sex + season_outcome +
            domestic_breed + cfa_breed, 
          data = cats, family = binomial(link = "logit"))
summary(final_model)
