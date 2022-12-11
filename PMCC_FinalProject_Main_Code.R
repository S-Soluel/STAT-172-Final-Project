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

# VISUALIZATIONS --------------

# let's look at name_length
ggplot(data = cats) +
  geom_histogram(aes(x = name_length)) +
  labs(x = "Name Length", y = "Amount of Cats") +
  ggtitle("Distribution of Name Lengths") 
# lots of no-names here, let's approach this variable from another angle

ggplot(data = cats) + 
  geom_boxplot(aes(x = name_length))

ggplot(data = cats) + 
  geom_boxplot(aes(x = name_length, y = outcome_bin)) +
  labs(x = "Name Length", y = "Adoption Outcome") +
  ggtitle("Distribution of Name Lengths") 

# (graph included in the supplement document)
ggplot(data = cats) +
  geom_violin(aes(x = name_length, y = outcome_bin, fill = outcome_bin)) +
  # geom_jitter(aes(x = name_length, y = outcome_bin, fill = outcome_bin)) +
  labs(x = "Name Length", y = "Adoption Outcome") + 
  ggtitle("Distribution of Name Lengths vs Adoption Outcome") +
  scale_fill_manual(values = c("#6d7771", "#FFFDD0"), 
                    name = "Adoption \nOutcome")
# can see the frequencies in each graph, more no-named cats that resulted in
# no adoption as opposed to getting adopted
# cats that have more "traditional" names that fall between 4-7 letters
# were adopted at a higher frequency

# let's look at season_outcome
ggplot(data = cats) +
  geom_bar(aes(x = season_outcome))

ggplot(data = cats) +
  geom_bar(aes(x = season_outcome, fill = outcome_bin, colour = outcome_bin)) +
  scale_fill_grey("Adoption \nOutcome")

# (graph included in the supplement document)
ggplot(data = cats) +
  geom_bar(aes(x = season_outcome, fill = outcome_bin), position = "fill") +
  labs(x = "Season of the Outcome", y = "Amount of Cats") + 
  ggtitle("Distribution of Seasons vs Adoption Outcome") +
  scale_fill_manual(values = c("#6D7771", "#FFFDD0"), 
                    name = "Adoption \nOutcome")
# general trend that the further along in the year, the higher the adoption rate
# could relate to the holidays in the winter, wanting to adopt

# let's look at sex and spay.neuter
ggplot(data = cats) +
  geom_bar(aes(x = sex, fill = outcome_bin), position = "fill")

# without fill to understand frequency distributions
# (graph included in the supplement document)
ggplot(data = cats) +
  geom_bar(aes(x = outcome_age_group, fill = outcome_bin)) +
  labs(x = "Age Group", y = "Proportion") + 
  facet_wrap(~Spay.Neuter, nrow = 2, labeller = labels) +
  scale_fill_manual(values = c("#6D7771", "#FFFDD0"), 
                    name = "Adoption \nOutcome") +
  coord_flip() +
  ggtitle("Outcomes of Spay/Neuter by Age Group")
# generally a higher amount of cats that have been spayed/neutered
# furthermore, those cats are more likely to get adopted
# let's look at the proportions in a more meaningful way

labels <- as_labeller(c("No" = "Not Spayed or Neutered", "Yes" = "Spayed or Neutered"))
# with fill to understand proportions
# (graph included in the supplement document)
ggplot(data = cats) +
  geom_bar(aes(x = outcome_age_group, fill = outcome_bin), position = "fill") +
  labs(x = "Age Group", y = "Proportion") + 
  facet_wrap(~Spay.Neuter, nrow = 2, labeller = labels) +
  scale_fill_manual(values = c("#6D7771", "#FFFDD0"), 
                    name = "Adoption \nOutcome") +
  coord_flip() +
  ggtitle("Outcomes of Spay/Neuter by Age Group")
# higher proportions of adoption for cats that have spayed/neutered
# general downward trend for this subset of cats
# could imply that potential adopters do not care about spay/neutering 
# the older the cat is


#                                 Estimate  Std. Error z value Pr(>|z|) 
# (Intercept)                    -6.278657   0.415560 -15.109  < 2e-16 ***
#   name_length                   0.277979   0.005857  47.464  < 2e-16 ***
#   Spay.NeuterYes                2.733662   0.050528  54.101  < 2e-16 ***
#   outcome_age_group1-3 months   2.819162   0.061116  46.128  < 2e-16 ***
#   outcome_age_group3-6 months   1.889418   0.076400  24.731  < 2e-16 ***
#   outcome_age_group6-12 months  0.658901   0.065440  10.069  < 2e-16 ***
#   outcome_age_group1-10 years   0.698195   0.061474  11.358  < 2e-16 ***
#   outcome_age_group10+ years    0.640101   0.106874   5.989  2.11e-09 ***
#   outcome_hour                  0.187533   0.005104  36.742  < 2e-16 ***
#   outcome_weekdayMonday        -0.215718   0.065935  -3.272  0.00107 ** 
#   outcome_weekdaySaturday       0.823268   0.064877  12.690  < 2e-16 ***
#   outcome_weekdaySunday         0.565916   0.065643   8.621  < 2e-16 ***
#   outcome_weekdayThursday      -0.175065   0.067886  -2.579  0.00991 ** 
#   outcome_weekdayTuesday       -0.205153   0.064037  -3.204  0.00136 ** 
#   outcome_weekdayWednesday     -0.146139   0.066249  -2.206  0.02739 *  
#   sexMale                       0.240743   0.035865   6.712  1.91e-11 ***
#   season_outcomeSummer          0.230451   0.051821   4.447  8.71e-06 ***
#   season_outcomeFall           -0.009918   0.052539  -0.189  0.85027    
#   season_outcomeWinter          0.279770   0.055576   5.034  4.80e-07 ***
#   domestic_breedTrue           -1.163966   0.400486  -2.906  0.00366 ** 
#   cfa_breedTrue                -0.604301   0.394852  -1.530  0.12591

# Statistical Analysis -----

beta_hat <- coef(final_model)
#maximum likelihood estimates of odds ratios (exp(beta))
exp(beta_hat)

# (Intercept)                   0.001875919                
# name_length                   1.320457999      
# Spay.NeuterYes                15.389133058 
# outcome_age_group1-3 months   16.762795658                                  
# outcome_age_group3-6 months   6.615518898
# outcome_age_group6-12 months  1.932666976 
# outcome_age_group1-10 years   2.010121017 
# outcome_age_group10+ years    1.896673058
# outcome_hour                  1.206270344
# outcome_weekdayMonday         0.805962167
# outcome_weekdaySaturday       2.277931356 
# outcome_weekdaySunday         1.761059511 
# outcome_weekdayThursday       0.839402294 
# outcome_weekdayTuesday        0.814522845
# outcome_weekdayWednesday      0.864037972
# sexMale                       1.272194284
# season_outcomeSummer          1.259168380
# season_outcomeFall            0.990130720 
# season_outcomeWinter          1.322825318 
# domestic_breedTrue            0.312245468 
# cfa_breedTrue                 0.546456449 

confint(final_model)#conf int for beta

#                                  2.5 %      97.5 %
# (Intercept)                  -7.1089653 -5.47222729
# name_length                   0.2665403  0.28949914
# Spay.NeuterYes                2.6352576  2.83334146
# outcome_age_group1-3 months   2.6999704  2.93955884
# outcome_age_group3-6 months   1.7401453  2.03964676
# outcome_age_group6-12 months  0.5308419  0.78738404
# outcome_age_group1-10 years   0.5779620  0.81895635
# outcome_age_group10+ years    0.4310385  0.85006947
# outcome_hour                  0.1775772  0.19758586
# outcome_weekdayMonday        -0.3450045 -0.08652874
# outcome_weekdaySaturday       0.6962772  0.95060352
# outcome_weekdaySunday         0.4373778  0.69470868
# outcome_weekdayThursday      -0.3081601 -0.04203955
# outcome_weekdayTuesday       -0.3307162 -0.07968091
# outcome_weekdayWednesday     -0.2760152 -0.01631018
# sexMale                       0.1704670  0.31106209
# season_outcomeSummer          0.1289115  0.33205743
# season_outcomeFall           -0.1129246  0.09303567
# season_outcomeWinter          0.1708804  0.38874634
# domestic_breedTrue           -1.9413939 -0.36288649
# cfa_breedTrue                -1.3704453  0.18630747


exp(confint(final_model))#conf int for exp(beta)

#                                   2.5 %       97.5 %
# (Intercept)                  8.177407e-04  0.004201863
# name_length                  1.305440e+00  1.335758287
# Spay.NeuterYes               1.394690e+01 17.002178181
# outcome_age_group1-3 months  1.487929e+01 18.907503184
# outcome_age_group3-6 months  5.698171e+00  7.687893052
# outcome_age_group6-12 months 1.700363e+00  2.197639960
# outcome_age_group1-10 years  1.782402e+00  2.268131477
# outcome_age_group10+ years   1.538855e+00  2.339809396
# outcome_hour                 1.194320e+00  1.218457680
# outcome_weekdayMonday        7.082172e-01  0.917109189
# outcome_weekdaySaturday      2.006270e+00  2.587270659
# outcome_weekdaySunday        1.548641e+00  2.003125436
# outcome_weekdayThursday      7.347977e-01  0.958831857
# outcome_weekdayTuesday       7.184090e-01  0.923410950
# outcome_weekdayWednesday     7.588014e-01  0.983822110
# sexMale                      1.185859e+00  1.364873959
# season_outcomeSummer         1.137589e+00  1.393832899
# season_outcomeFall           8.932180e-01  1.097500882
# season_outcomeWinter         1.186349e+00  1.475130321
# domestic_breedTrue           1.435038e-01  0.695665396
# cfa_breedTrue                2.539938e-01  1.204792644

# Linear Predictor for final model -----
# ni = -6.278 + (0.278)name_length + (2.733)Spay.NueterYes
# + (2.818)outcome_age_group1-3 months + (1.889)outcome_age_group3-6 months 
# + (0.658)outcome_age_group6-12 months + (0.697)outcome_age_group1-10 years 
# + (0.639)outcome_age_group10+ years + (0.187)outcome_hour 
# - (0.215)outcome_weekdayMonday + (0.823)outcome_weekdaySaturday
# + (0.565)outcome_weekdaySunday - (0.175)outcome_weekdayThursday
# - (0.205)outcome_weekdayTuesday - (0.146)outcome_weekdayWednesday 
# + (0.240)sexMale + (0.230)season_outcomeSummer 
# - (0.009)season_outcomeFall + (0.279)season_outcomeWinter
# - (1.164)domestic_breedTrue - (0.604)cfa_breedTrue

# Statistical Coefficient Interpretations -----

# For context of interpretations, the y variable of outcome_bin is "Yes" 
# (Cat had outcome of adopted) or "No" (cat did not have outcome of adopted). 

#B0: e^b0 is the odds that a cat is adopted when all explanatory variables 
# are set to 0. The odds of a cat being adopted that has a name length of 0, 
# was not spayed/neutered, was in the outcome age group for 0-1 months, 
# had been adopted at the 0th hour of the day, was adopted on Friday, 
# is female, was adopted in the season of Spring, was not a domestic breed, 
# and was not a CFA(Cat Fanciers' Association) breed is e^-6.27 = 0.0018. 

#b1: Holding all factors constant, the odds of a cat being adopted change by 
# a factor of e^0.277 = 1.319 for each additional letter/character added to 
# the name length of a cat.

#b2: Holding all factors constant, the odds of a cat being adopted that is spayed/neutered
# is e^2.733 = 15.378 times the odds of a cat being adopted that is not spayed/neutered. 

#b3: Holding all factors constant, the odds of a cat being adopted that is in the 
# outcome age group of 1-3 months is e^2.819 = 16.760 times the odds of a cat being adopted 
# that is in the outcome age group of 0-1 months.

#b4: Holding all factors constant, the odds of a cat being adopted that is in the 
# outcome age group of 3-6 months is e^1.889 = 6.612 times the odds of a cat being adopted 
# that is in the outcome age group of 0-1 months. 

#b5: Holding all factors constant, the odds of a cat being adopted that is in the 
# outcome age group of 6-12 months is e^0.658 = 1.930 times the odds of a cat being adopted 
# that is in the outcome age group of 0-1 months. 

#b6: Holding all factors constant, the odds of a cat being adopted that is in the 
# outcome age group of 1-10 years is e^0.698 = 2.009 times the odds of a cat being adopted 
# that is in the outcome age group of 0-1 months.

#b7: Holding all factors constant, the odds of a cat being adopted that is in the 
# outcome age group of 10+ years is e^.640 = 1.896 times the odds of a cat being adopted 
# that is in the outcome age group of 0-1 months. 

#b8: Holding all factors constant, the odds of a cat being adopted change by 
# a factor of e^0.187 = 1.205 for each additional hour in the day that the cat was adopted.

#b9: Holding all factors constant, the odds of a cat being adopted that happened on 
# a Friday is e^-0.215 = 0.806 times the odds of a cat being adopted on Monday.

#b10: Holding all factors constant, the odds of a cat being adopted that happened on 
# a Friday is e^0.565 = 1.759 times the odds of a cat being adopted on Saturday.

#b11: Holding all factors constant, the odds of a cat being adopted that happened on 
# a Friday is e^0.823 = 2.277 times the odds of a cat being adopted on Sunday.

#b12: Holding all factors constant, the odds of a cat being adopted that happened on 
# a Friday is e^-0.175 = 0.839 times the odds of a cat being adopted on Thursday.

#b13: Holding all factors constant, the odds of a cat being adopted that happened on 
# a Friday is e^-0.205 = 0.814 times the odds of a cat being adopted on Tuesday.

#b14: Holding all factors constant, the odds of a cat being adopted that happened on 
# a Friday is e^-0.146 = 0.864 times the odds of a cat being adopted on Wednesday.

#b15: Holding all factors constant, the odds of a cat being adopted that is male is 
# e^0.240 = 1.271 times the odds of a cat being adopted that is female.

#b16: Holding all factors constant, the odds of a cat being adopted that happened 
# in Spring is e^0.230 = 1.258 times the odds of a cat being adopted in Summer.

#b17: Holding all factors constant, the odds of a cat being adopted that happened 
# in Spring is e^-0.009 = 0.991 times the odds of a cat being adopted in Fall.

#b18: Holding all factors constant, the odds of a cat being adopted that happened 
# in Spring is e^0.279 = 1.321 times the odds of a cat being adopted in Winter. 

#b19: Holding all factors constant, the odds of a cat being adopted that is a 
# domestic breed is e^-1.163 = 0.312 times the odds of cat being adopted
# that is not a domestic breed.

#b20: Holding all factors constant, the odds of a cat being adopted that is a
# CFA(Cat Fanciers' Association) breed is e^0.546 = 0.546 times the odds
# of a cat being adopted that was not a CFA breed. 



# B1: Holding all factors constant, we are 95% confident that the odds of a 
# cat being adopted change by a factor between the interval of (1.305 , 1.335) for 
# each additional letter/character added to the name of the cat. 

# B2: Holding all factors constant, we are 95% confident that the odds of a spayed/neutered 
# cat being adopted is in the range of (13.946 and 17.002) times the odds of a 
# non spayed/neutered cat being adopted. 

# B3: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that is in outcome age group of 1-3 months is in the range of (14.879 , 18.907)
# times the odds of a cat being adopted that is in the outcome age group of 0-1 months. 

# B4: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that is in outcome age group of 3-6 months is in the range of (5.698 , 7.687)
# times the odds of a cat being adopted that is in the outcome age group of 0-1 months.

# B5: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that is in outcome age group of 6-12 months is in the range of (1.700 , 2.197)
# times the odds of a cat being adopted that is in the outcome age group of 0-1 months. 

# B6: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that is in outcome age group of 1-10 years is in the range of (1.782 , 2.268)
# times the odds of a cat being adopted that is in the outcome age group of 0-1 months. 

# B7: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that is in outcome age group of 10+ years is in the range of (1.538 , 2.339)
# times the odds of a cat being adopted that is in the outcome age group of 0-1 months.

# B8: Holding all factors constant, we are 95% confident that the odds of a 
# cat being adopted change by a factor between the interval of (1.305 , 1.335) for 
# each hour in the day that a cat had been adopted. 

# B9: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that happened on a Monday is in the range of (0.708 , 1.218) times the odds of a cat
# being adopted on a Friday.

# B9: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that happened on a Monday is in the range of (0.708 , 1.218) times the odds of a cat
# being adopted on a Friday. 

# B10: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that happened on a Saturday is in the range of (2.006 , 2.587) times the odds of a cat
# being adopted on a Friday. 

# B11: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that happened on a Sunday is in the range of (1.548 , 2.003) times the odds of a cat
# being adopted on a Friday. 

# B12: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that happened on a Thursday is in the range of (0.073 , 0.958) times the odds of a cat
# being adopted on a Friday. 

# B13: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that happened on a Tuesday is in the range of (0.071 , 0.923) times the odds of a cat
# being adopted on a Friday. 

# B14: Holding all factors constant, we are 95% confident that the odds of a cat adopted
# that happened on a Wednesday is in the range of (0.075 , 0.983) times the odds of a cat
# being adopted on a Friday. 

# B15: Holding all factors constant, we are 95% confident that the odds of a cat being 
# adopted that is male is in the range of (1.185 , 1.364) times the odds of a cat being 
# adopted that is female. 

# B16: Holding all factors constant, we are 95% confident that the odds of a cat being
# adopted in the Summer is in the range of (1.137 , 1.393) times the odds of a cat being 
# adopted in the Spring. 

# B17: Holding all factors constant, we are 95% confident that the odds of a cat being
# adopted in the Fall is in the range of (1.137 , 1.393) times the odds of a cat being 
# adopted in the Spring.  

# B18: Holding all factors constant, we are 95% confident that the odds of a cat being
# adopted in the Winter is in the range of (1.137 , 1.393) times the odds of a cat being 
# adopted in the Spring.  

# B19: Holding all factors constant, we are 95% confident that the odds of a cat being 
# adopted that is a domestic breed is in the range of (0.143 , 0.695) times the odds
# of a cat being adopted that is not a domestic breed.

# B20: Holding all factors constant, we are 95% confident that the odds of a cat being
# adopted that is a CFA(Cat Fanciers' Association) breed is in the range of 
# (0.253 , 1.204) times the odds of a cat being adopted that is not a 
# CFA(Cat Fanciers' Association) breed.
