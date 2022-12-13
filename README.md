# ReadMe: Deliverable for Purrfect Match Cats & Co. 
*This ReadMe file corresponds to the Final Project & Presentation of the PCTeam for STAT 172. This group consists of members Joel Aguirre, Madelyn Osten, and Sam Solheim. The analysis done throughout this project is intended solely for the purpose of showcasing what we have learned throughout the semester. In the References section below, we detail from whom and where our data was sourced from.*

## Summary
Purrfect Match Cats & Co. is an non-profit cat rescue that houses thousands of felines every year. Additionally, for this project we can assume that they are located within the Greater Des Moines area. Purrfect Match came to our team recently looking to find out what leads to a "perfect match" for animals in their care, so that more of their rescued cats can find their forever homes. They came to us because they have lots of data on the cats that have been adopted or returned to their owners, and would like data driven interpretations to help shift focus as needed. 

- *Problem:* Purrfect Match has a lot of data on the animals in their shelters, but they don't know how to use it. 
- *Goal:* Determine what factors lead to cats getting adopted or returned to their owner (if applicable). 

### Methods
1. Data Cleaning & Variable Creation
   - Creating Binary Y Variable (Outcome)
     - Clean 'Empty' Values in Outcome
     - Convert to a binary variable with levels "Yes" & "No"
   - Creating Additional X Variables
     - Season Outcome created from Outcome Month
     - Name Length created from Name
     - Outcome Age Group created from Age Upon Outcome
2. Predictive Modeling
   - Random Forest
   - ROC Curve
   - Variable Importance Plots

3. Descriptive Modeling
   - Logistic Regression, specifically utilizing a binomial random component and a logit link
   - Interpretations of Variable Coefficients

### Documents & References
#### Main Documents
1. *PMCC_FinalProject_Main_Code.R:* This R file is the shared environment we utilized to make sure everyone was using the same standard code throughout the entire project. For each section of our code, there should be additional comments provided for context in our thought processes and motivations. PMCC at the beginning of this document's name stands for "Purrfect Match Cats & Co." since that is the company we designed as our customer. 

2. *RandomForest_Charts for Final Project Code.docx:* This Word document contains charts and plots created during the Predictive Modeling portion of the project. They include a chart for our baseforest plotting the estimated OOB error rate by the number of trees within the forest, a chart showing the estimated OOB error rate as compared to the number of sampled variables per tree, an ROC Curve based on our final random forest, and lastly the variable importance plot created from our final forest. 

3. **STAT 172 Final model summary.docx:** This document provides the output of our final model chosen from the "Descriptive Modeling" section of our project. It includes our GLM formula including both the distribution and link components, along with the general summary statistics of the resulting model. 

4. *STAT 172 Visualizations.docx:* This document provides visuals that will be utilized in the final presentation of this project. Charts included are a violin chart on the distribution of name lengths vs adoption outcome, a bar chart showing the proportion of cats adopted per season, and two charts that provide context on how a cat's age group and whether they are spayed or neutered affects the final adoption outcome. One of these charts shows this relationship utilizing the total number of cats adopted / not adopted in each age group conditional on whether they are spayed or neutered, and the other chart shows the total proportions of adopted / not adopted cats based on the same criteria. 

5. *aac_shelter_cat_outcome_eng.csv.zip:* Lastly, this file contains the original dataset that was used throughout the course of our project. 


#### Secondary Documents
1. *Professor Follett's Feedback:* For easy access to the entire team, we created a text file with the feedback Professor Follett had given us after the submission of our Project Proposal. Additionally, we wrote quick responses to the comments so we would all be on the same page on where we wanted to take the feedback. 

2. *Final_Project_BinaryYCreation.R:* This R file contains all the code utilized during the cleaning of our Y variable and its subsequent transformation into a binary variable. The levels for this variable are "Yes" and "No", where "Yes" corresponds to all the cats that were adopted or returned to their owners, and "No" goes to all cats not included in that grouping. This code was later put directly into our main code file. 

3. *Custom_Color_Palettes:* This document contains the R code that was utilized to create a cat themed color palette that we could utilize in creating meaningful visuals. 

#### References
*Data Source:* The data used is titled "Austin Animal Center Shelter Outcomes" and comes from the user named AaronSchlegel on Kaggle. The original intention of this dataset was to provide outcomes for cats in the Austin Animal Center Shelter, but our group repurposed it for use in our project. 

