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
#### Documents

1. Purpose of Files in GitHub Repository - A standard practice for ReadMe files is to explain the purpose of each individual file in our repository. 

#### References
*Data Source:* The data used is titled "Austin Animal Center Shelter Outcomes" and comes from the user named AaronSchlegel on Kaggle. The original intention of this dataset was to provide outcomes for cats in the Austin Animal Center Shelter, but our group repurposed it for use in our project. 

