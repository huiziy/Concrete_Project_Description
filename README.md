# Project Description

## Dataset Introduction
The data set was provided by an international vertically integrated cement/concrete producer (VIP) from across a range of different concrete production sites and consisted of 9994 measured compressive strengths from job-site mixtures that were sampled across various locations in the United States. Three samples corresponding to a given mixture was collected at the job-site (i.e., in the form of 6″×12″ cylinders) and cured following ASTM C39 for 28 days after which time their compressive strengths were measured. 

The data set contained mixture proportions in terms of: 
1. w/c, cement and fly ash contents (in kg per m3 of concrete)
2. water-reducing admixture (WRA) and air-entraining admixture contents (in kg per 100 kg of cementitious material) 
3. coarse and fine aggregate contents (in kg per m3 of concrete)
4. fresh air content (in volume %) for each mixture
5. the strength that was targeted (target) 
6. the strength that was actually measured (output)

The mixture proportions reported reflect the actual mixture proportions, i.e., based on the batch weights. Furthermore, all mixtures in the industrial dataset used ASTM C150 compliant Type I/II OPC. In general, Class F fly ash compliant with ASTM C618 was used in all relevant cases. Finally, the aggregates used were compliant with ASTM C33.

## Project Objective:
Based on composition (input), target and strength (output), one can discriminate two classes of concrete:
1. those that meet or exceed the target and 
2. those are below target (poor strength).

The goal is to use machine learning algorithms (SVM, linear regression, LASSO, random forest, artificial neural network XGBoost) to accuratly predict: 

1. the propensity for a concrete to be below target (poor strength) based on the knowledge of the concrete mixture design (inputs) and 
2. the level of overdesign (ratio of real strength divided by target strength). Concretes below target are also those with a level of overdosing lower than 1.







