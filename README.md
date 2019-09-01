# concrete_overdesign_PARIS Project Description

The dataset contains concrete strength based on different inputs (concrete mixture design). 

For each type of concrete (with different inputs), the dataset contains 
(a) the strength that was targeted (target) 
(b) the strength that was actually measured (output). 

Based on this, one can discriminate two classes of concrete: 
(a) those that meet or exceed the target and 
(b) those are below target (poor strength).

Here, we would like to predict 
(a) the propensity for a concrete to be below target (poor strength) based on the knowledge of the concrete mixture design (inputs) and 
(b) the level of overdesign (ratio of real strength divided by target strength). Concretes below target are also those with a level of overdosing lower than 1.

(A) is a classification problem that can be tackled with SVM
(B) is a regression problem that can be approached by linear regression, random forest, artificial neural network, etc.



