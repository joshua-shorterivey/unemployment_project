# unemployment_project
* The Current Population Survey (CPS), sponsored jointly by the U.S. Census Bureau and the U.S. Bureau of Labor Statistics (BLS), is the primary source of labor force statistics for the population of the United States. This project uses the August 2022 edition of the Basic Monthly CPS. 

## Project Objectives  
> Document code, process data (through entire pipeline), and articulate key findings and takeways in a jupyter notebook final report 
* Create modules that faciliate project repeatability, as well as final report readability

> Ask/Answer exploratory questions of data and attributes to understand drivers of home value  
* Utilize charts, statistical tests, and various clustering models to drive linear regression models; improving baseline model

> Construct model to predict `unemployed` status 
* 
*  

> Refine work into report in form of jupyter notebook. 
* Detail work done, underlying rationale for decisions, methodologies chosen, findings, and conclusions.

> Be prepared to answer classmate questions about all project areas

## Project  Goals
> Construct ML Regression model that accurately predicts log error of *Single Family Properties* using clustering techniques to guide feature selection for modeling</br>

> '</br>

> Deliver report that the data science team can read through and replicate, while understanding what steps were taken, why and what the outcome was.

> Make recommendations on what works or doesn't work in predicting log error, and insights gained from clustering

## Deliverables
> Github repo with a complete README.md, a final report (.ipynb), other supplemental artifacts and modules created while working on the project (e.g. exploratory/modeling notebook(s))</br>
</br>

## Data Dictionary
|       Target             |           Datatype       |     Definition      |
|:-------------------------|:------------------------:|-------------------:|  
logerror                   | 51736 non-null  float64  | Log(Zestimate) - Log(SalePrice): Difference in estimated and actual

|       Feature            |           Datatype       |     Definition      |
|:------------------------|:------------------------:|-------------------:|  
hryear4                  | 51736 non-null  float64  | Count of bathrooms

HEHOUSUT		2		TYPE OF HOUSING UNIT									
HEFAMINC		2		FAMILY INCOME
HRNUMHOU		2		TOTAL NUMBER OF PERSONS LIVING 
HRHTYPE			2		HOUSEHOLD TYPE					
HUBUS			2		DOES ANYONE IN THIS HOUSEHOLD
GEDIV			1		DIVISION
GESTFIPS		2		FEDERAL INFORMATION
GTMETSTA		1		METROPOLITAN STATUS						
GTCBSASZ		1		Metropolitan Area (CBSA) SIZE
----
PRTAGE			2		PERSONS AGE
PEMARITL		2		MARITAL STATUS 										
PEAFEVER		2		DID YOU EVER SERVE ON ACTIVE 								
PEAFNOW			2		ARE YOU NOW IN THE ARMED FORCES 							
PEEDUCA			2		HIGHEST LEVEL OF SCHOOL 								
PTDTRACE		2		RACE											
PEHSPNON		2		HISPANIC OR NON-HISPANIC								
PRMARSTA		2		MARITAL STATUS BASED ON 								
PENATVTY		3		COUNTRY OF BIRTH
PEMNTVTY
PEFNTVTY									
-----
PRCITSHP		2		CITIZENSHIP STATUS									
PUBUS1			2		LAST WEEK, DID YOU DO ANY								
PERET1			2		DO YOU CURRENTLY WANT A JOB, EITHER 							
PUDIS2			2		DO YOU HAVE A DISABILITY THAT PREVENTS 							
PEHRFTPT		2		DO YOU USUALLY WORK 35 HOURS OR	MORE PER WEEK							
PEHRUSLT		3		SUM OF HRUSL1 AND HRUSL2.								
PELKAVL			2		LAST WEEK, COULD YOU HAVE STARTED  a job if one had been offered					
PEDWLKO			2		DID YOU LOOK FOR WORK AT ANY TIME 	in last 12 months						
PEDWWK			2		DID YOU ACTUALLY WORK AT A JOB OR BUSINESS DURING THE LAST 12 MONTHS?
PEJHWANT		2		DO YOU INTEND TO LOOK FOR WORK DURING
----
PREMPNOT		2		MLR - EMPLOYED, UNEMPLOYED, OR NILF
PRMJIND1		2		MAJOR INDUSTRY RECODE - JOB 1
PRMJOCC1		2		MAJOR OCCUPATION RECODE
PEERNUOT		2		DO YOU USUALLY RECEIVE OVERTIME PAY,
PRCHLD			2		PRESENCE OF OWN CHILDREN <18 YEARS	
PECERT1	  		2		DOES � HAVE A CURRENTLY
-----

# Major Questions and Hypotheses
## Question 1 -  ?
* ${H_0}$: 
* ${H_a}$: 
> Conclusion: 

## Question 2 -  ?
* ${H_0}$: 
* ${H_a}$: 
> Conclusion:

## Question 3 -  ?
* ${H_0}$: 
* ${H_a}$: 
> Conclusion:
 
## Question  -  ?
* ${H_0}$: 
* ${H_a}$: 
> Conclusion:


## Summary of Key Findings and Takeaways
* 
* Model gain on predictive performance vs. baseline prediction using median `logerror` was minimal on test set
    * Baseline Prediction RMSE: 0.1531
    * Model RMSE: 0.1529 (Lower is better) 
-----
</br></br></br>

# Pipeline Walkthrough
## Plan
> Create and build out project README  
> Create skeletons of required, as well as supporting, project modules and notebooks
* `env.py`, `wrangle_zillow.py`,  `model.py`,  `Final Report.ipynb`
* `wrangle.ipynb`,`model.ipynb` ,
> Decide which colums to import
* Work off Data Dictionary from Census
* Survey data begins with 1000, need to shrink that down to being more manageable
* Further pare down from 188 to 35
* Add to data dictionary during process
* Deleted columns identified as filler in census data dictionary, and reduced to items not dependent on other responses
> Make decision on how to handle various subgroups in the survey reponses 
* Decide target disposition, and NLF inclusion
> Deal with null values
> Investigate columns that can be dropped or have redundant information  
> Decide how to deal with outliers  
> Work through questions involving variables typically present on resume
* Primary focus for first pass through explore phase
> Craft general explore section outline
* Include areas for feature engineering via cluster, rfe, and selectKbest
> Work on modeling section
* Craft functions for model testing. **NO BIG MASS TESTING FUNCTION**
* Prep MVP
* Decide which questions to highlight within Final Report
* Craft outline/skeleton for final report



> Explore
- 
> Clustering
- Decide on which features to use when crafting clusters
- Create cluster feature sets
- Add cluster labels as features  
> Statistical testing based on clustering
- Create functions that iterate through statistical tests

> Modeling
* Create functions that automate iterative model testing
    - Adjust parameters and feature makes
* Handle acquire, explore, and scaling in wrangle
> Verify docstring is implemented for each function within all notebooks and modules 
 

## Acquire
> Acquired community survey data from census.gov
* Created local .csv of raw data upon initial acquisition for later use
* 
* Take care of any null values -> Decide on impute or elimination
* Eliminated 
> 

## Prepare
> Univariate exploration: 
* Basic histograms/boxplot for categories
> Take care of outliers  
> Handle any possible threats of data leakage
* Removed log error bins to prevent leakage

> Feature Engineering **shifted to accomodate removal of outliers*
* 
* Cluster modeling: 
> - 

> Split data  
> Scale data  
> Collect and collate section *Takeaways*  
> Add appropirate artifacts into `wrangle.py` or `explore.py`

## Explore
* Removed Year Built, and Tax Amount
> Bivariate exploration
* Investigate and visualize features against log error
> Identify additional possible areas for feature engineering (clustering)
* Use testing and visualizations to determine which features are significant in determining difference in logerror
> Multivariate:
* Visuals exploring features as they relate to home value
> Statistical Analysis:
> Collect and collate section *Takeaways*

## Model
> Ensure all data is scaled  
> Create dummy vars of categorical columns  
> Set up comparison dataframes for evaluation metrics and model descriptions    
> Set Baseline Prediction and evaluate RMSE and r^2 scores  
> Explore various models and feature combinations.
* For initial M.V.P of each model include only single cluster label features
> Choose **Four** Best Models to add to final report

>Choose **one** model to evaluate on Test set
* GLM 1
* Power: 3
* Alpha: 0
* Features: All features and cluster labels 

> Collect and collate section *Takeaways*

## Deliver **
> Create project report in form of jupyter notebook  
> Finalize and upload project repository with appropriate documentation 
> Created recorded presentation for delivery
----
</br>

## Project Reproduction Requirements
> Requires personal `----` file containing database credentials  
> Steps:
* Fully examine this `README.md`
* Download --- to working directory
* Create and add personal --- file to directory. Requires user, password, and host variables
* Run `Final Report.ipynb`