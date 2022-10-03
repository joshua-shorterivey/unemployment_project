# unemployment_project
* The Current Population Survey (CPS), sponsored jointly by the U.S. Census Bureau and the U.S. Bureau of Labor Statistics (BLS), is the primary source of labor force statistics for the population of the United States. This project uses the August 2022 edition of the Basic Monthly CPS. 

## Project Objectives  
> Document code, process data (through entire pipeline), and articulate key findings and takeways in a jupyter notebook final report 
* Create modules that faciliate project repeatability, as well as final report readability

> Construct model to predict `employed` status 

> Refine work into report in form of jupyter notebook. 
* Detail work done, underlying rationale for decisions, methodologies chosen, findings, and conclusions.

> Be prepared to answer classmate questions about all project areas

## Project  Goals
> Construct ML Classification model that accurately predicts `employed` of *Single Family Properties* using clustering techniques to guide feature selection for modeling</br>

> '</br>

> Deliver report that the classmates can read through and replicate, while understanding what steps were taken, why and what the outcome was.

> Make recommendations on what works or doesn't work in predicting `employed` status

## Deliverables
> Github repo with a complete README.md, a final report (.ipynb), other supplemental artifacts and modules created while working on the project (e.g. exploratory/modeling notebook(s))</br>
</br>

## Data Dictionary
|       Target           | Non_Null Count/Datatype  |     Definition     |
|:-----------------------|:------------------------:|-------------------:|  
employed                 | 50556 non-null  object  | employed

|       Feature          |           Datatype      |     Definition     |
|:----------------------|:------------------------:|-------------------:|  
housing_type		    | 50556 non-null  object  |    TYPE OF HOUSING UNIT
family_income		    | 50556 non-null  object  |	FAMILY INCOME
household_num		    | 50556 non-null  object  |	TOTAL NUMBER OF PERSONS LIVING 
household_type		    | 50556 non-null  object  |	HOUSEHOLD TYPE					
own_bus_or_farm		    | 50556 non-null  object  |	DOES ANYONE IN THIS HOUSEHOLD
country_region		    | 50556 non-null  object  |	DIVISION
state		            | 50556 non-null  object  |	FEDERAL INFORMATION
metropolitan		    | 50556 non-null  object  |	METROPOLITAN STATUS						
metro_area_size		    | 50556 non-null  object  |	Metropolitan Area (CBSA) SIZE
age                     | 50556 non-null  object  |	PERSONS AGE
marital_status		    | 50556 non-null  object  |	MARITAL STATUS 										
veteran		            | 50556 non-null  object  |	DID YOU EVER SERVE ON ACTIVE DUTY								
education			    | 50556 non-null  object  |	HIGHEST LEVEL OF SCHOOL 								
race		            | 50556 non-null  object  |	RACE											
hispanic_or_non		    | 50556 non-null  object  |	HISPANIC OR NON-HISPANIC								
birth_country		    | 50556 non-null  object  |	COUNTRY OF BIRTH
mother_birth_country    | 50556 non-null  object  |	COUNTRY OF MOTHER'S BIRTH
father_birth_country    | 50556 non-null  object  |	COUNTRY OF FATHER'S BIRTH									
citizenship	            | 50556 non-null  object  |	CITIZENSHIP STATUS									
upaid_work_last_week	| 50556 non-null  object  |	LAST WEEK, DID YOU DO ANY													
usual_hours_worked		| 50556 non-null  object  |	DO YOU USUALLY WORK 35 HOURS OR	MORE PER WEEK							
children_in_household	| 50556 non-null  object  |	PRESENCE OF OWN CHILDREN <18 YEARS	
professional_certification |51736 non-null float64 |    DOES � HAVE A CURRENTLY

### Spotlight - Professional Certification
* **Question:** What is the effect of having a professional certification? 
 
* **Answer:** Most indivduals do not have certification, but those that do have a 2% unemployment rate vs 4% for those without.

#### Statistical Hypothesis
>* ${H_0}$: There is no relationship between having a `professional_certification` and `employment`  
>* ${H_a}$: There is a relationship between having a `professional_certification` and `employment`    
>* ${\alpha}$: .05
>* Result: There is enough evidence to reject our null hypothesis. 

### Spotlight - Race
* **Question:** Which industry shows the largest population proportion change between employed and unemployed?

* **Answer:** Indivduals identifying as White show the largest population proportion change with a drop of nearly 10% when comparing employed vs unemployed. Those identifying as mixed race other than with white, and Indigenous have the highest unemployed rates at 12% and 7% respectively. 

#### Statistical Hypothesis
>* ${H_0}$: There is no relationship between `race` and `employment` status   
>* ${H_a}$: There is a relationship between `race` and `employment` status   
>* ${\alpha}$: .05  
>* Result: There is enough evidence to reject our null hypothesis. 

### Spotlight - Industry 
* **Question:** Which industry shows the largest population proportion change between employed and unemployed?  
* **Answer:** Leisure and Hospitality. This industry also has the highest unemployment rate at 6%

#### Statistical Hypothesis
>* ${H_0}$: There is no relationship between industry of typical employment and employment status   
>* ${H_a}$: There is a relationship between industry of typical employment and employment status  
>* ${\alpha}$: .05  
>* Result: There is enough evidence to reject our null hypothesis. 
 
### Spotlight - Professional Certification
* **Question:** What is the effect of having a professional certification? 
 
* **Answer:** Most indivduals do not have certification, but those that do have a 2% unemployment rate vs 4% for those without.

#### Statistical Hypothesis
>* ${H_0}$: There is no relationship between having a `professional_certification` and `employment`  
>* ${H_a}$: There is a relationship between having a `professional_certification` and `employment`    
>* ${\alpha}$: .05
>* Result: There is enough evidence to reject our null hypothesis.

## Summary of Key Findings
>* As a portion the population unemployed individuals are small.
>- Overwhelming majorities of people are either employed or not in the labor force due to disability, retirement, or other reasons. 
>* Testing confirms that being educated, married, and with a professional certification are good ways to stay away from unemployement. 
>- From a modeling/ML perspective this project was not one where the Accuracy/Scoring played a big part. The best models in that area underperformed when it came to correctly identifying unemployed persons.
>* Shifted focus to generating models with better Sensitivy/True Negative Rate performance
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
* Go back through data dictionary
* Craft outline/skeleton for final report
    * Takeaways highlighted    
* Look into creating feature that blends `household_num` and `family_income` to measure at or above median income for family of that size
----

# Wrangle (Acquire and Prep)
> * Largest and most time intensive part of wrangle was deciding which columns to drop. 
> - Deviated heavily from initial plan to **only** feature resume-like categories. Sub-optimal decision.

### Nulls/Missing Values
> * Dropped any records that have 'NaN' for target variable, indicates incomplete survey/data.
> - Same rationale used to justify dropping most observations that had a '-1' value as that indicated the repsondent did not an apporiate repsonse for the area of the survey
> * For some others such as `usual_hours_worked` information from multiple columns was used to infer proper disposition
> - Example: Individuals that report variable work hours were assigned population mean hours work for above/or below 35 hours. 
> * Further work on the project will require more research of how the survey is conducted and the data entered. 
---
### Feature Engineering 
> * Decided against engineering features due to poor model performance with accurately predicting employment disposition. No point increasing complexity
---   
### Flattening
> * Had to make decisions in order to remove optionality from certain categorical columns when preparing the data
> - Example: `race` orginally had over 20 different categories and was ~flattened~ down to 7 
> * Decisons here driven mostly by desire to create larger cohorts within  features because the unemployment is already such as small amount

## Exploration Summary
* Overall the conventional wisdom surrounding job prospects held true.

* It benefits an indvidual to acquire advanced dregrees and certifications
* Having a job or career in an industry that leans more towards being a profession helps
* With more time I want to dive into cross examinations of factors to see how they interact, but I'm doubtful that would help more than simply satisfying my curiousity. 
---
# Modeling
* Models process was initially guided by Accuracy, but over time realized that approach was not actually doing a good job of capturing the obervations that were actually unemployed

* After the 2 round of mass model testing adjust to filtering for models that had higher Sensitivity scores to more effectively capture the target. Prefer a model that correctly identifies a person that is unemployed than one that misses the vast majority

* The models appeared to have an inverse relationship in this regard with many of the models having a Sensitivity rate above 30% also having an Accuracy below 70%

* With the employed vs unemployed numbers being around 96% vs 4% this led me to feel the overall project is going to need **much** more work to have any greater value

## Feature Groups for Modeling
* Grouped by subject matter into four clumps in leiu of clustering

* Feature Set 1: `industry`, `occupation`, `country_region`, `metro_area_size` , `professional_certification`, `own_bus_or_farm`,`education`  
    - Chosen to highlight the business oriented concerns around employement
    ---
* Feature Set 2: `household_num`, `children_in_household`, `education`, `enrolled_in_school`, `family_income`, `marital_status`
    - Highlights family and environment characteristics
    ---
* Feature Set 3: `age`, `is_male`, `veteran`, `hispanic_non`, `race`, `birth_country`, `mother_birth_country`, `father_birth_country`, `citizenship`, `education` 
    - Highlights personal characteristics 
    ---
* Feature Set 4: `age`, `industry`, `occupation`,`professional_certification`,`education`,`marital_status`,`is_male`,`citizenship`
    - Highlights areas that may appear on the typical resume or job applicaton

## Deliver **
> Create project report in form of jupyter notebook  
> Finalize and upload project repository with appropriate documentation 
> Share link for classmate review
----
</br>

## Project Reproduction Requirements
> Requires .csv featuring Aug 2022 CPS Data from U.S. Census Bureau
> Steps:
* Fully examine this `README.md`
* Download .csv to working directory
* Run `Final Report.ipynb`