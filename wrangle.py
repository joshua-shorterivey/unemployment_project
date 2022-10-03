import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def acquire_aug():
    ''' 
    Purpose:
        Acquires local csv that contains information from August 2022 Current Population
        Survey released by the U.S. Census Bureau
    ---
    Parameters:
        None
    ---
    Output:
        df: acquired dataframe
    ---
    '''

    df = pd.read_csv('aug22pub.csv')

    df.columns = df.columns.str.lower()
    
    return df

# create function to flatten race in less categories
def flatten_race(val):
    ''' 
    Purpose:
        To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val == 1:
        val = 'white'
    elif val == 2:
        val = 'black'
    elif val == 3:
        val = 'AI/NA'
    elif val == 4:
        val = 'asian'
    elif val == 5:
        val = 'HI/PI'
    elif 5 < val < 10:
        val = 'mixed_white'
    elif 10 < 27 :
        val = 'mixed_other'
    return val

def flatten_education(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val <= 38:
        val = 'no_high_school'
    elif val <= 40:
        val = 'high_school_ged'
    elif val < 43:
        val = 'associates'
    elif val == 43:
        val = 'bachelor'
    elif val > 43:
        val = 'post-grad'
    return val

def flatten_marital(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val in [1,2,5]:
        val = 'married'
    else:
        val = 'single'
    return val

def flatten_citizenship(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val <= 3:
        val = 'native'
    elif val == 4 :
        val = 'naturalized'
    else :
        val = 'foreign'

    return val

def flatten_household_type(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val == 0:
        val = 'unknown'
    elif val == 2:
        val = 'black'
    elif val == 3:
        val = 'AI/NA'
    elif val == 4:
        val = 'asian'
    elif val == 5:
        val = 'HI/PI'
    elif 5 < val < 10:
        val = 'mixed_white'
    elif 10 < 27 :
        val = 'mixed_other'
    return val

def flatten_birth_country(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val == 57:
        val = 'us_50'
    elif val < 100:
        val = 'us_territories'
    else:
        val = 'foreign_country_elsewhere'
    return val

def flatten_parent(val):
    ''' 
    Purpose:
    To reduce the number of options  within the feature
    ---
    Parameters:
        val
    ---
    Output:
        val
    ---
    '''
    if val == 57:
        val = 'us_50'
    elif val < 100:
        val = 'us_territories'
    else:
        val = 'foreign_country_elsewhere'
    return val

def prep_values(df):
    ''' 
    Purpose:
    ---
    Parameters:
        df: 
    ---
    Output:
        df: prepared data frame with values adjusted and unwanted items removed
    ---
    '''
    #decided to dropna's cleared up problem with target variable
    df = df.dropna()

    # fixing peafnow  --> get rid of peafnow
    mask = df.peafnow == 1
    df.loc[mask,'peafever'] = 1

    # drop invalid responses for industry and occupation
    df = df[(df.prmjind1 != -1) | (df.prmjocc1 != -1)]

    #work done to fix usual_hours_worked with manual imputation 
    #mean hours worked for those less that work less than 35
    more_than_35 = round(df[df.pehruslt > 35].pehruslt.mean())
    
    #mean hours worked for those less that work less than 35
    less_than_35 = round(df[(df.pehruslt > 0) & (df.pehruslt < 35)].pehruslt.mean())
    df[['pehrftpt', 'pehruslt']].value_counts()

    mask =  df.peafnow == 1
    df.loc[mask,'pehruslt'] = more_than_35
    df.loc[mask,'pehruslt'] = less_than_35

    # remove responses from people not in the labor force by means other that discouragement
    df = df[df.prempnot != 4]
    # remove responses from unqualified respondents (children and active armed forces members)
    df = df[df.prempnot != (-1)]
    # binary recode of premmpnot to include repsondents discourage from workforce participation as unemployed
    df.prempnot = np.where(df.prempnot == 1, 1, 0)

    return df

    
def prep_columns(df): 
    ''' 
    Purpose:

    ---
    Parameters:
        df
    ---
    Output:
        df
    ---
    '''

    df['ptdtrace'] = df['ptdtrace'].apply(flatten_race)
    df['penatvty'] = df['penatvty'].apply(flatten_birth_country)
    df['pefntvty'] = df['pefntvty'].apply(flatten_parent)
    df['pemntvty'] = df['pemntvty'].apply(flatten_parent)
    df['hrhtype'] = df['hrhtype'].apply(flatten_household_type)
    df['prcitshp'] = df['prcitshp'].apply(flatten_citizenship)
    df['pemaritl'] = df['pemaritl'].apply(flatten_marital)
    df['peeduca'] = df['peeduca'].apply(flatten_education)


    final_list = ['hehousut', 'hefaminc', 'hrnumhou', 'hrhtype','hubus',\
    'gediv', 'gestfips', 'gtmetsta', 'gtcbsasz', 'prtage', 'pesex', \
    'pemaritl', 'peafever', 'peeduca', 'ptdtrace', 'pehspnon', 'penatvty', 'pemntvty',
    'pefntvty', 'prcitshp', 'pubus1', 'pehruslt', 'pelkavl', 'pedwlko', 'pedwwk', 'pejhwant', 'prempnot',\
    'prmjind1', 'prmjocc1', 'peschenr', 'prchld', 'pecert1']

    df = df[final_list]

    df = df.rename(columns={'hehousut': 'housing_type',
        'hefaminc': 'family_income',
        'hrnumhou': 'household_num',
        'hrhtype': 'household_type',
        'hubus': 'own_bus_or_farm',
        'gediv': 'country_region',
        'gestfips': 'state',
        'gtmetsta': 'metropolitan',
        'gtcbsasz': 'metro_area_size',
        'prtage': 'age',
        'pesex': 'is_male',
        'pemaritl': 'marital_status',
        'peafever': 'veteran',
        'peeduca': 'education',
        'ptdtrace': 'race',
        'pehspnon': 'hispanic_or_non',
        'penatvty': 'birth_country',
        'pemntvty': 'mother_birth_country',
        'pefntvty': 'father_birth_country',
        'prcitshp': 'citizenship',
        'pubus1': 'upaid_work_last_week',
        'pehruslt': 'usual_hours_worked',
        'prempnot': 'employed',
        'prmjind1': 'industry',
        'prmjocc1': 'occupation',
        'peschenr': 'enrolled_in_school',
        'prchld': 'children_in_household',
        'pecert1': 'professional_certification'})

    #fixing types on categorical columns
    categorical_cols = ['housing_type','family_income','household_type',
                        'country_region','state','metropolitan','metro_area_size',
                        'marital_status','education','race','birth_country',
                        'mother_birth_country','father_birth_country','citizenship',
                        'industry','occupation']

    binary_cols = ['own_bus_or_farm', 'is_male', 'veteran','hispanic_or_non', 
                'upaid_work_last_week','employed', 'enrolled_in_school',
                'professional_certification']  

    #for loop to handle assignment as object
    for col in categorical_cols:
        df[col] = (df[col].astype('object'))

    # changed all non-affirmative answers to negative. school, most affected
    for col in binary_cols:
        df[col] = np.where(df[col] == 1, 1, 0)                                 

    return df

def prep_aug():
    ''' 
    Purpose:  
        To prepare August 2022 data
    ---
    Parameters:
        None
    ---
    Output:
        df: a dataframe with prepared columns and values
    ---
    '''

    df = acquire_aug()

    df = prep_values(df)

    df = prep_columns(df)

    return df

def split_scale(df, dummy='n'):
    ''' 
    Purpose:
        To split and scale the input dataframe
    ---
    Parameters:
        df: a tidy dataframe
        dummy: deprecated
    ---
    Output:
        train: unscaled subset of dataframe for exploration and model training
        validate: unscaled and unseen data for model testing
        test: unscaled and unseen data for final model test
        train_scaled: scaled subset of dataframe for exploration and model training
        validate_scaled: scaled and unseen data for model testing
        test_scaled: scaled and unseen data for model testing
    ---
    '''
    #train_test_split
    train_validate, test = train_test_split(df, test_size=.2, random_state=514, stratify=df['employed'])
    train, validate = train_test_split(train_validate, test_size=.3, random_state=514, stratify=train_validate['employed'])
    
    #create scaler object
    scaler = MinMaxScaler()

    # create copies to hold scaled data
    train_scaled = train.copy(deep=True)
    validate_scaled = validate.copy(deep=True)
    test_scaled =  test.copy(deep=True)

    #create list of numeric columns for scaling
    num_cols = train.select_dtypes(include='number')

    #fit to data
    scaler.fit(num_cols)

    # apply
    train_scaled[num_cols.columns] = scaler.transform(train[num_cols.columns])
    validate_scaled[num_cols.columns] =  scaler.transform(validate[num_cols.columns])
    test_scaled[num_cols.columns] = scaler.transform(test[num_cols.columns])

    train_scaled = pd.get_dummies(train_scaled)
    validate_scaled = pd.get_dummies(validate_scaled)
    test_scaled = pd.get_dummies(test_scaled)

    return train, validate, test, train_scaled, validate_scaled, test_scaled
