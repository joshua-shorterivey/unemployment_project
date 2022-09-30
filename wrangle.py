import pandas as pd
import numpy as np

def acquire_aug():
    ''' 
    Purpose:

    ---

    Parameters:

    ---

    Output:
    
    ---
    '''

    df = pd.read_csv('aug22pub.csv')

    df.columns = df.columns.str.lower()
    
    return df

# create function to flatten race in less categories
def flatten_race(val):
    ''' 
    Purpose:

    ---

    Parameters:

    ---

    Output:
    
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


def prep_values(df):
    ''' 
    Purpose:

    ---

    Parameters:

    ---

    Output:
    
    ---
    '''
    #decided to dropna's cleared up problem with target variable
    df = df.dropna()

    # fixing peafnow  --> get rid of peafnow
    df.iloc[np.where(df.peafnow == 1)]['peafever'] = 1

    #work done to fix usual_hours_worked with manual imputation 
    #mean hours worked for those less that work less than 35
    more_than_35 = round(df[df.pehruslt > 35].pehruslt.mean())
    #mean hours worked for those less that work less than 35
    less_than_35 = round(df[(df.pehruslt > 0) & (df.pehruslt < 35)].pehruslt.mean())
    df[['pehrftpt', 'pehruslt']].value_counts()

    df[df.pehrftpt == 1]['pehruslt'] = more_than_35
    df[df.pehrftpt == 2]['pehruslt'] = less_than_35

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

    ---

    Output:
    
    ---
    '''

    df['ptdtrace'] = df['ptdtrace'].apply(flatten_race)

    final_list = ['hehousut', 'hefaminc', 'hrnumhou', 'hrhtype','hubus',\
    'gediv', 'gestfips', 'gtmetsta', 'gtcbsasz', 'prtage', 'pesex', \
    'pemaritl', 'peafever', 'peeduca', 'ptdtrace', 'pehspnon', 'penatvty', 'pemntvty',
    'pefntvty', 'prcitshp', 'pubus1', 'pudis2', 'pehruslt', 'pelkavl', 'pedwlko', 'pedwwk', 'pejhwant', 'prempnot',\
    'prmjind1', 'prmjocc1', 'peernuot', 'peschenr', 'prchld', 'pecert1']

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
        'pesex': 'sex',
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
        'pudis2': 'disability_preventing_work_in_next_6_months',
        'pehruslt': 'usual_hours_worked',
        'prempnot': 'employed',
        'prmjind1': 'industry',
        'prmjocc1': 'occupation',
        'peernuot': 'usual_ot_tips_commis',
        'peschenr': 'enrolled_in_school',
        'prchld': 'children_in_household',
        'pecert1': 'professional_certification'})

    #fixing types on categorical columns
    categorical_cols = ['housing_type','family_income','household_type',
                        'country_region','state','metropolitan','metro_area_size',
                        'marital_status','education','race','birth_country',
                        'mother_birth_country','father_birth_country','citizenship',
                        'industry','occupation']

    binary_cols = ['own_bus_or_farm', 'sex', 'veteran','hispanic_or_non', 
                'upaid_work_last_week','employed', 'usual_ot_tips_commis',
                'enrolled_in_school','professional_certification']  

    #for loop to handle assignment as object
    for col in categorical_cols:
        df[col] = (df[col].astype('object'))

    # changed all non-affirmative answers to negative. tips, school, most affected
    for col in binary_cols:
        df[col] = np.where(df[col] == 1, 1, 0)                                 

    return df

def prep_aug():
    ''' 
    Purpose:

    ---

    Parameters:

    ---

    Output:

    ---
    '''

    df = acquire_aug()

    df = prep_values(df)

    df = prep_columns(df)

    return df