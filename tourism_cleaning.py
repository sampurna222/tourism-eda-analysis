import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(99)

# STEP 1 - SIMULATE MESSY TOURISM DATASET
print('='*60)
print('STEP 1: Generating raw messy tourism dataset...')
print('='*60)

n = 300
countries_clean = ['India', 'France', 'USA', 'Thailand', 'Italy', 'Spain', 'Japan', 'UAE', 'Brazil', 'Canada']
purposes       = ['Leisure', 'Business', 'Medical', 'Education', 'Transit']
seasons        = ['Spring', 'Summer', 'Autumn', 'Winter']
accommodations = ['Hotel', 'Hostel', 'Airbnb', 'Resort', 'Guesthouse']

raw_data = {
    'Tourist_ID'    : [f'T{str(i).zfill(4)}' for i in range(1, n+1)],
    'Country'       : np.random.choice(countries_clean, n),
    'Age'           : np.random.randint(18, 75, n),
    'Gender'        : np.random.choice(['Male', 'Female', 'Other'], n),
    'Visit_Date'    : pd.date_range('2022-01-01', periods=n, freq='D').strftime('%Y-%m-%d').tolist(),
    'Duration_Days' : np.random.randint(1, 30, n),
    'Spend_USD'     : np.round(np.random.uniform(200, 5000, n), 2),
    'Purpose'       : np.random.choice(purposes, n),
    'Satisfaction'  : np.random.randint(1, 6, n),
    'Season'        : np.random.choice(seasons, n),
    'Accommodation' : np.random.choice(accommodations, n),
    'Group_Size'    : np.random.randint(1, 10, n),
}
df_raw = pd.DataFrame(raw_data)

# Inject dirty issues
for col in ['Age','Spend_USD','Duration_Days','Satisfaction','Purpose']:
    idx = np.random.choice(df_raw.index, size=int(n*0.08), replace=False)
    df_raw.loc[idx, col] = np.nan

dup_rows = df_raw.sample(12, random_state=1)
df_raw = pd.concat([df_raw, dup_rows], ignore_index=True)

typo_map = {'India':['india','INDIA','Indiaa'],'France':['france','Frnace'],'USA':['usa','U.S.A','Usa'],'Thailand':['Thailand ','thailand'],'Italy':['italy','ITALY']}
for correct, variants in typo_map.items():
    mask = df_raw['Country'] == correct
    vi = df_raw[mask].sample(frac=0.3, random_state=2).index
    df_raw.loc[vi, 'Country'] = np.random.choice(variants, len(vi))

df_raw.loc[np.random.choice(df_raw.index, 15, replace=False), 'Gender'] = np.random.choice(['M','F','male','female','MALE','FEMALE'], 15)
df_raw.loc[10,  'Spend_USD']     = 950000
df_raw.loc[55,  'Age']           = 185
df_raw.loc[120, 'Duration_Days'] = 500
df_raw.loc[200, 'Spend_USD']     = -300
bad_date_idx = np.random.choice(df_raw.index, 20, replace=False)
df_raw.loc[bad_date_idx, 'Visit_Date'] = pd.to_datetime(df_raw.loc[bad_date_idx, 'Visit_Date']).dt.strftime('%d/%m/%Y')
df_raw.loc[np.random.choice(df_raw.index, 8, replace=False), 'Satisfaction'] = np.random.choice([0,6,7,-1], 8)
df_raw.loc[np.random.choice(df_raw.index, 5, replace=False), 'Group_Size'] = -1

df_raw.to_csv(r'c:\Users\LOQ\OneDrive\Documents\New\raw_cleaning_data.csv', index=False)
print(f'Raw dataset shape  : {df_raw.shape}')
print(f'Total missing vals : {df_raw.isnull().sum().sum()}')
print(f'Saved raw_cleaning_data.csv')

# STEP 2 - INITIAL INSPECTION
print('\n'+'='*60+'\nSTEP 2: Initial Inspection\n'+'='*60)
df = df_raw.copy()
missing_before = df.isnull().sum()
dups_before = df.duplicated().sum()
print('Missing values:\n', missing_before)
print(f'Duplicates: {dups_before}')

# STEP 3 - REMOVE DUPLICATES
print('\n'+'='*60+'\nSTEP 3: Removing Duplicates\n'+'='*60)
before = len(df)
df = df.drop_duplicates()
print(f'Removed {before - len(df)} duplicates. Rows now: {len(df)}')

# STEP 4 - DATE FORMAT
print('\n'+'='*60+'\nSTEP 4: Standardizing Date Format\n'+'='*60)
df['Visit_Date'] = pd.to_datetime(df['Visit_Date'], dayfirst=False, errors='coerce')
still_null = df['Visit_Date'].isnull().sum()
if still_null > 0:
    df['Visit_Date'] = pd.to_datetime(df['Visit_Date'], dayfirst=True, errors='coerce')
df['Visit_Month'] = df['Visit_Date'].dt.month
df['Visit_Year']  = df['Visit_Date'].dt.year
print('Dates standardized to YYYY-MM-DD. Month/Year extracted.')

# STEP 5 - INCONSISTENT TEXT
print('\n'+'='*60+'\nSTEP 5: Fixing Country & Gender Inconsistencies\n'+'='*60)
df['Country'] = df['Country'].str.strip().str.title()
corrections = {'Indiaa':'India','Frnace':'France','U.S.A':'USA','Usa':'USA'}
df['Country'] = df['Country'].replace(corrections)
print('Countries:', sorted(df['Country'].unique()))

gender_map = {'M':'Male','male':'Male','MALE':'Male','F':'Female','female':'Female','FEMALE':'Female'}
df['Gender'] = df['Gender'].replace(gender_map)
print('Genders:', df['Gender'].unique())

# STEP 6 - IMPOSSIBLE VALUES
print('\n'+'='*60+'\nSTEP 6: Fixing Impossible Values\n'+'='*60)
df.loc[(df['Age']<1)|(df['Age']>120), 'Age'] = np.nan
df.loc[(df['Duration_Days']<1)|(df['Duration_Days']>365), 'Duration_Days'] = np.nan
df.loc[df['Spend_USD']<0, 'Spend_USD'] = np.nan
df.loc[(df['Satisfaction']<1)|(df['Satisfaction']>5), 'Satisfaction'] = np.nan
df.loc[df['Group_Size']<1, 'Group_Size'] = np.nan
print('Impossible values in Age, Duration_Days, Spend_USD, Satisfaction, Group_Size replaced with NaN')

# STEP 7 - OUTLIERS
print('\n'+'='*60+'\nSTEP 7: Outlier Detection & Winsorization\n'+'='*60)
outlier_log = {}
for col in ['Spend_USD','Age','Duration_Days','Group_Size']:
    cd = df[col].dropna()
    Q1,Q3 = cd.quantile(0.25), cd.quantile(0.75)
    IQR = Q3-Q1
    lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
    n_iqr = int(((cd<lo)|(cd>hi)).sum())
    z = np.abs(stats.zscore(cd))
    n_z = int((z>3).sum())
    df[col] = np.clip(df[col], lo, hi)
    outlier_log[col] = {'IQR':n_iqr,'ZScore':n_z,'Lower':round(lo,2),'Upper':round(hi,2)}
    print(f'  {col:20s}: IQR={n_iqr}, Z-score={n_z}, capped to [{lo:.2f},{hi:.2f}]')

# STEP 8 - MISSING VALUES
print('\n'+'='*60+'\nSTEP 8: Imputing Missing Values\n'+'='*60)
for col in ['Age','Spend_USD','Duration_Days','Group_Size']:
    m = df[col].median()
    n_filled = int(df[col].isnull().sum())
    df[col] = df[col].fillna(m)
    print(f'  {col:20s}: filled {n_filled} NaN with median={m:.2f}')

mode_sat = df['Satisfaction'].mode()[0]
n_sat = int(df['Satisfaction'].isnull().sum())
df['Satisfaction'] = df['Satisfaction'].fillna(mode_sat)
print(f'  Satisfaction        : filled {n_sat} NaN with mode={mode_sat}')

mode_p = df['Purpose'].mode()[0]
n_p = int(df['Purpose'].isnull().sum())
df['Purpose'] = df['Purpose'].fillna(mode_p)
print(f'  Purpose             : filled {n_p} NaN with mode={mode_p}')

# STEP 9 - DATA TYPES & FEATURES
print('\n'+'='*60+'\nSTEP 9: Type Correction & Feature Engineering\n'+'='*60)
df['Age'] = df['Age'].astype(int)
df['Duration_Days'] = df['Duration_Days'].astype(int)
df['Satisfaction']  = df['Satisfaction'].astype(int)
df['Group_Size']    = df['Group_Size'].astype(int)
df['Spend_Per_Day']    = (df['Spend_USD'] / df['Duration_Days']).round(2)
df['Spend_Per_Person'] = (df['Spend_USD'] / df['Group_Size']).round(2)
df['Age_Group'] = pd.cut(df['Age'], bins=[0,25,35,50,65,120], labels=['18-25','26-35','36-50','51-65','65+'])
print('Types corrected. New features: Spend_Per_Day, Spend_Per_Person, Age_Group')

# STEP 10 - VALIDATION
print('\n'+'='*60+'\nSTEP 10: Data Validation\n'+'='*60)
assert df.isnull().sum().sum() == 0
assert (df['Spend_USD'] >= 0).all()
assert (df['Age'].between(1,120)).all()
assert (df['Satisfaction'].between(1,5)).all()
assert (df['Duration_Days'].between(1,365)).all()
assert df['Gender'].isin(['Male','Female','Other']).all()
assert df.duplicated().sum() == 0
print('All validation checks PASSED!')

# STEP 11 - SAVE
df.to_csv(r'c:\Users\LOQ\OneDrive\Documents\New\cleaned_tourism_preprocessing.csv', index=False)
print(f'\nCleaned dataset saved: {df.shape}')

# Summary
print('\n'+'='*60+'\nCLEANING SUMMARY\n'+'='*60)
print(f'Raw shape              : {df_raw.shape}')
print(f'Cleaned shape          : {df.shape}')
print(f'Duplicates removed     : {before - len(df)}')
print(f'Missing vals raw       : {df_raw.isnull().sum().sum()}')
print(f'Missing vals cleaned   : {df.isnull().sum().sum()}')
print(f'Outlier treatment      : IQR Winsorization')
print(f'New features added     : Spend_Per_Day, Spend_Per_Person, Age_Group')
print(f'Validation checks      : All Passed')
print('\ntourism_cleaning.py complete.')
