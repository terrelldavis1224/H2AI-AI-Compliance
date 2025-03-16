#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
#%%
# Set random seed for reproducibility
np.random.seed(42)
#%%
# Number of records to generate
n_records = 100000
#%%

# Generate data
def generate_synthetic_data():
    # Start date for recipient signatures (3 years ago)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 3, 15)

    # Operation types
    operation_types = [
        "Bilateral Tubal Ligation",
        "Vasectomy",
        "Tubal Occlusion",
        "Salpingectomy",
        "Essure Procedure"
    ]

    # Generate random dates for recipient signatures
    random_days = np.random.randint(0, (end_date - start_date).days, n_records)
    recipient_signature_dates = [start_date + timedelta(days=int(day)) for day in random_days]

    # Format dates as strings
    recipient_signature_dates_str = [date.strftime('%m/%d/%Y') for date in recipient_signature_dates]

    # Generate operation types (random selection)
    operation_types_data = np.random.choice(operation_types, n_records)

    # Recipient's signature (binary: 1 for signed, 0 for not signed)
    # Most will be signed (95%)
    recipient_signature = np.random.choice([1, 0], n_records, p=[0.95, 0.05])

    # Race and ethnic designation
    race_ethnicities = ["Asian", "White", "Hispanic", "Black"]
    race_ethnicity_data = np.random.choice(race_ethnicities, n_records)

    # Interpreter's signature (binary: 1 for interpreter used, 0 for no interpreter)
    # Assume 20% of cases need an interpreter
    interpreter_signature = np.random.choice([1, 0], n_records, p=[0.2, 0.8])

    # Generate waiting period data
    # Create a distribution of waiting days, mostly above 30 days (compliant)
    waiting_days = []
    for _ in range(n_records):
        if random.random() < 0.85:  # 85% compliant cases (over 30 days)
            days = random.randint(31, 180)
        else:  # 15% non-compliant or exception cases (under 30 days)
            days = random.randint(1, 30)
        waiting_days.append(days)

    # Waiting period flag (0 if more than 30 days, 1 if less than 30 days)
    waiting_period_flag = [0 if days > 30 else 1 for days in waiting_days]

    # Create DataFrame
    df = pd.DataFrame({
        'SpecifyTypeOfOperation': operation_types_data,
        'RecipientSignature': recipient_signature,
        'DateOfRecipientSignature': recipient_signature_dates_str,
        'RaceAndEthnicDesignation': race_ethnicity_data,
        'InterpreterSignature': interpreter_signature,
        'WaitingPeriodDays': waiting_days,
        'WaitingPeriodFlag': waiting_period_flag
    })

    return df

#%%
# Generate the data
sterilization_data = generate_synthetic_data()
#%%
# Display the first few rows
print(sterilization_data.head())
#%%
# Check for NaN values
print(f"\nNumber of NaN values in the dataset: {sterilization_data.isna().sum().sum()}")
#%%
# Summary statistics
print("\nSummary Statistics:")
print(sterilization_data.describe(include='all'))
#%%
# Distribution of operation types
print("\nOperation Types Distribution:")
print(sterilization_data['SpecifyTypeOfOperation'].value_counts())
#%%
# Distribution of race/ethnicity
print("\nRace and Ethnicity Distribution:")
print(sterilization_data['RaceAndEthnicDesignation'].value_counts())

# Count of records with waiting period less than 30 days
print(f"\nRecords with waiting period less than 30 days: {sterilization_data['WaitingPeriodFlag'].sum()}")

# Save to CSV
sterilization_data.to_csv('tenncare_sterilization_consent_synthetic_data.csv', index=False)
print("\nData saved to 'tenncare_sterilization_consent_synthetic_data.csv'")