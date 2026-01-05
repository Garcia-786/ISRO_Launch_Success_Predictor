import pandas as pd

def load_and_clean(csv_path="data/isro_launches.csv"):
    # Load CSV safely
    df = pd.read_csv(csv_path, encoding="latin1")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Handle missing orbit_type
    df['orbit_type'] = df['orbit_type'].fillna('Unknown')
    
    # Clean launch_vehicle names
    df['launch_vehicle'] = df['launch_vehicle'].str.split('/').str[0]
    df['launch_vehicle'] = df['launch_vehicle'].str.split(',').str[0]
    df['launch_vehicle'] = df['launch_vehicle'].str.strip()

    # Clean orbit_type
    df['orbit_type'] = df['orbit_type'].str.split('(').str[0].str.strip()

    # Handle outcome column
    if 'remarks' in df.columns:
        df = df.rename(columns={'remarks': 'outcome'})
    df = df[df['outcome'].isin(['Launch successful', 'Launch unsuccessful'])]
    df['outcome'] = df['outcome'].map({'Launch successful': 1, 'Launch unsuccessful': 0})

    # One-hot encode categorical columns
    categorical_cols = ['launch_vehicle', 'orbit_type', 'application']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df
