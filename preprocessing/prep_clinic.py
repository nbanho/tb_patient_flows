import pandas as pd

# Path to the Excel file
file_path = 'data-raw/clinical/complete-clinic-data.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# Select and rename the columns
df = df[["ID", "Start time", "Completion time", "2. Gender", "3. Age", "4. HIV status", "7. Date of confirmation (start of TB treatment)", "9.\xa0Xpert TB test 1 MTB result", "12. Chest X-ray result"]]
df.columns = ["id", "start_time", "completion_time", "gender", "age", "hiv_status", "tb_treat_status", "tb_test_res", "chest_xray_res"]

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.lower().str.strip()

df['start_time'] = pd.to_datetime(df['start_time'])
df['completion_time'] = pd.to_datetime(df['completion_time'])
df['date'] = df['start_time'].dt.date
df['start_time'] = df['start_time'].apply(lambda x: x.timestamp() * 1000)
df['completion_time'] = df['completion_time'].apply(lambda x: x.timestamp() * 1000)

# Preprocessing steps
df["tb_treat_status"] = df["tb_treat_status"].where(pd.notnull(df["tb_treat_status"]), None)

df["tb_test_res"] = df["tb_test_res"].replace({
    "not detected": "negative",
    "1detected(mtb+)": "positive",
    "detected": "positive",
    "not submitted": None,
    "3invalid/error": None
})

# Define the function to determine tb_status
def determine_tb_status(row):
    if pd.notnull(row['tb_treat_status']):
        date_diff = (row['date'] - row['tb_treat_status'].date()).days
        if date_diff < 30:
            return "infectious"
        else:
            return "not infectious"
    if row['tb_test_res'] == "positive":
        return "infectious"
    if row['chest_xray_res'] == "positive":
        return "infectious"
    return "presumptive"

# Apply the function to each row
df['tb_status'] = df.apply(determine_tb_status, axis=1)

# Example usage
print(df.head())
print(pd.to_datetime(df.start_time[0], unit='ms', origin='unix', utc=True))
print(df.tb_test_res.value_counts(dropna=False))
print(df.tb_status.value_counts(dropna=False))

# save
df.to_csv('data-clean/clinical/tb_cases.csv', index=False)