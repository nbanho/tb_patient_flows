import pandas as pd

# Path to the Excel file
file_path = 'data-raw/clinical/complete-clinic-data edit 17.02.25.xlsx'

# Read the Excel file
df = pd.read_excel(file_path, dtype=str)

# Select and rename the columns
df = df[["ID", "Start time", "Completion time", "2. Gender", "3. Age", "4. HIV status", "7. Date of confirmation (start of TB treatment)", "9.\xa0Xpert TB test 1 MTB result", "12. Chest X-ray result"]]
df.columns = ["clinic_id", "start_time", "completion_time", "gender", "age", "hiv_status", "tb_treat_status", "tb_test_res", "chest_xray_res"]

# convert dates
def parse_dates(date_series, formats):
    parsed_dates = []
    for date_str in date_series:
        parsed_date = None
        for fmt in formats:
            try:
                parsed_date = pd.to_datetime(date_str, format=fmt)
                break
            except (ValueError, TypeError):
                continue
        if parsed_date is None:
            parsed_date = pd.to_datetime(date_str, errors='coerce')  # Fallback to default parsing
        parsed_dates.append(parsed_date)
    return pd.Series(parsed_dates)

# parse dates
date_formats = [
    "%Y-%d-%m %H:%M:%S",  
    "%m/%d/%Y %H:%M:%S"
]

df['start_time'] = parse_dates(df['start_time'], date_formats)
df['completion_time'] = parse_dates(df['completion_time'], date_formats)
df['date'] = df['start_time'].dt.date
df["tb_treat_status"] = parse_dates(df['tb_treat_status'], ["%m.%d.%y", "%m/%d/%Y"]).dt.date

# preprocess test results
for col in ['tb_test_res', 'chest_xray_res']:
    df[col] = df[col].str.lower().str.strip()
df["tb_test_res"] = df["tb_test_res"].replace({
    "not detected": "negative",
    "detected(mtb+)": "positive",
    "detected": "positive",
    "not submitted": None,
    "3invalid/error": None
})

# determine tb_status
def determine_tb_status(row):
    if not pd.isna(row['tb_treat_status']):
        date_diff = (row['date'] - row['tb_treat_status']).days
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

# save
df.to_csv('data-clean/clinical/tb_cases.csv', index=False)