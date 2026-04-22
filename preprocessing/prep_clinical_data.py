"""Preprocess raw clinical TB data from the questionnaire Excel file.

Parses dates, standardizes test results, and classifies each patient's TB
status (infectious, presumptive, or not infectious) based on Xpert MTB test,
chest X-ray, and treatment start date.

Reads from:  data-raw/clinical/complete-clinic-data edit 17.02.25.xlsx
Writes to:   data-clean/clinical/tb_cases.csv
"""

import pandas as pd


def parse_dates(date_series, formats):
    """Parse a Series of date strings, trying each format in order."""
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
            parsed_date = pd.to_datetime(date_str, errors='coerce')
        parsed_dates.append(parsed_date)
    return pd.Series(parsed_dates)


def determine_tb_status(row):
    """Classify a patient's TB status based on test results and treatment date.

    Classification hierarchy:
      1. If on TB treatment within 30 days of clinic visit -> 'infectious'
      2. If Xpert MTB test positive -> 'infectious'
      3. If chest X-ray positive -> 'infectious'
      4. Otherwise -> 'presumptive' (symptoms but no confirmed diagnosis)
    """
    if not pd.isna(row['tb_treat_status']):
        # Patients within 30 days of starting TB treatment are considered infectious
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


if __name__ == "__main__":
    file_path = 'data-raw/clinical/complete-clinic-data edit 17.02.25.xlsx'
    df = pd.read_excel(file_path, dtype=str)

    # Select and rename the columns
    df = df[["ID", "Start time", "Completion time", "2. Gender", "3. Age", "4. HIV status", "7. Date of confirmation (start of TB treatment)", "9.\xa0Xpert TB test 1 MTB result", "12. Chest X-ray result"]]
    df.columns = ["clinic_id", "start_time", "completion_time", "gender", "age", "hiv_status", "tb_treat_status", "tb_test_res", "chest_xray_res"]

    # Parse dates (clinical data entry used multiple date formats)
    date_formats = [
        "%Y-%d-%m %H:%M:%S",
        "%m/%d/%Y %H:%M:%S"
    ]

    df['start_time'] = parse_dates(df['start_time'], date_formats)
    df['completion_time'] = parse_dates(df['completion_time'], date_formats)
    df['date'] = df['start_time'].dt.date
    df["tb_treat_status"] = parse_dates(df['tb_treat_status'], ["%m.%d.%y", "%m/%d/%Y"]).dt.date

    # Standardize test results
    for col in ['tb_test_res', 'chest_xray_res']:
        df[col] = df[col].str.lower().str.strip()
    df["tb_test_res"] = df["tb_test_res"].replace({
        "not detected": "negative",
        "detected(mtb+)": "positive",
        "detected": "positive",
        "not submitted": None,
        "3invalid/error": None
    })

    df['tb_status'] = df.apply(determine_tb_status, axis=1)

    df.to_csv('data-clean/clinical/tb_cases.csv', index=False)
