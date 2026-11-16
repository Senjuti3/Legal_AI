import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip()

def prepare_dataset(csv_path="data/legal_problem_solution_dataset.csv"):
    print(f"Loading raw dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean the input problem
    df["input_text"] = df["problem"].apply(clean_data)

    # Format full legal solution with all fields separated by ' / '
    df["target_text"] = (
        "Law: " + df["law"].apply(clean_data) +
        " / Section: " + df["section"].apply(clean_data) +
        " / Risk Category: " + df["risk_category"].apply(clean_data) +
        " / Solution: " + df["solution"].apply(clean_data) +
        " / Legal Basis: " + df["solution_based_on_law"].apply(clean_data) +
        " / Applicable Conditions: " + df["applicable_conditions"].apply(clean_data) +
        " / Expected Answer: " + df["expected_answer"].apply(clean_data)
    )

    print("Sample Input:\n", df["input_text"].iloc[0])
    print("\nSample Target (with Law, Section, Risk Category separated by ' / '):\n", df["target_text"].iloc[0][:200])

    # Save full cleaned dataset
    full_path = "data/cleaned_legal_data.csv"
    try:
        df.to_csv(full_path, index=False)
        print(f"Saved full cleaned dataset to {full_path} ({len(df)} rows)")
    except PermissionError:
        full_path = "data/cleaned_legal_data_new.csv"
        df.to_csv(full_path, index=False)
        print(f"Saved full cleaned dataset to {full_path} ({len(df)} rows)")

    # 90% train, 10% validation split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_path = "data/train_data.csv"
    try:
        train_df.to_csv(train_path, index=False)
        print(f"Saved train data to {train_path} ({len(train_df)} rows)")
    except PermissionError:
        train_path = "data/train_data_split.csv"
        train_df.to_csv(train_path, index=False)
        print(f"Saved train data to {train_path} ({len(train_df)} rows)")

    val_path = "data/val_data.csv"
    try:
        val_df.to_csv(val_path, index=False)
        print(f"Saved validation data to {val_path} ({len(val_df)} rows)")
    except PermissionError:
        val_path = "data/val_data_split.csv"
        val_df.to_csv(val_path, index=False)
        print(f"Note: Saved to {val_path} ({len(val_df)} rows)")

if __name__ == "__main__":
    prepare_dataset()
