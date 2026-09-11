import re
import pandas as pd
from sklearn.model_selection import train_test_split

# Clean text function (removes line breaks, extra spaces, and HTML tags)
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

    # Clean all text columns
    for col in df.columns:
        df[col] = df[col].apply(clean_data)

    # Input: user provides only the problem
    df["input_text"] = "legal problem: " + df["problem"]

    # Target: rest of the columns combined into a structured legal solution
    df["target_text"] = (
        "Law: " + df["law"] +
        " | Section: " + df["section"] +
        " | Risk Category: " + df["risk_category"] +
        " | Solution: " + df["solution"] +
        " | Legal Basis: " + df["solution_based_on_law"] +
        " | Applicable Conditions: " + df["applicable_conditions"] +
        " | Expected Answer: " + df["expected_answer"]
    )

    # Save full cleaned dataset
    full_path = "data/cleaned_legal_data.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved full cleaned dataset to {full_path} ({len(df)} rows)")

    # 90% train, 10% validation split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_path = "data/train_data.csv"
    val_path = "data/val_data.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Saved train data to {train_path} ({len(train_df)} rows)")
    print(f"Saved validation data to {val_path} ({len(val_df)} rows)")

if __name__ == "__main__":
    prepare_dataset()
