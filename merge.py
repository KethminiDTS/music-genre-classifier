import pandas as pd

COLS = ["artist_name", "track_name", "release_date", "genre", "lyrics"]


mendeley = pd.read_csv("data/mendeley.csv")
mendeley.columns = [c.lower().strip() for c in mendeley.columns]
mendeley = mendeley[COLS].dropna(subset=["lyrics", "genre"])
mendeley["genre"] = mendeley["genre"].str.lower().str.strip()


student = pd.read_csv("Student_dataset.csv")
student.columns = [c.lower().strip() for c in student.columns]
student = student[COLS].dropna(subset=["lyrics", "genre"])
student["genre"] = student["genre"].str.lower().str.strip()

# Make sure release_date is just the year
mendeley["release_date"] = mendeley["release_date"].astype(str).str[:4]
student["release_date"]  = student["release_date"].astype(str).str[:4]

print("\nMendeley genres:")
print(mendeley["genre"].value_counts())
print("\nStudent genres:")
print(student["genre"].value_counts())

merged = pd.concat([mendeley, student], ignore_index=True)
merged = merged[COLS]
merged.to_csv("Merged_dataset.csv", index=False)

print(f"Merged dataset: {len(merged)} total rows")
print("All genres:")
print(merged["genre"].value_counts())
print("Columns:", merged.columns.tolist())