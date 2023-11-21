import pandas as pd

# Read the three CSV files into pandas DataFrames
df1 = pd.read_csv("outputGyro1.csv", header=None)
df2 = pd.read_csv("outputGyro2.csv", header=None)
df3 = pd.read_csv("outputGyro3.csv", header=None)

# Concatenate the DataFrames vertically
result_df = pd.concat([df1, df2, df3], ignore_index=True)

# Write the concatenated DataFrame to a new CSV file
result_df.to_csv("outputGyro.csv", index=False, header=False)
