import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("network_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Count of faults
print("\nFault Count:")
print(df["fault"].value_counts())

# Plot CPU usage vs Fault
plt.scatter(df["cpu_usage"], df["fault"])
plt.xlabel("CPU Usage")
plt.ylabel("Fault")
plt.title("CPU Usage vs Fault")
plt.show()

# Plot Latency vs Fault
plt.scatter(df["latency"], df["fault"])
plt.xlabel("Latency")
plt.ylabel("Fault")
plt.title("Latency vs Fault")
plt.show()