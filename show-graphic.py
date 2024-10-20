import pandas as pd
import matplotlib.pyplot as plt

# Load the Pokémon dataset
df = pd.read_csv('pokemon.csv')

# Select only numeric columns for mean calculation
numeric_columns = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']

# Group by the 'type1' (primary type) and calculate the mean of numeric stats
mean_stats = df.groupby('type1')[numeric_columns].mean()

# Plotting the mean stats for each primary type
fig, ax = plt.subplots(figsize=(12, 6))

# Create a bar plot for each stat
mean_stats.plot(kind='bar', ax=ax)

# Set title and labels
ax.set_title('Mean Stats by Pokémon Type (Primary Type)', fontsize=16)
ax.set_xlabel('Primary Type', fontsize=12)
ax.set_ylabel('Mean Value', fontsize=12)
ax.legend(title='Stats', bbox_to_anchor=(1.05, 1), loc='upper left')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
