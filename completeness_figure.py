import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data (20 rows for Magnitude, 16 columns for Sky motion)
np.random.seed(42)
data = np.random.randint(0, 101, size=(20, 16))

# Convert the data to a DataFrame
magnitude_values = np.linspace(26, 20, data.shape[0])  # From 26.0 to 20.0
sky_motion_values = np.arange(1, data.shape[1] * 4 + 1, 4)  # From 1 to 80 in steps of 4

df = pd.DataFrame(data, index=np.round(magnitude_values, 2), columns=sky_motion_values)

# Create the heatmap using seaborn
plt.figure(figsize=(12, 8))
ax = sns.heatmap(df, annot=True, cmap='Purples', cbar_kws={'label': 'Completeness [%]'},
                 linewidths=0.5, linecolor='white', fmt='d')

# Customize the plot
ax.set_xlabel('Sky motion [arcsec h⁻¹]')
ax.set_ylabel('Magnitude')
ax.invert_yaxis()  # Match the original order of Magnitude
ax.set_title('Heatmap of Completeness')

# Show the plot
plt.tight_layout()
plt.show()