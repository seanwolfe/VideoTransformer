import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml

with open('config_regressor.yaml', 'r') as f:
    config = yaml.safe_load(f)

file_path = 'inference_results_completeness_reg.csv'  # Replace with your file path if different
data = pd.read_csv(file_path)
#new_data = data[(data['omega'] >= 658) & (data['omega'] <= 789) & (data['V'] >= 16.7) & (data['V'] <= 17.1)]
new_data = data[(data['position_MAE'] * 224 < 8)]
new_data.loc[:, ['mean_error_y']] = new_data['mean_error_y'] * config['image_size'] * config['pixel_scale'] * 1000

figure = plt.figure()
print(np.mean(new_data['mean_error_y']))
print(np.std(new_data['mean_error_y']))
plt.hist(new_data['mean_error_y'], bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Astrometric Residual (mas)')
plt.ylabel('Count')
plt.show()