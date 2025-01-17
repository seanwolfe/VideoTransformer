import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import yaml
from scipy.stats import linregress


# Function to correct predicted coordinates along the best-fit line
def correct_predicted(row):
    predicted = np.array(row['predicted_positions'])

    # Separate x and y coordinates (accounting for alternating structure)
    x_coords = predicted[0::2]  # Every even index for x
    y_coords = predicted[1::2]  # Every odd index for y

    # Fit a line to the (x, y) points
    slope, intercept, _, _, _ = linregress(x_coords, y_coords)

    # Project each point onto the line to find optimal x-coordinates
    projected_x = (x_coords + slope * (y_coords - intercept)) / (1 + slope ** 2)

    # Determine the optimal range for corrected x-coordinates
    x_min, x_max = projected_x.min(), projected_x.max()

    # Generate evenly spaced corrected x-coordinates within this range
    corrected_x = np.linspace(x_min, x_max, len(x_coords))

    # Compute the corresponding y-coordinates using the line equation
    corrected_y = slope * corrected_x + intercept

    # Recombine corrected (x, y) pairs into a single list
    corrected_pred = np.column_stack((corrected_x, corrected_y)).flatten().tolist()

    return corrected_pred


def str_to_tuple(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


def read_master(configuration, master_file_path):
    columns_to_convert = configuration['velocity_columns_to_convert']
    return pd.read_csv(master_file_path, sep=',', header=0,
                       converters={col: str_to_tuple for col in columns_to_convert},
                       names=configuration['velocity_columns'])


# Function to calculate MAE and Mean Error
def calculate_errors(row):
    # Convert predicted and label to arrays for comparison
    predicted = np.array(row['pred_corrected'])
    label = np.array(row['true_positions'])

    # Separate x and y for predicted and label
    predicted_x = predicted[0::2]
    predicted_y = predicted[1::2]
    label_x = label[0::2]
    label_y = label[1::2]

    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(predicted_x - label_x)) + np.mean(np.abs(predicted_y - label_y))

    # Calculate Mean Error (not absolute)
    mean_error = np.mean(predicted_x - label_x) + np.mean(predicted_y - label_y)

    return pd.Series([mae, mean_error])


# Function to compute mean error (not absolute)
def calculate_mean_error(row):
    predicted = np.array(row['predicted_positions'])
    label = np.array(row['true_positions'])
    mean_error = np.mean(predicted - label)
    return mean_error

# Function to compute mean error (not absolute)
def calculate_mean_error_xy(row):

    predicted = np.array(row['predicted_positions'])
    label = np.array(row['true_positions'])

    # Separate x and y for predicted and label
    predicted_x = predicted[0::2]
    predicted_y = predicted[1::2]
    label_x = label[0::2]
    label_y = label[1::2]

    mean_error_x = np.mean(predicted_x - label_x)
    mean_error_y = np.mean(predicted_y - label_y)
    return pd.Series([mean_error_x, mean_error_y])


# Function to calculate the final velocity (vx, vy)
def calculate_final_velocity(row, dt):
    predicted = np.array(row['predicted_positions'])  # or use 'label' for true positions

    # Separate the x and y coordinates
    x_coords = predicted[0::2]  # Even indices for x
    y_coords = predicted[1::2]  # Odd indices for y

    # Calculate total displacement from the first to the last frame
    delta_x = x_coords[-1] - x_coords[0]
    delta_y = y_coords[-1] - y_coords[0]

    # Total time elapsed (number of frames - 1) * time per frame (dt)
    total_time = (len(x_coords) - 1) * dt

    # Velocity components
    v_x = delta_x / total_time
    v_y = delta_y / total_time

    return v_x, v_y


def calculate_true_velocity(row, pixel_scale, image_size):
    omega = row['omega']  # Apparent motion in arcsec/hour
    theta = np.deg2rad(360 - row['theta'])  # Convert angle to radians

    # Convert omega to pixel units
    omega_pixels = omega * pixel_scale / 3600

    # Calculate true velocity components
    vx_true = omega_pixels * np.cos(theta)
    vy_true = omega_pixels * np.sin(theta)

    # Normalize by image size
    vx_norm = vx_true / image_size
    vy_norm = vy_true / image_size

    return vx_norm, vy_norm


# Function to compute velocity MAE and Mean Error
def calculate_velocity_errors(row):
    vx_pred, vy_pred = row['vx'], row['vy']
    vx_true, vy_true = row['vx_true'], row['vy_true']

    # Compute MAE
    vel_mae = (abs(vx_pred - vx_true) + abs(vy_pred - vy_true)) / 2

    # Compute Mean Error
    vel_meanerror = ((vx_pred - vx_true) + (vy_pred - vy_true)) / 2

    return vel_mae, vel_meanerror

with open('config_regressor.yaml', 'r') as f:
    config = yaml.safe_load(f)


df = read_master(config, config['output_file_name'])

# Apply the function to each row and create a new column
df[['mean_error_x', 'mean_error_y']] = df.apply(calculate_mean_error_xy, axis=1)

# Apply the function to each row in the DataFrame
#df['pred_corrected'] = df.apply(correct_predicted, axis=1)

# Apply error calculation
#df[['pos_mae_corr', 'pos_meanerror_corr']] = df.apply(calculate_errors, axis=1)

# Example usage: Apply final velocity calculation assuming time between frames is known
#dt = config['dt']  # Example: Time between frames is 1 unit (adjust as needed)

#df[['vx', 'vy']] = df.apply(lambda row: calculate_final_velocity(row, dt), axis=1, result_type="expand")

# Parameters (adjust as needed)
#pixel_scale = config['pixel_scale']  # Example: arcsec/pixel (adjust based on your setup)
#image_size = config['image_size']   # Size of the image in pixels

# Apply the function to each row
#df[['vx_true', 'vy_true']] = df.apply(lambda row: calculate_true_velocity(row, pixel_scale, image_size), axis=1, result_type='expand')

# Apply the function to each row
#df[['vel_mae', 'vel_meanerror']] = df.apply(calculate_velocity_errors, axis=1, result_type='expand')

df.to_csv(config['output_file_name'], sep=',', header=True, index=False)

# visualize line
# for idx, row in df.iterrows():
#
#     size = 224
#     x_pred = np.array(row['predicted_positions'][0::2]) * size
#     y_pred = np.array(row['predicted_positions'][1::2]) * size
#     x_corr = np.array(row['pred_corrected'][0::2]) * size
#     y_corr = np.array(row['pred_corrected'][1::2]) * size
#     fig = plt.figure()
#     plt.scatter(x_pred, y_pred)
#     plt.plot(x_corr, y_corr)
#     plt.show()