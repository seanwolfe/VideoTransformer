import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

which = 'regressor'


if 'classifier' in which:

    ##################################################
    # Accuracy heatmap for classifier
    #################################################

    # Load the CSV file
    file_path = 'inference_results_completeness.csv'  # Replace with your file path if different
    data = pd.read_csv(file_path)
    data = data[data['V'] < 23]

    # Add a "correct" column to identify true positives (predicted == true == 1)
    data['correct'] = (data['predicted'] == 0) & (data['true'] == 0)

    # Define bin edges for both V (magnitude) and Omega (sky motion)
    v_bins = np.linspace(data['V'].min(), data['V'].max(), 20)  # 10 bins for V
    omega_bins = np.linspace(data['Omega'].min(), data['Omega'].max(), 20)  # 10 bins for Omega
    print(omega_bins)

    # Digitize the data to assign it to bins
    data['v_bin'] = np.digitize(data['V'], v_bins[1:-1], right=True)
    data['omega_bin'] = np.digitize(data['Omega'], omega_bins[1:-1], right=True)
    print(data['v_bin'])
    print(data['omega_bin'])

    # Group by the bins and calculate accuracy for each bin
    binned_data = data.groupby(['v_bin', 'omega_bin']).agg(
        total=('correct', 'size'),
        correct=('correct', 'sum')
    ).reset_index()
    print(binned_data)
    binned_data['accuracy'] = (binned_data['correct'] / binned_data['total']) * 100

    # Pivot the data to create a matrix for the heatmap
    heatmap_matrix = binned_data.pivot(index='v_bin', columns='omega_bin', values='accuracy')

    # Replace bin indices with bin midpoints for better readability
    # heatmap_matrix.index = [f"{(v_bins[i-1] + v_bins[i]) / 2:.2f}" for i in heatmap_matrix.index]
    # heatmap_matrix.columns = [f"{(omega_bins[i-1] + omega_bins[i]) / 2:.2f}" for i in heatmap_matrix.columns]

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(heatmap_matrix, annot=True, cmap='Purples', cbar_kws={'label': 'Accuracy [%]'},
                     linewidths=0.5, fmt='.1f', linecolor='white')

    ax.set_xticks(np.arange(len(omega_bins)))  # Shift to match heatmap grid
    ax.set_yticks(np.arange(len(v_bins)))
    ax.set_xticklabels([f"{edge:.0f}" for edge in omega_bins])  # Format tick labels
    ax.set_yticklabels([f"{edge:.1f}" for edge in v_bins])

    # Customize the plot
    ax.set_xlabel('Sky motion (arcsec/h) [Omega]')
    ax.set_ylabel('Magnitude [V]')
    ax.invert_yaxis()  # Match the typical order of magnitudes

    # Display the plot
    plt.tight_layout()
    plt.show()

elif 'regressor' in which:
    #################################################
    # MAE heatmap for regressor, position
    ################################################

    # Load the CSV file
    file_path = 'inference_results_completeness_reg.csv'  # Replace with your file path if different
    data = pd.read_csv(file_path)
    data = data[data['V'] < 23]

    # Define bin edges for both V (magnitude) and Omega (sky motion)
    v_bins = np.linspace(data['V'].min(), data['V'].max(), 20)  # 10 bins for V
    omega_bins = np.linspace(data['omega'].min(), data['omega'].max(), 20)  # 10 bins for Omega

    # Digitize the data to assign it to bins
    data['v_bin'] = np.digitize(data['V'], v_bins[1:-1], right=True)
    data['omega_bin'] = np.digitize(data['omega'], omega_bins[1:-1], right=True)
    print(data['v_bin'])
    print(data['omega_bin'])

    # Group by the bins and calculate accuracy for each bin
    binned_data = data.groupby(['v_bin', 'omega_bin']).agg(
        pos_mae=('position_MAE', 'mean')
    ).reset_index()
    binned_data['pos_mae'] = binned_data['pos_mae'] * 224

    # Pivot the data to create a matrix for the heatmap
    heatmap_matrix = binned_data.pivot(index='v_bin', columns='omega_bin', values='pos_mae')

    # Replace bin indices with bin midpoints for better readability
    # heatmap_matrix.index = [f"{(v_bins[i-1] + v_bins[i]) / 2:.2f}" for i in heatmap_matrix.index]
    # heatmap_matrix.columns = [f"{(omega_bins[i-1] + omega_bins[i]) / 2:.2f}" for i in heatmap_matrix.columns]

    norm = mcolors.Normalize(vmax=8)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(heatmap_matrix, annot=True, cmap='Purples_r', norm=norm, cbar_kws={'label': 'Position MAE [pixels]'},
                     linewidths=0.5, fmt='.1f', linecolor='white')

    ax.set_xticks(np.arange(len(omega_bins)))  # Shift to match heatmap grid
    ax.set_yticks(np.arange(len(v_bins)))
    ax.set_xticklabels([f"{edge:.0f}" for edge in omega_bins])  # Format tick labels
    ax.set_yticklabels([f"{edge:.1f}" for edge in v_bins])

    # Customize the plot
    ax.set_xlabel('Sky motion (arcsec/h) [Omega]')
    ax.set_ylabel('Magnitude [V]')
    ax.invert_yaxis()  # Match the typical order of magnitudes

    # Display the plot
    plt.tight_layout()
    plt.show()

elif 'velocity' in which:
    ###############################################
    # MAE heatmap for regressor, velocity
    ###############################################
    # Load the CSV file
    file_path = 'inference_results_completeness_reg.csv'  # Replace with your file path if different
    data = pd.read_csv(file_path)
    #data = data[data['V'] < 23]

    # Define bin edges for both V (magnitude) and Omega (sky motion)
    v_bins = np.linspace(data['V'].min(), data['V'].max(), 20)  # 10 bins for V
    omega_bins = np.linspace(data['omega'].min(), data['omega'].max(), 20)  # 10 bins for Omega

    # Digitize the data to assign it to bins
    data['v_bin'] = np.digitize(data['V'], v_bins[1:-1], right=True)
    data['omega_bin'] = np.digitize(data['omega'], omega_bins[1:-1], right=True)
    print(data['v_bin'])
    print(data['omega_bin'])

    # Group by the bins and calculate accuracy for each bin
    binned_data = data.groupby(['v_bin', 'omega_bin']).agg(
        vel_mae=('vel_mae', 'mean')
    ).reset_index()
    binned_data['vel_mae'] = binned_data['vel_mae'] * 224 / 0.66 * 3600

    # Pivot the data to create a matrix for the heatmap
    heatmap_matrix = binned_data.pivot(index='v_bin', columns='omega_bin', values='vel_mae')

    # Replace bin indices with bin midpoints for better readability
    # heatmap_matrix.index = [f"{(v_bins[i-1] + v_bins[i]) / 2:.2f}" for i in heatmap_matrix.index]
    # heatmap_matrix.columns = [f"{(omega_bins[i-1] + omega_bins[i]) / 2:.2f}" for i in heatmap_matrix.columns]

    #norm = mcolors.Normalize(vmax=6)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(heatmap_matrix, annot=True, cmap='Purples', cbar_kws={'label': 'Velocity MAE [pixels]'},
                     linewidths=0.5, fmt='.1f', linecolor='white')

    ax.set_xticks(np.arange(len(omega_bins)))  # Shift to match heatmap grid
    ax.set_yticks(np.arange(len(v_bins)))
    ax.set_xticklabels([f"{edge:.0f}" for edge in omega_bins])  # Format tick labels
    ax.set_yticklabels([f"{edge:.1f}" for edge in v_bins])

    # Customize the plot
    ax.set_xlabel('Sky motion (arcsec/h) [Omega]')
    ax.set_ylabel('Magnitude [V]')
    ax.invert_yaxis()  # Match the typical order of magnitudes

    # Display the plot
    plt.tight_layout()
    plt.show()


else:
    # mean snr
    # Load the CSV file
    file_path = 'inference_results_completeness_reg.csv'  # Replace with your file path if different
    data = pd.read_csv(file_path)
    data = data[data['V'] < 23]

    # Define bin edges for both V (magnitude) and Omega (sky motion)
    v_bins = np.linspace(data['V'].min(), data['V'].max(), 20)  # 10 bins for V
    omega_bins = np.linspace(data['Omega'].min(), data['Omega'].max(), 20)  # 10 bins for Omega

    # Digitize the data to assign it to bins
    data['v_bin'] = np.digitize(data['V'], v_bins[1:-1], right=True)
    data['omega_bin'] = np.digitize(data['Omega'], omega_bins[1:-1], right=True)

    # Group by the bins and calculate accuracy for each bin
    binned_data = data.groupby(['v_bin', 'omega_bin']).agg(
        SNR=('SNR', 'mean')
    ).reset_index()

    # Pivot the data to create a matrix for the heatmap
    heatmap_matrix = binned_data.pivot(index='v_bin', columns='omega_bin', values='SNR')

    # Replace bin indices with bin midpoints for better readability
    # heatmap_matrix.index = [f"{(v_bins[i-1] + v_bins[i]) / 2:.2f}" for i in heatmap_matrix.index]
    # heatmap_matrix.columns = [f"{(omega_bins[i-1] + omega_bins[i]) / 2:.2f}" for i in heatmap_matrix.columns]

    norm = mcolors.Normalize(vmax=6)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(heatmap_matrix, annot=True, cmap='Purples', norm=norm, cbar_kws={'label': 'Mean SNR'},
                     linewidths=0.5, fmt='.1f', linecolor='white')

    ax.set_xticks(np.arange(len(omega_bins)))  # Shift to match heatmap grid
    ax.set_yticks(np.arange(len(v_bins)))
    ax.set_xticklabels([f"{edge:.0f}" for edge in omega_bins])  # Format tick labels
    ax.set_yticklabels([f"{edge:.1f}" for edge in v_bins])

    # Customize the plot
    ax.set_xlabel('Sky motion (arcsec/h) [Omega]')
    ax.set_ylabel('Magnitude [V]')
    ax.invert_yaxis()  # Match the typical order of magnitudes

    # Display the plot
    plt.tight_layout()
    plt.show()