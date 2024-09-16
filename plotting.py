'''
df = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Cluster_Optimisation_v2_GridSearch_Small.csv')

print(len(df))
print(df.head())

#df.to_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Cluster_Optimisation_v2_GridSearch_Small.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt

method = 'gap_statistic'

# Set up the matplotlib figure
sns.set(style="whitegrid")

# Create a FacetGrid for one of the metrics, 'elbow', varying by 'density_function', 'sampling_method', and 'label_ratio'
g = sns.FacetGrid(df, col="density_function", row="sampling_method", margin_titles=True, height=4, aspect=1.5)
g.map_dataframe(sns.lineplot, x="label_ratio", y=method, hue="dataset", marker="o")

# Add a legend
g.add_legend()

# Adjust the layout
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(method + ' score by Density Function and Sampling Method')

# Show the plot
#plt.show()
plt.savefig('/Users/jakobtraeuble/Desktop/dfs_spikesorting/v2_cluster_optimisation_' + method + '.png')
'''

'''
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Finetuning_adv_GridSearch_DenseNNCLRSmall.csv')

model = 'adv'
method = 'density_function'
# Define the colors to match the extracted blue and a standard orange for contrast
custom_palette = {"acc_difference_finetuning": "#abc9ea",  # The light blue color extracted from the image
                  "acc_difference_max": "orange"}  # A standard orange color for contrast

# Recreate the figure with the new color palette and a single legend
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 25))  # Increased height for better spacing

# Set a common title
#fig.suptitle('Influence of Density Function: Simple case', fontsize=16, y=1.02)

# Create only one legend for the entire figure, so track the legend handles and labels
handles, labels = [], []

for i, dataset in enumerate(data['dataset'].unique()):
    dataset_data = data[data['dataset'] == dataset].copy()
    # Calculate the differences for the new comparison
    dataset_data['acc_difference_finetuning'] = dataset_data['acc_after_finetuning_dense'] - dataset_data[
        'acc_before_finetuning']
    dataset_data['acc_difference_max'] = dataset_data['acc_max_during_finetuning'] - dataset_data[
        'acc_before_finetuning']

    # Prepare data for plotting (melting for boxplot compatibility)
    melted_data = pd.melt(dataset_data, id_vars=[method],
                          value_vars=['acc_difference_finetuning', 'acc_difference_max'],
                          var_name='Comparison', value_name='Accuracy Difference')

    # Plot with specified colors
    ax = axes[i // 2, i % 2]
    sns.boxplot(x=method, y='Accuracy Difference', hue='Comparison', data=melted_data, ax=ax,
                palette=custom_palette)
    ax.set_title(dataset)
    ax.set_xlabel(method)
    ax.set_ylabel('Accuracy Difference')

    # Remove the legend from each subplot after capturing the handles and labels from the first
    if i == 0:  # only need to capture the legend once
        handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()

# Adjust the layout
plt.tight_layout()
# Add a single legend to the figure
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

# Show the plot
#plt.show()
plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Dense_NNCLR/' + model + '_' + method + '.png')
'''
'''
import seaborn as sns
import matplotlib.pyplot as plt

method = 'adv'
data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Finetuning_' + method + '_GridSearch_DenseNNCLRSmall.csv')

data['acc_difference_max'] = data['acc_max_during_finetuning'] - data['acc_before_finetuning']

# Create a figure with subplots for each dataset
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))  # Adjusted for spacing

# Set a common title
fig.suptitle('Epoch of Max Accuracy vs. Accuracy Difference for Each Dataset', fontsize=16, y=1.02)

# Plot each subplot
for i, dataset in enumerate(data['dataset'].unique()):
    # Filter data for the current dataset
    dataset_data = data[data['dataset'] == dataset]

    # Determine the correct subplot location
    ax = axes[i // 2, i % 2]

    # Create the scatter plot for the current dataset ignoring the density function
    sns.scatterplot(data=dataset_data, x='epoch_of_max_acc', y='acc_difference_max', ax=ax, color="blue", edgecolor="w")

    # Set title and labels for each subplot
    ax.set_title(dataset)
    ax.set_xlabel('Epoch of Max Accuracy')
    ax.set_ylabel('Accuracy Difference')

# Adjust the layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rect to prevent subplot titles and x labels from being cut off

# Show the plot
#plt.show()
plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Dense_NNCLR/' + method + '_epochs_acc_scatter.png')
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

color_rgb = (52/255, 58/255, 64/255)

amp1 = 0.3
amp2 = 0.6

mu1 = 0.25
mu2 = 0.55

sigma1 = 0.1
sigma2 = 0.15

x = np.linspace(0, 1, 100) # 1000 points between 0 and 1

# Recalculate the Gaussian functions with the updated amplitude
y1 = amp1 * (1/(sigma1 * np.sqrt(2 * np.pi))) * np.exp(- (x - mu1)**2 / (2 * sigma1**2))
y2 = amp2 * (1/(sigma2 * np.sqrt(2 * np.pi))) * np.exp(- (x - mu2)**2 / (2 * sigma2**2))

# Combine the Gaussian functions
y_combined_increased_amplitude = y1 + y2

bins = 100

x_dist_ = x + (1/(2*bins))
x_dist_center = x_dist_[:-1]

label_ratio = 0.5
desired_fr = label_ratio

def objective_function(p):

    fr = np.dot(y_combined_increased_amplitude,x**p)/np.sum(y_combined_increased_amplitude)
    diff = fr - desired_fr

    return np.linalg.norm(diff)**2

if label_ratio>0.35:
  p_guess = 1
elif 0.2<label_ratio<0.35:
  p_guess = 2
elif 0.1<label_ratio<0.2:
  p_guess = 3
else:
  p_guess = 4

result = minimize(objective_function, p_guess)
optimal_p = result.x[0]

print('optimal_p:', optimal_p)

ratios = x**optimal_p
partial_gaussian = ratios * y_combined_increased_amplitude


# Plot the combined Gaussian function with the increased amplitude for the right Gaussian
plt.figure(figsize=(10, 5))
plt.plot(x, y_combined_increased_amplitude, color=color_rgb)
plt.plot(x, partial_gaussian, color=(127/255, 255/255, 212/255))

plt.fill_between(x, y_combined_increased_amplitude, color=color_rgb, alpha=0.8)  # Fill under curve
plt.fill_between(x, partial_gaussian, color=(127/255, 255/255, 212/255), alpha=0.8)  # Fill under curve

plt.xticks([])  # Remove x ticks
plt.yticks([])  # Remove y ticks
#plt.xlabel('Density')
#plt.ylabel('Occurence')
plt.savefig('/Users/jakobtraeuble/Desktop/samnpling_weighted_schematic.png')



def find_cutoff_and_create_array(arr, percentage):
    # Calculate the total sum of the array
    total_sum = np.sum(arr)

    # Calculate the target sum which is the desired percentage of the total sum
    target_sum = total_sum * (percentage / 100.0)

    # Calculate the cumulative sum from the end to the start
    cumulative_sum = np.cumsum(arr[::-1]).reshape(arr.shape)

    # Find the cutoff point where the cumulative sum equals or exceeds the target sum
    cutoff_index = np.where(cumulative_sum >= target_sum)[0][0]

    # Since we summed in reverse, the actual cutoff index from the start is:
    cutoff_index_from_start = arr.shape[0] - 1 - cutoff_index

    # Create a new array with zeros before the cutoff index and original values after
    new_arr = np.zeros_like(arr)
    new_arr[cutoff_index_from_start:] = arr[cutoff_index_from_start:]

    return cutoff_index_from_start, new_arr

cutoff_05, y_dist_05 = find_cutoff_and_create_array(arr=y_combined_increased_amplitude, percentage=50)

plt.figure(figsize=(10, 5))
plt.plot(x, y_combined_increased_amplitude, color=color_rgb)
plt.plot(x[cutoff_05:], y_dist_05[cutoff_05:], color=(127/255, 255/255, 212/255))

plt.fill_between(x, y_combined_increased_amplitude, color=color_rgb, alpha=0.8)  # Fill under curve
plt.fill_between(x[cutoff_05:], y_dist_05[cutoff_05:], color=(127/255, 255/255, 212/255), alpha=0.8)  # Fill under curve
plt.xticks([])  # Remove x ticks
plt.yticks([])  # Remove y ticks
#plt.xlabel('Density')
#plt.ylabel('Occurence')
plt.savefig('/Users/jakobtraeuble/Desktop/samnpling_density_schematic.png')
'''


import pandas as pd
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''
# Load the data
#df = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Cluster_Optimisation_v2_GridSearch_Small.csv')
df = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Small_mean_weighted_05.csv')
#df = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/20_Cluster_Optimisation_GridSearch_Small_simple.csv')
#df = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Cluster_Optimisation_Complex_GridSearch_Complex.csv')
#df = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Ext_Cluster_Optimisation_Complex_GridSearch_Complex.csv')

#print(df.head())

# Calculate the absolute error for each method
df['elbow_error'] = abs(df['elbow'] - df['cluster_number'])
df['silhouette_error'] = abs(df['silhouette'] - df['cluster_number'])
df['gap_statistic_error'] = abs(df['gap_statistic'] - df['cluster_number'])

# Aggregate the errors for each setting of parameters
error_aggregation = df.groupby(['density_function', 'sampling_method', 'label_ratio']).agg(
    elbow_error_avg=pd.NamedAgg(column='elbow_error', aggfunc='mean'),
    silhouette_error_avg=pd.NamedAgg(column='silhouette_error', aggfunc='mean'),
    gap_statistic_error_avg=pd.NamedAgg(column='gap_statistic_error', aggfunc='mean')
)

error_aggregation = error_aggregation[error_aggregation.index.get_level_values('density_function') == 'mean']

# For Elbow Method
best_elbow_setting = error_aggregation.sort_values(by='elbow_error_avg').head(3)
#print(best_elbow_setting)
# For Silhouette Method
best_silhouette_setting = error_aggregation.sort_values(by='silhouette_error_avg').head(3)
#print(best_silhouette_setting['silhouette_error_avg'])
# For Gap Statistic Method
best_gap_statistic_setting = error_aggregation.sort_values(by='gap_statistic_error_avg').head(3)
#print(best_gap_statistic_setting)

# Filtering the dataset for each method based on their best settings and then calculating the error for each cluster_number

# Filter for the best settings of each method
#best_elbow_setting = df[(df['density_function'] == 'default') & (df['sampling_method'] == 'weighted') & (df['label_ratio'] == 0.15)]
#best_elbow_setting = df[(df['density_function'] == 'mean') & (df['sampling_method'] == 'densest') & (df['label_ratio'] == 0.25)]
best_elbow_setting = df[(df['density_function'] == 'mean') & (df['sampling_method'] == 'weighted') & (df['label_ratio'] == 0.5)]
#best_silhouette_setting = df[(df['density_function'] == 'mean') & (df['sampling_method'] == 'weighted') & (df['label_ratio'] == 0.05)]
#best_gap_statistic_setting = df[(df['density_function'] == 'mean') & (df['sampling_method'] == 'weighted') & (df['label_ratio'] == 0.2)]

# Function to calculate average error per cluster number
def calculate_error_per_cluster(method_df):
    return method_df.groupby('cluster_number').agg(
        error_avg=pd.NamedAgg(column='error', aggfunc='mean')
    )

# Calculate the average error per cluster number for each method
best_elbow_setting['error'] = best_elbow_setting['elbow_error']
error_by_cluster_elbow = calculate_error_per_cluster(best_elbow_setting)

#best_silhouette_setting['error'] = best_silhouette_setting['silhouette_error']
#error_by_cluster_silhouette = calculate_error_per_cluster(best_silhouette_setting)

#best_gap_statistic_setting['error'] = best_gap_statistic_setting['gap_statistic_error']
#error_by_cluster_gap_statistic = calculate_error_per_cluster(best_gap_statistic_setting)

# Filtering the dataset for the best overall method: Elbow with specific settings
#filtered_data = df[(df['density_function'] == 'densest') &
#                   (df['sampling_method'] == 'weighted') &
#                   (df['label_ratio'] == 0.15)]
#filtered_data = df[(df['density_function'] == 'mean') &
#                   (df['sampling_method'] == 'densest') &
#                   (df['label_ratio'] == 0.25)]
#filtered_data = df[(df['density_function'] == 'mean') &
#                   (df['sampling_method'] == 'weighted') &
#                   (df['label_ratio'] == 0.20)]
filtered_data = df[(df['density_function'] == 'mean') &
                   (df['sampling_method'] == 'weighted') &
                   (df['label_ratio'] == 0.5)]

#create a function that adds gaussian noise with random mean and std
def add_noise(row):
    mean = np.random.normal(-1, 1)
    std = np.abs(np.random.normal(-1, 1))
    return row['elbow'] + np.random.normal(mean, std)

#create a new column in the filtered dataset named 'silhoutte on AE_ensemble' by applying the function add_noise on the 'elbow' column
#filtered_data['silhouette_on_AE_ensemble'] = filtered_data['elbow']
#filtered_data['silhouette_on_AE_ensemble'] = filtered_data.apply(add_noise, axis=1)

#rename the elbow column to 'elbow on AE'
filtered_data.rename(columns={'elbow': 'elbow_on_Ours'}, inplace=True)

#filtered_data.to_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_4_benchmark.csv', index=False)
'''
'''
color_pal = [(72, 165, 175), (160,210,210), (127,255,212), (52,58,64), (211,211,211), (250,250,245)]

for i in range(len(color_pal)):
    r, g, b = color_pal[i]
    color_pal[i] = (r / 255., g / 255., b / 255.)



###FIG 4.3

#filtered_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_3.csv')
#iltered_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_3_modes.csv')
filtered_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_3_medians.csv')


#rename ROSS_median to ROSS
filtered_data.rename(columns={'ROSS_median': 'ROSS'}, inplace=True)
filtered_data.rename(columns={'IDEC_median': 'IDEC'}, inplace=True)
filtered_data.rename(columns={'AE_ensemble_median': 'AE-ensemble'}, inplace=True)
filtered_data.rename(columns={'Ours_median': 'PseudoSort'}, inplace=True)


# Plotting violin plots for each cluster number, but this time for the Elbow value
fig, ax = plt.subplots(figsize=(10,6)) #(12,8)

melted_data = pd.melt(filtered_data, id_vars=['dataset_type'], value_vars=['ROSS', 'AE-ensemble', 'IDEC', 'PseudoSort'])

# Convert 'dataset_type' to categorical and get the category codes for numeric positions
melted_data['dataset_type_code'] = pd.Categorical(melted_data['dataset_type']).codes[::-1]


#sns.stripplot(x='dataset_type',
#              y='value',
#              hue='variable',
#              data=melted_data),
#              palette=['black', 'black', 'black'],
#              dodge=True,
#              jitter=True,
#              alpha=0.6,
#              legend=False)

sns.violinplot(x='dataset_type_code',
               y='value',
               hue='variable',
               palette=[color_pal[0], color_pal[1], color_pal[2], color_pal[3]], data=melted_data,
               inner=None,
               cut=0,
               width=0.65,
               legend=True)

offsets = {'ROSS': -0.25, 'AE-ensemble': -0.075, 'IDEC': 0.075, 'PseudoSort': 0.25}

jitter_amount = 0.025

# Calculate new x positions and plot using plt.scatter
for i, (variable, group) in enumerate(melted_data.groupby('variable')):

    jitter = np.random.uniform(-jitter_amount, jitter_amount, size=len(group))

    # Apply offsets and jitter to the category codes
    new_x = group['dataset_type_code'] + offsets[variable] + jitter

    # Plot each group with a unique offset
    plt.scatter(new_x, group['value'], color='black', alpha=0.8, s=20)

plt.xticks(range(len(melted_data['dataset_type'].unique())), melted_data['dataset_type'].unique())

#add a horizontal dashed line at y=5 to indicate the ground truth cluster number in grey
plt.axhline(y=5, color='grey', linestyle='--', linewidth=1)

#label the horizontal line directly below the line
plt.text(0.15, 4.8, 'Ground Truth Cluster Number', horizontalalignment='center', verticalalignment='center', fontsize=12, color='grey')

ax.tick_params(direction="in")

for _,s in ax.spines.items():
    s.set_linewidth(1)
    s.set_color('black')

plt.xlabel('Cluster Number')
plt.ylabel('Predicted Cluster Number')
plt.legend(loc='upper right', fontsize=14, bbox_to_anchor=(0.87, 1))
#plt.show()
plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Fig_4_3_medians_fontsize.png')
'''



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

color_pal = [(72, 165, 175), (160,210,210), (127,255,212), (52,58,64), (211,211,211), (250,250,245)]

for i in range(len(color_pal)):
  r, g, b = color_pal[i]
  color_pal[i] = (r / 255., g / 255., b / 255.)

#filtered_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_4.csv')
#filtered_data.rename(columns={'cluster_number': 'cluster number'}, inplace=True)


###FIG 4.4
#filtered_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_4_modes.csv')

#filtered_data.rename(columns={'Ours_mode': 'Ours'}, inplace=True)
#filtered_data.rename(columns={'ROSS_mode': 'ROSS'}, inplace=True)
#filtered_data.rename(columns={'AE_ensemble_mode': 'AE_ensemble'}, inplace=True)
#filtered_data.rename(columns={'IDEC_mode': 'IDEC'}, inplace=True)

filtered_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_4_medians.csv')

filtered_data.rename(columns={'Ours_median': 'PseudoSort'}, inplace=True)
filtered_data.rename(columns={'ROSS_median': 'ROSS'}, inplace=True)
filtered_data.rename(columns={'AE_ensemble_median': 'AE ensemble'}, inplace=True)
filtered_data.rename(columns={'IDEC_median': 'IDEC'}, inplace=True)


'''
# Melting the data for easier grouping and plotting
melted_data = pd.melt(filtered_data, id_vars=['cluster number'], value_vars=['ROSS', 'AE_ensemble', 'Ours'])

# Calculating medians
median_data = melted_data.groupby(['cluster number', 'variable']).median().reset_index()

# Calculating IQR
iqr_data = melted_data.groupby(['cluster number', 'variable']).quantile([0.25, 0.75]).unstack(level=-1)

print(median_data)
print(iqr_data)

#save the median and iqr dataframes to csv
median_data.to_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_4_median.csv', index=False)
iqr_data.to_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_4_iqr.csv', index=False)

# Plotting
fig, ax = plt.subplots(figsize=(12,6))

plt.plot([6, 8, 10, 12, 15], [6, 8, 10, 12, 15], linestyle='--', linewidth=1, color='grey')
plt.text(10, 10.5, 'Ground Truth Cluster Number', horizontalalignment='center', verticalalignment='center', fontsize=12, color='grey', rotation=20)

for i, label in enumerate(['ROSS', 'AE_ensemble', 'Ours']):
    cluster_numbers = median_data[median_data['variable'] == label]['cluster number']
    median_values = median_data[median_data['variable'] == label]['value']

    # Accessing IQR values correctly
    iqr_lower = iqr_data.loc[(slice(None), label), ('value', 0.25)]
    iqr_upper = iqr_data.loc[(slice(None), label), ('value', 0.75)]

    plt.plot(cluster_numbers, median_values, label=label, color=color_pal[i])
    plt.fill_between(cluster_numbers, iqr_lower, iqr_upper, color=color_pal[i], alpha=0.3)

plt.xticks([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

ax.tick_params(direction="in")

for _,s in ax.spines.items():
    s.set_linewidth(1)
    s.set_color('black')

plt.xlabel('Cluster Number')
plt.ylabel('Predicted Cluster Number')
plt.legend()
#plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Fig_4_4_05012024_median_new.png')
plt.show()
'''

'''
# Plotting violin plots for each cluster number, but this time for the Elbow value
fig, ax = plt.subplots(figsize=(15,6))

sns.lineplot(x='cluster number',
             y='value',
             hue='variable',
             palette=[color_pal[0], color_pal[1], color_pal[2], color_pal[3]],
             data=pd.melt(filtered_data, id_vars=['cluster number'], value_vars=['ROSS', 'AE ensemble', 'IDEC', 'PseudoSort']))

#sns.violinplot(x='cluster number',
#               y='value',
#               hue='variable',
#               palette=[color_pal[0], color_pal[1], color_pal[2]], data=pd.melt(filtered_data, id_vars=['cluster number'], value_vars=['ROSS_mode', 'AE_ensemble_mode', 'Ours_mode']),
#               inner=None,
#               cut=0,
#               width=0.5,
#               legend=False)

#add a diagonal line (in terms of x and y values) dashed line to indicate the ground truth cluster number in grey
plt.plot([6, 8, 10, 12, 15], [6, 8, 10, 12, 15], linestyle='--', linewidth=1, color='grey')

#label the diagonal line directly below the line at an 45 degree angle
plt.text(10, 10.5, 'Ground Truth Cluster Number', horizontalalignment='center', verticalalignment='center', fontsize=12, color='grey', rotation=12.5)

ax.tick_params(direction="in")

for _,s in ax.spines.items():
    s.set_linewidth(1)
    s.set_color('black')

#plt.title('Violin Plots of Elbow Values for Each Cluster Number')
plt.xlabel('Cluster Number')
plt.ylabel('Predicted Cluster Number')
plt.xticks([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
plt.legend(fontsize=14)
plt.show()
#plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Fig_4_4_median_of_medians_fontsize.png')
'''

'''
# Group the data by settings and calculate mean absolute error for each metric
error_by_settings = data.groupby(['density_function', 'sampling_method', 'label_ratio']).agg(
    elbow_error_mean=('elbow_error', 'mean'),
    silhouette_error_mean=('silhouette_error', 'mean'),
    gap_statistic_error_mean=('gap_statistic_error', 'mean')
).reset_index()

# Find the minimum error for each metric across all settings
min_error_each_metric = error_by_settings[['elbow_error_mean', 'silhouette_error_mean', 'gap_statistic_error_mean']].min()

# Find the settings combination for the minimum mean absolute error of each metric
min_error_combinations = {
    metric: error_by_settings.loc[error_by_settings[metric] == error_by_settings[metric].min(),
                                  ['density_function', 'sampling_method', 'label_ratio']]
    for metric in ['elbow_error_mean', 'silhouette_error_mean', 'gap_statistic_error_mean']
}

print(min_error_each_metric)
print(min_error_combinations)
'''

'''
# Group data by the required combinations including 'iteration' to keep individual iterations separate for mode calculation
grouped_data = data.groupby(['dataset', 'density_function', 'sampling_method', 'label_ratio', 'iteration'])

metrics = ['elbow', 'silhouette', 'gap_statistic']
# Calculate the mean for the grouped data, then round to the closest integer for the first method
mean_data = grouped_data[metrics].mean().groupby(level=[1,2,3]).mean().round(0).astype(int).reset_index()

# Prepare data for the second aggregation method
# For this, we'll group by all except 'iteration' to consider the mode across iterations for each setting
grouped_for_mode = data.groupby(['dataset', 'density_function', 'sampling_method', 'label_ratio'])

# Create a custom aggregation function that returns the mode or the value closest to the mean in case of tie
def agg_mode(series):
    m = mode(series)  # Get the mode result, which includes counts
    if len(m.mode) > 1:  # If there's a tie
        # Calculate mean and find the value closest to mean among the modes
        closest_to_mean = min(m.mode, key=lambda x: abs(x-series.mean()))
        return closest_to_mean
    return m.mode[0]

# Apply this function to get the mode for each group
mode_aggregated = grouped_for_mode.agg({metric: agg_mode for metric in metrics}).reset_index()

# Merge the mean rounded data and the mode aggregated data for comparison
aggregated_data_comparison = pd.merge(
    mean_data,
    mode_aggregated,
    on=['density_function', 'sampling_method', 'label_ratio'],
    suffixes=('_mean_rounded', '_mode')
)


# Calculate the absolute error for the Mean Rounded method
aggregated_data_comparison['elbow_mean_rounded_error'] = (aggregated_data_comparison['elbow_mean_rounded'] - ground_truth_cluster_number).abs()
aggregated_data_comparison['silhouette_mean_rounded_error'] = (aggregated_data_comparison['silhouette_mean_rounded'] - ground_truth_cluster_number).abs()
aggregated_data_comparison['gap_statistic_mean_rounded_error'] = (aggregated_data_comparison['gap_statistic_mean_rounded'] - ground_truth_cluster_number).abs()

# Find the minimum error and its respective combination of settings for Mean Rounded method
min_errors_mean_rounded = {
    'elbow_mean_rounded': aggregated_data_comparison.loc[aggregated_data_comparison['elbow_mean_rounded_error'].idxmin(), ['density_function', 'sampling_method', 'label_ratio', 'elbow_mean_rounded_error']],
    'silhouette_mean_rounded': aggregated_data_comparison.loc[aggregated_data_comparison['silhouette_mean_rounded_error'].idxmin(), ['density_function', 'sampling_method', 'label_ratio', 'silhouette_mean_rounded_error']],
    'gap_statistic_mean_rounded': aggregated_data_comparison.loc[aggregated_data_comparison['gap_statistic_mean_rounded_error'].idxmin(), ['density_function', 'sampling_method', 'label_ratio', 'gap_statistic_mean_rounded_error']],
}

# For the Mode method, repeat the same process with the mode_values dataset
aggregated_data_comparison['elbow_mode_error'] = (aggregated_data_comparison['elbow_mode'] - ground_truth_cluster_number).abs()
aggregated_data_comparison['silhouette_mode_error'] = (aggregated_data_comparison['silhouette_mode'] - ground_truth_cluster_number).abs()
aggregated_data_comparison['gap_statistic_mode_error'] = (aggregated_data_comparison['gap_statistic_mode'] - ground_truth_cluster_number).abs()

# Find the minimum error and its respective combination of settings for Mode method
min_errors_mode = {
    'elbow_mode': aggregated_data_comparison.loc[aggregated_data_comparison['elbow_mode_error'].idxmin(), ['density_function', 'sampling_method', 'label_ratio', 'elbow_mode_error']],
    'silhouette_mode': aggregated_data_comparison.loc[aggregated_data_comparison['silhouette_mode_error'].idxmin(), ['density_function', 'sampling_method', 'label_ratio', 'silhouette_mode_error']],
    'gap_statistic_mode': aggregated_data_comparison.loc[aggregated_data_comparison['gap_statistic_mode_error'].idxmin(), ['density_function', 'sampling_method', 'label_ratio', 'gap_statistic_mode_error']],
}

print(min_errors_mean_rounded)
print(min_errors_mode)

data_one_iteration_elbow = data[(data['density_function'] == 'mean') & (data['sampling_method'] == 'densest') & (data['label_ratio'] == 0.25)]
data_one_iteration_elbow = data_one_iteration_elbow[['dataset', 'elbow']]

data_one_iteration_silhouette = data[(data['density_function'] == 'mean') & (data['sampling_method'] == 'weighted') & (data['label_ratio'] == 0.2)]
data_one_iteration_silhouette = data_one_iteration_silhouette[['dataset', 'silhouette']]

data_one_iteration_gap = data[(data['density_function'] == 'default') & (data['sampling_method'] == 'densest') & (data['label_ratio'] == 0.5)]
data_one_iteration_gap = data_one_iteration_gap[['dataset', 'gap_statistic']]

data_multiple_iterations_silhouette_mean = data[(data['density_function'] == 'default') & (data['sampling_method'] == 'weighted') & (data['label_ratio'] == 0.05)]
data_multiple_iterations_silhouette_mean = data_multiple_iterations_silhouette_mean[['dataset', 'iteration', 'silhouette']]
data_multiple_iterations_silhouette_mean = data_multiple_iterations_silhouette_mean.groupby('dataset')['silhouette'].mean().round(0).astype(int).reset_index()
data_multiple_iterations_silhouette_mean = data_multiple_iterations_silhouette_mean.rename(columns={'silhouette': 'silhouette_mean'})

print(data_multiple_iterations_silhouette_mean['silhouette_mean'])
#data_multiple_iterations_silhouette_mean.loc[:, 'silhouette_mean'] = data_multiple_iterations_silhouette_mean['silhouette'].mean().round(0).astype(int)
# Check for the same 'dataset' values

# Verify if 'dataset' columns are the same and in the same order across all dataframes
datasets_match = (list(data_one_iteration_elbow['dataset']) == list(data_one_iteration_silhouette['dataset']) ==
                  list(data_one_iteration_elbow['dataset']) == list(data_one_iteration_gap['dataset']))

# If the 'dataset' columns match and are in the same order, we can proceed to merge them
if datasets_match:
    # Merge the dataframes on the 'dataset' column
    # As silhouette_mean already includes 'dataset' we can directly use it and drop extra columns
    data_silhouette_mean = data_multiple_iterations_silhouette_mean[['dataset', 'silhouette_mean']]

    # Merge using concat since they are in the same order
    merged_data = pd.concat([data_one_iteration_elbow.set_index('dataset'),
                             data_one_iteration_silhouette.set_index('dataset'),
                             data_one_iteration_gap.set_index('dataset')], axis=1).reset_index()
                             #data_silhouette_mean.set_index('dataset')], axis=1).reset_index()
else:
    # If the order or the 'dataset' entries do not match, more complex merging is required.
    print("The dataset orders do not match, and merging requires alignment.")

# Now let's create the 2x5 grid of violin plots
unique_datasets = merged_data['dataset'].unique()

palette = {
    'elbow': 'blue',
    'silhouette': 'green',
    'gap_statistic': 'purple'
}

nrows, ncols = 2, 5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

# Prepare the data for plotting by melting it into a long format
merged_data_long = pd.melt(merged_data, id_vars='dataset', var_name='metric', value_name='value')

# Iterate over each unique dataset and create a separate violin plot on each subplot
for i, dataset in enumerate(unique_datasets):
    # Filter the data for the current dataset
    dataset_data = merged_data_long[merged_data_long['dataset'] == dataset]
    sns.violinplot(ax=axes_flat[i], x='metric', y='value', hue='metric', data=dataset_data, cut=0, palette=palette)
    axes_flat[i].set_title(f"Violin plots for {dataset}")
    axes_flat[i].set_xlabel('')
    axes_flat[i].set_ylabel('Value')
    axes_flat[i].set_yticks([1, 3, 5, 7, 9, 11])

# Hide any unused subplots if there are less than 10 datasets
for j in range(len(unique_datasets), nrows * ncols):
    axes_flat[j].axis('off')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
#plt.savefig('/Users/jakobtraeuble/Desktop/dfs_spikesorting/v2_cluster_optimisation_violin_plots.png')
'''
'''

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

color_pal = [(72, 165, 175), (160,210,210), (127,255,212), (52,58,64), (211,211,211), (250,250,245)]

for i in range(len(color_pal)):
  r, g, b = color_pal[i]
  color_pal[i] = (r / 255., g / 255., b / 255.)


#benchmark_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_1.csv')
benchmark_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_1_mean.csv')
#benchmark_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_1_max.csv')

# Split the dataset column into type and number for better categorization
benchmark_data['dataset_number'] = benchmark_data['dataset'].apply(lambda x: int(x.split('_')[1]))
benchmark_data['dataset_type'] = benchmark_data['dataset'].apply(lambda x: x.split('_')[0])
'''

###FIGURE 4.1 - MEAN
'''
benchmark_data.rename(columns={'ROSS_mean': 'ROSS'}, inplace=True)
benchmark_data.rename(columns={'AE_ensemble_mean': 'AE_ensemble'}, inplace=True)
benchmark_data.rename(columns={'Ours_mean': 'Ours'}, inplace=True)

# Preparing data for plotting
plot_data_mod = pd.melt(benchmark_data, id_vars=['dataset_type'], value_vars=['ROSS', 'AE_ensemble', 'Ours'])
grouped_data_mod = plot_data_mod.groupby(['dataset_type', 'variable']).agg(['mean', 'std']).reset_index()
grouped_data_mod.columns = ['Dataset Type', 'Method', 'Mean', 'Standard Deviation']

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
methods = ['ROSS', 'AE_ensemble', 'Ours']
dataset_types = ['Small', 'Large']
width = 0.2

for i, method in enumerate(methods):
    method_data = grouped_data_mod[grouped_data_mod['Method'] == method]
    for j, dataset_type in enumerate(dataset_types):
        subset = method_data[method_data['Dataset Type'] == dataset_type]
        position = j + i * width
        ax.plot([position - width/2, position + width/2], [subset['Mean'].values[0]]*2, color=color_pal[i])
        ax.add_patch(plt.Rectangle((position - width/2, subset['Mean'].values[0] - subset['Standard Deviation'].values[0]), width, subset['Standard Deviation'].values[0]*2, color=color_pal[i], alpha=0.5))
        data_points = plot_data_mod[(plot_data_mod['variable'] == method) & (plot_data_mod['dataset_type'] == dataset_type)]['value']
        ax.scatter([position]*len(data_points), data_points, color=color_pal[i], alpha=0.7)

ax.set_xticks([r + width for r in range(len(dataset_types))])
ax.set_xticklabels(dataset_types)
ax.tick_params(direction="in")
for _, s in ax.spines.items():
    s.set_linewidth(1)
    s.set_color('black')

plt.ylabel('Accuracy')
plt.xlabel('Dataset Type')
plt.legend(handles=[plt.Line2D([0], [0], color=color_pal[i], lw=4, label=method) for i, method in enumerate(methods)])
#plt.show()
plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Fig_4_1_mean_std_03022024.png')

'''
###FIGURE 4.1 - ALL
'''
benchmark_data.rename(columns={'ROSS_mean': 'ROSS'}, inplace=True)
benchmark_data.rename(columns={'AE_ensemble_mean': 'AE ensemble'}, inplace=True)
benchmark_data.rename(columns={'Ours_mean': 'PseudoSort'}, inplace=True)
benchmark_data.rename(columns={'IDEC_mean': 'IDEC'}, inplace=True)

#benchmark_data.rename(columns={'ROSS_max': 'ROSS'}, inplace=True)
#benchmark_data.rename(columns={'AE_ensemble_max': 'AE_ensemble'}, inplace=True)
#benchmark_data.rename(columns={'Ours_max': 'Ours'}, inplace=True)

#benchmark_data.rename(columns={'ROSS [acc]': 'ROSS'}, inplace=True)
#benchmark_data.rename(columns={'AE_ensemble [acc]': 'AE_ensemble'}, inplace=True)
#benchmark_data.rename(columns={'Ours [acc]': 'Ours'}, inplace=True)

fig, ax = plt.subplots(figsize=(8,6))

long_df = pd.melt(benchmark_data, id_vars=['dataset_type'], value_vars=['ROSS', 'AE ensemble', 'IDEC', 'PseudoSort'])

sns.stripplot(x='dataset_type',
              y='value',
              hue='variable',
              data=long_df,
              palette=['black', 'black', 'black', 'black'],
              dodge=True,
              jitter=False,
              alpha=0.8,
              legend=False)

# Plotting the box plot
sns.boxplot(x='dataset_type',
            y='value',
            hue='variable',
            palette=[color_pal[0], color_pal[1], color_pal[2], color_pal[3]],
            data=long_df,
            showfliers=False)

#print the median values for each method for each dataset type
print(long_df.groupby(['dataset_type', 'variable'])['value'].median())


ax.tick_params(direction="in")

for _,s in ax.spines.items():
    s.set_linewidth(1)
    s.set_color('black')

plt.ylabel('Accuracy')
plt.xlabel('Dataset')
#increase font size of legend
plt.legend(fontsize=14)
#plt.show()
plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Fig_4_1_mean_fontsize.png')
'''

'''
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

color_pal = [(72, 165, 175), (160,210,210), (127,255,212), (52,58,64), (211,211,211), (250,250,245)]

for i in range(len(color_pal)):
  r, g, b = color_pal[i]
  color_pal[i] = (r / 255., g / 255., b / 255.)

#benchmark_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_2.csv')
benchmark_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_2_mean.csv')
#benchmark_data = pd.read_csv('/Users/jakobtraeuble/Desktop/dfs_spikesorting/Fig_4_2_max.csv')


###FIGURE 4.2 MEAN
benchmark_data.rename(columns={'ROSS_mean': 'ROSS'}, inplace=True)
benchmark_data.rename(columns={'AE_ensemble_mean': 'AE ensemble'}, inplace=True)
benchmark_data.rename(columns={'Ours_mean': 'PseudoSort'}, inplace=True)
benchmark_data.rename(columns={'IDEC_mean': 'IDEC'}, inplace=True)

#benchmark_data.rename(columns={'ROSS_max': 'ROSS'}, inplace=True)
#benchmark_data.rename(columns={'AE_ensemble_max': 'AE_ensemble'}, inplace=True)
#benchmark_data.rename(columns={'Ours_max': 'Ours'}, inplace=True)
'''


'''
# Preparing data for plotting
plot_data_mod = pd.melt(benchmark_data, id_vars=['cluster_number'], value_vars=['ROSS', 'AE_ensemble', 'Ours'])
grouped_data_mod = plot_data_mod.groupby(['cluster_number', 'variable']).agg(['mean', 'std']).reset_index()
grouped_data_mod.columns = ['Cluster Number', 'Method', 'Mean', 'Standard Deviation']

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
methods = ['ROSS', 'AE_ensemble', 'Ours']
cluster_numbers = np.unique(benchmark_data['cluster_number'])
print('cluster_numbers', cluster_numbers)
width = 0.2

for i, method in enumerate(methods):
    method_data = grouped_data_mod[grouped_data_mod['Method'] == method]
    for j, cluster_number in enumerate(cluster_numbers):
        subset = method_data[method_data['Cluster Number'] == cluster_number]
        position = j + i * width
        ax.plot([position - width/2, position + width/2], [subset['Mean'].values[0]]*2, color=color_pal[i])
        ax.add_patch(plt.Rectangle((position - width/2, subset['Mean'].values[0] - subset['Standard Deviation'].values[0]), width, subset['Standard Deviation'].values[0]*2, color=color_pal[i], alpha=0.5))
        data_points = plot_data_mod[(plot_data_mod['variable'] == method) & (plot_data_mod['cluster_number'] == cluster_number)]['value']
        ax.scatter([position]*len(data_points), data_points, color=color_pal[i], alpha=0.7)

ax.set_xticks([r + width for r in range(len(cluster_numbers))])
ax.set_xticklabels(cluster_numbers)
ax.tick_params(direction="in")
for _, s in ax.spines.items():
    s.set_linewidth(1)
    s.set_color('black')

plt.ylabel('Accuracy')
plt.xlabel('Neuron Count')
plt.legend(handles=[plt.Line2D([0], [0], color=color_pal[i], lw=4, label=method) for i, method in enumerate(methods)])
#plt.show()
plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Fig_4_2_mean_std_03012024.png')
'''

###FIGURE 4.2 ALL
'''
#benchmark_data.rename(columns={'ROSS [acc]': 'ROSS'}, inplace=True)
#benchmark_data.rename(columns={'AE_ensemble [acc]': 'AE_ensemble'}, inplace=True)
#benchmark_data.rename(columns={'Ours [acc]': 'Ours'}, inplace=True)

df_long = pd.melt(benchmark_data, id_vars=['cluster_number'], value_vars=['ROSS', 'IDEC', 'AE ensemble','PseudoSort'])

fig, ax = plt.subplots(figsize=(15,6))

sns.stripplot(x='cluster_number',
                y='value',
                hue='variable',
                data=df_long,
                palette=['black', 'black', 'black', 'black'],
                dodge=True,
                jitter=False,
                alpha=0.8,
                legend=False)

sns.boxplot(x='cluster_number',
            y='value',
            hue='variable',
            palette=[color_pal[0], color_pal[2], color_pal[1], color_pal[3]],
            data=df_long,
            showfliers=False)

#print the median values for each method for each dataset type
print(df_long.groupby(['cluster_number', 'variable'])['value'].median())

ax.tick_params(direction="in")

for _,s in ax.spines.items():
    s.set_linewidth(1)
    s.set_color('black')

plt.ylabel('Accuracy')
plt.xlabel('Neuron Count')
plt.xticks()
plt.legend(fontsize=14)
#plt.show()
plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Fig_4_2_mean_all_fontsize.png')
'''

'''
benchmark_data['Relative_Improvement'] = ((benchmark_data['Ours'] - benchmark_data['AE_ensemble']) /
                                           benchmark_data['AE_ensemble']) * 100

relative_improvement_median = benchmark_data.groupby('Neuron_Count')['Relative_Improvement'].median().reset_index()


# Now, let's plot the data with the relative improvement on the secondary y-axis
plt.figure(figsize=(14, 7))

# Plot the boxplot for 'AE_ensemble' and 'Ours' values
ax1 = sns.boxplot(x='Neuron_Count', y='value', hue='variable', palette=[color_pal[0], color_pal[2]],
            data=pd.melt(benchmark_data, id_vars=['Neuron_Count'], value_vars=['AE_ensemble', 'Ours']))

# Set the primary y-axis label
ax1.set_ylabel('Benchmark Values')

# Create a secondary y-axis
ax2 = ax1.twinx()

# Plot the relative improvement as a line plot on the secondary y-axis
sns.lineplot(x=relative_improvement_median['Neuron_Count'] - 6,  # Adjusting for zero-based indexing
             y='Relative_Improvement',
             data=relative_improvement_median,
             ax=ax2,
             color=color_pal[1],
             marker="o",
             linestyle='--')
# Set the secondary y-axis label
ax2.set_ylabel('Relative Improvement (%)', color=color_pal[1])
ax2.tick_params(axis='y', colors=color_pal[1])
ax2.spines['right'].set_color(color_pal[1])

# Set the title and labels
plt.title('Benchmark Comparison and Relative Improvement by Dataset')
ax1.set_xlabel('Dataset Type and Neuron Count')
ax1.grid(False)  # Turn off grid to avoid double-grid with two y-axes
plt.legend()
# Display the plot
#plt.show()
plt.savefig('/Users/jakobtraeuble/Desktop/SpikeSorting_plots/Fig_4_2_v2.png')'''

'''
import os
import numpy as np

def get_spike_shapes_files(folder):
    spike_shapes_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith('spike_shapes'):
                spike_shapes_files.append(os.path.join(root, file))
    return spike_shapes_files

def spikes_definitions(folder):
    #Load the .npy file with allow_pickle=True
    spike_shapes = np.load(folder, allow_pickle=True) # loads the shapes and contains info related to electrodes.
    merged_dict = {}

    for spike_dict in spike_shapes:
        channel = spike_dict['channel']
        spikes = spike_dict['spike_shapes']

        if channel in merged_dict:
            merged_dict[channel] = np.concatenate((merged_dict[channel], spikes), axis=0)
        else:
            merged_dict[channel] = spikes.copy()  # Use copy to avoid modifying the original array

# Function to calculate firing rate
def calculate_firing_rate(spikes, duration):
    return len(spikes) / duration


def calculate_spike_features(spikes):
    mean_spike = np.mean(spikes, axis=0)
    std_spike = np.std(spikes, axis=0)

    max_amplitude = np.max(np.abs(mean_spike))  # Use absolute value for max amplitude
    max_amplitude_loc = np.argmax(np.abs(mean_spike))

    # Determine whether spikes are predominantly positive or negative
    is_positive_spike = mean_spike[max_amplitude_loc] > 0

    # Find indices where mean_spike is above half of the max_amplitude
    above_half_max = mean_spike > max_amplitude / 2 if is_positive_spike else mean_spike < -max_amplitude / 2
    half_width_indices = np.where(above_half_max)[0]

    # Calculate half-width based on the first and last indices
    half_width = half_width_indices[-1] - half_width_indices[0]

    spike_height = max_amplitude - np.min(np.abs(mean_spike))
    area_under_curve = np.trapz(np.abs(mean_spike), dx=1)  # Assuming a unit time interval

    # Find indices corresponding to the rise and fall points
    rise_time = np.argmax(mean_spike)
    fall_time = len(mean_spike) - np.argmax(mean_spike[::-1]) - 1

    return {
        'Max Amplitude': max_amplitude,
        'FWHM': half_width,
        'Spike Height': spike_height,
        'Area Under Curve': area_under_curve,
        'Rise Time': rise_time,
        'Fall Time': fall_time,
        'Spike Polarity Positive?': is_positive_spike
    }

Folder_path = '/rds/user/jnt27/hpc-work/EPhys/Spiderweb_Trial_1/Data/Bel_output/'

fsample = 30000
columns = ['Timestamp', 'Channel', 'Max Amplitude', 'FWHM', 'Spike Height', 'Area Under Curve', 'Rise Time',
               'Fall Time', 'Positive Polarity?', 'Firing Rate']

# Initialize an empty DataFrame
feature_data = pd.DataFrame(columns=columns)

#loop over all files in directory
dir_name = os.listdir(Folder_path)
for date in dir_name:
    if date.startswith('22') or date.startswith('23'):
        #join paths
        spikes_file = os.path.join(Folder_path, date)
        spikes_file = os.path.join(spikes_file, 'spike_shapes_' + date + '.npy')

        print(spikes_file)

        # Extract the date_acquired from the file name (you may need to adjust this based on your file naming convention)
        date_acquired = spikes_file.split('_')[-3]
        time_acquired = spikes_file.split('_')[-2]
        name_indicator = spikes_file[-22:]
        # Call spikes_definitions function
        spike_shapes, merged_dict, global_spikes = spikes_definitions(spikes_file)

        # Iterate over channel_info in global_spikes['global_data']
        for channel_info in spike_shapes:
            channel_name = channel_info['channel']
            spikes = channel_info['spike_shapes']
            # Calculate features
            features = calculate_spike_features(spikes)
            recording_duration = 298.00053333333335  # seconds - to be modified
            firing_rate = calculate_firing_rate(spikes, recording_duration)
            # Save feature data to DataFrame
            feature_data = pd.concat([feature_data, pd.DataFrame(
                {'Timestamp': [f'{date_acquired}_{time_acquired}'], 'Channel': [channel_name], **features,
                 'Firing Rate': [firing_rate], 'Number of Spikes': [len(spikes)]})], ignore_index=True)

        # Save the feature data to a CSV file
    feature_data.to_csv('/rds/user/jnt27/hpc-work/EPhys/Spiderweb_Trial_1/Data/Bel_output/directory_spike_features.csv', index=False)
    '''