import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Helper class to produce plots for the extracted events and post-processed the explanation
class HelperClass:
    
    def __init__(self, base_dir):
        self.base_dir = base_dir


    def event_region(self, X, y, inc_dec_parametrized_events, class_labels):
        print(f"""
                Extracted increasing and decreasing events:
                -------------------------------
                """)
        num_classes = len(class_labels)
        subplot_height = 2.5  # Adjust the height of each subplot as needed

        fig, axes = plt.subplots(num_classes, 1, figsize=(6, subplot_height * num_classes), sharex=True)

        for i, classe in enumerate(class_labels):
            class_index = np.where(y == i)[0][0]
            ax = axes[i]

            ax.plot(X[class_index], color='C0', linewidth=1.5)

            for event in inc_dec_parametrized_events['Increasing_ch1'][class_index]:
                ax.axvspan(event[0], event[0] + event[1], ymin=-1, ymax=1, alpha=0.2, color='green', label="Region of increasing events")

            for event in inc_dec_parametrized_events['Decreasing_ch1'][class_index]:
                ax.axvspan(event[0], event[0] + event[1], ymin=0, ymax=1, alpha=0.2, color='red', label="Region of decreasing events")

            ax.set_title(f'Class: {classe}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(alpha=0.5)
            # Create a custom legend
            legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.3) for color in ['green', 'red']]
            ax.legend(legend_patches, ['Increasing', 'Decreasing'], loc='upper right')

        # Adjust the vertical spacing between subplots
        plt.subplots_adjust(hspace=0.4)

        # Create a custom legend
        # legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.3) for color in ['green', 'red']]
        # plt.legend(legend_patches, ['Increasing', 'Decreasing'], loc='upper right')

        # Save the plot as an image and a PDF
        plt.tight_layout()
        plt.savefig(f'{self.base_dir}/inc_dec_events.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.base_dir}/inc_dec.pdf', format='pdf', dpi=300, bbox_inches='tight')

        plt.show()


    def event_points(self, X, y, max_min_parametrized_events, class_labels):
        print(f"""
                Extracted local maximum and local minimum events:
                -------------------------------
                """)
        num_classes = len(class_labels)
        subplot_height = 2.5  # Adjust the height of each subplot as needed

        fig, axes = plt.subplots(num_classes, 1, figsize=(6, subplot_height * num_classes), sharex=True)

        for i, classe in enumerate(class_labels):
            class_index = np.where(y == i)[0][0]
            ax = axes[i]

            ax.plot(X[class_index], color='C0', linewidth=1.5)

            for max_event in max_min_parametrized_events['LocalMax_ch1'][class_index]:
                ax.plot(max_event[0], max_event[1], '*', c='r')

            for min_event in max_min_parametrized_events['LocalMin_ch1'][class_index]:
                ax.scatter(min_event[0], min_event[1], marker='*', c='b')

            ax.set_title(f'Class: {classe}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(alpha=0.5)

        # Create a custom legend
        legend_patches = [
            plt.Line2D([0], [0], marker='*', color='r', label='Local Max', linestyle=''),
            plt.Line2D([0], [0], marker='*', color='b', label='Local Min', linestyle='')
        ]

        # Place the legend at the bottom of the plot
        plt.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)

        # Save the plot as an image and a PDF
        plt.savefig(f'{self.base_dir}/maxMinEvents.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.base_dir}/maxMinEvents.pdf', format='pdf', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_3D(self, kmeans, flatten_events, scaler, col_name):
        data_scaled = scaler.fit_transform(flatten_events)
        labels = kmeans.predict(data_scaled)
        centers = kmeans.cluster_centers_

        flatten_events = pd.DataFrame(scaler.inverse_transform(data_scaled), columns=['Timestep', 'Duration', 'Average_Value'])

        cmap = plt.get_cmap('tab20')
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Create a scatter plot in 3D
        sc = ax.scatter(flatten_events['Timestep'], flatten_events['Duration'], flatten_events['Average_Value'], c=labels, cmap='viridis')

        # # Add color bar
        # cbar = plt.colorbar(sc)
        # cbar.set_label('Cluster Label', fontsize=12)

        ax.set_title(f'K-Means Clusters for {col_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Duration', fontsize=12)
        ax.set_zlabel('Average Value', fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{self.base_dir}/kmeans_3D_plot4{col_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.base_dir}/kmeans_3D_plot4{col_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')

        plt.show()


    def plot_2D(self, kmeans, flatten_events, scaler, col_name):
        data_scaled = scaler.fit_transform(flatten_events)
        labels = kmeans.predict(data_scaled)
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)

        flatten_events = pd.DataFrame(scaler.inverse_transform(data_scaled), columns=['Timestep', 'Duration'])

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(flatten_events['Timestep'], flatten_events['Duration'], c=labels, cmap='viridis')

        for i, center in enumerate(centroids):
            ax.annotate(f"Cluster {i+1}", center, fontsize=12, fontweight='bold',
                        color='red', xytext=(-30, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Duration', fontsize=12)
        ax.set_title(f'K-Means Clusters for {col_name}', fontsize=14, fontweight='bold')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.base_dir}/kmeans_2D_plot4{col_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.base_dir}/kmeans_2D_plot4{col_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
   

    def plot_events_bar_chart(self, merged_df, test_preds, class_labels):
        # Number of classes
        num_classes = len(class_labels)

        # Calculate lengths and means for each class
        lengths = []
        means = []
        for i in range(num_classes):
            indices = torch.nonzero(test_preds == i).squeeze()
            lengths_df = merged_df.applymap(lambda x: len(x))
            lengths.append(lengths_df.iloc[indices])
            means.append(lengths[i].mean())

        # Set up positions for the bars
        x = range(len(means[0]))

        # Plotting
        bar_width = 0.1
        gap = 0.2
        fig, ax = plt.subplots(figsize=(7, 5))

        for i in range(num_classes):
            ax.bar([pos + i * (bar_width) for pos in x], means[i], width=bar_width, label=f'Class {class_labels[i]}')

        # Add labels and title
        plt.xlabel('Parametrized Event Primitives (PEP)', fontsize=12)
        plt.ylabel('Mean Value', fontsize=12)
        plt.title('Number of Extracted Events for Each Class', fontsize=12)
        plt.xticks([pos + (bar_width * (num_classes - 1) / 2) for pos in x], means[0].index, rotation=45, ha='right')

        # Add grid lines
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.legend(title='Classes', loc='upper right', bbox_to_anchor=(1.15, 1.0))

        plt.tight_layout()
        plt.savefig(f'{self.base_dir}/events_bar_plot.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.base_dir}/events_bar_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')

        plt.show()




        # def post_processed(self, inc_events_kmeans, dec_events_kmeans, max_events_kmeans, min_events_kmeans,
        #                     scaler_increasing, scaler_decreasing, scaler_max, scaler_min):
        #     inc_dict = {"icls_{}".format(i+1):"increases from time {:.0f} to {:.0f}".format(center[0], center[0]+ center[1])
        #                     for i,center in enumerate( scaler_increasing.inverse_transform(inc_events_kmeans.cluster_centers_))}
        #     dec_dict = {"dcls_{}".format(i+1):"decreases from time {:.0f} to {:.0f}".format(center[0], center[0]+ center[1])
        #                         for i,center in enumerate(scaler_decreasing.inverse_transform(dec_events_kmeans.cluster_centers_))}
        #     max_dict = {"maxcls_{}".format(i+1): "local max at {:.0f}".format(center[0], center[1])
        #                         for i,center in enumerate(scaler_max.inverse_transform(max_events_kmeans.cluster_centers_))}
        #     min_dict = {"mincls_{}".format(i+1): "local min at {:.0f}".format(center[0], center[1])
        #                         for i,center in enumerate(scaler_min.inverse_transform(min_events_kmeans.cluster_centers_))}
        #     g_feature_dict = {"global_feature": "Global Feature"}

        #     master_dict = {}
        #     for d in [inc_dict, dec_dict, max_dict, min_dict, g_feature_dict]:
        #         master_dict.update(d)
        #     return master_dict
        

    def plot_cluster_event_counts(self, test_preds, class_labels, appended_df):
        # Number of classes
        num_classes = len(class_labels)

        # Initialize lists to store means and labels
        means = []
        labels = []

        # Create a figure and axis for plotting
        fig, ax = plt.subplots(figsize=(7, 5))

        # Set up positions for the bars
        x = np.arange(len(appended_df.columns))

        # Define bar width and gap
        bar_width = 0.2

        for i in range(num_classes):
            indices = torch.nonzero(test_preds == i).squeeze()
            df_class = appended_df.iloc[indices]
            means.append(df_class.sum())  # Change from mean to sum to get the counts
            labels.append(f'Class {i}')

        # Plotting
        for i in range(num_classes):
            plt.bar(x + i * bar_width, means[i], width=bar_width, label=f'Class {class_labels[i]}')

        # Add labels and title
        plt.xlabel('Set of Clusters', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.title('Number of Events Belonging to Each Cluster for Each Class', fontsize=12)
        plt.xticks(x + (bar_width) * (num_classes - 1) / 2, appended_df.columns, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot in PNG format
        plt.savefig(f'{self.base_dir}/bar_plot4clusters.png', dpi=300, bbox_inches='tight')

        # Save the plot in PDF format
        plt.savefig(f'{self.base_dir}/bar_plot4clusters.pdf', format='pdf', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_events_on_timeseries(self, X, y, feature_importances, cluster_centroids, class_names):
        num_classes = len(class_names)
        subplot_height = 2.5  # Adjust the height of each subplot as needed
        fig, axes = plt.subplots(num_classes, 1, figsize=(6, subplot_height*num_classes), sharex=True)
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])

        def plot_single_class(class_idx, ax):
            class_name = class_names[class_idx]
            class_indices = np.where(y == class_idx)[0]
            class_index = class_indices[5] if len(class_indices) > 5 else None
            # print(class_index)

            if class_index is not None:
                ax.plot(X[class_index], color='C0', linewidth=1.5)
                feature_names = feature_importances[class_idx]

            for col_name in feature_names:
                # print(col_name)
                if "global_feature" in col_name.lower():
                    continue

                centroids = cluster_centroids[col_name]  # Assuming cluster_centroids is a dictionary

                if "increasing" in col_name.lower() or "decreasing" in col_name.lower():
                    start_time, duration, avg_value = centroids
                    start_time, duration = round(start_time), round(duration)
                    event_color = 'green' if "increasing" in col_name.lower() else 'red'
                    event_label = "Increasing Events" if "increasing" in col_name.lower() else "Decreasing Events"
                    ax.axvspan(start_time, start_time + duration, ymin=-1, ymax=1, alpha=0.2, color=event_color, label=event_label)

                # Handle local_max and local_min events
                elif "localmax" in col_name.lower() or "localmin" in col_name.lower():
                    time, point = centroids
                    time = round(time)
                    marker_color = 'r' if "localmax" in col_name.lower() else 'b'
                    marker_label = "Local Max" if "localmax" in col_name.lower() else "Local Min"
                    ax.plot(time, point, '*', c=marker_color, label=marker_label)

            ax.set_title(f'Class: {class_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(alpha=0.5)
            # Colored rectangles legend
            colored_rectangles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.3) for color in ['green', 'red']]
            legend_labels = ['Increasing', 'Decreasing']

            # Custom legend markers
            custom_markers = [plt.Line2D([0], [0], marker='*', color='r', label='Local Max', linestyle=''),
                                plt.Line2D([0], [0], marker='*', color='b', label='Local Min', linestyle='')]

            # Combine the legend patches and markers
            legend_patches = colored_rectangles + custom_markers
            legend_labels += ['Local Max', 'Local Min']

            # Create the legend
            ax.legend(legend_patches, legend_labels, loc='upper right')
            # ax.legend(loc='upper right')

        for i, ax in enumerate(axes):
            plot_single_class(i, ax)

        # Set the same x-axis label for both subplots
        ax.set_xlabel('Timesteps')

        # Create a legend for the entire figure
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(f'{self.base_dir}/important_features_line_plot.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.base_dir}/important_features_line_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_events_as_line_on_timeseries(self, X, y, feature_importances, cluster_centroids, class_names):
        num_classes = len(class_names)
        subplot_height = 2.5  # Adjust the height of each subplot as needed
        fig, axes = plt.subplots(num_classes, 1, figsize=(6, subplot_height*num_classes), sharex=True)
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])

        def plot_single_class(class_idx, ax):
            class_name = class_names[class_idx]
            class_indices = np.where(y == class_idx)[0]
            class_index = class_indices[5] if len(class_indices) > 5 else None
            # print(class_index)

            if class_index is not None:
                ax.plot(X[class_index], color='C0', linewidth=1.5)
                feature_names = feature_importances[class_idx]

            for col_name, importance in feature_names.items():
                # print(col_name)
                if "global_feature" in col_name.lower():
                    continue

                centroids = cluster_centroids[col_name]  # Assuming cluster_centroids is a dictionary

                if "increasing" in col_name.lower() or "decreasing" in col_name.lower():
                    start_time, duration, _ = centroids
                    start_time, duration = round(start_time), round(duration)
                    event_color = 'green' if "increasing" in col_name.lower() else 'red'
                    event_label = "Increasing Events" if "increasing" in col_name.lower() else "Decreasing Events"
                    # ax.axvspan(start_time, start_time + duration, ymin=-1, ymax=1, alpha=0.2, color=event_color, label=event_label)
                    # Draw a line from start to start + duration
                    time_values = np.arange(start_time, (start_time + duration))
                    # print(time_values)
                    ax.plot(time_values, X[class_index, start_time:(start_time + duration)], color=event_color, linewidth=2)
                    # Add label on top of the line
                    label_x = int(start_time + duration / 2)  # Adjust the label position as needed
                    label_y = X[class_index, label_x]
                    ax.annotate(f'{round(importance,2)}', (label_x, label_y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color=event_color)

                # Handle local_max and local_min events
                elif "localmax" in col_name.lower() or "localmin" in col_name.lower():
                    local_time, point = centroids
                    local_time = round(local_time)

                    marker_color = 'r' if "localmax" in col_name.lower() else 'b'
                    marker_label = "Local Max" if "localmax" in col_name.lower() else "Local Min"
                    # ax.plot(time, point, '*', c=marker_color, label=marker_label)
                    ax.plot(local_time, X[class_index, local_time], '*', c=marker_color, label=marker_label)
                    # Add label on top of the marker
                    ax.annotate(f'{round(importance, 2)}', (local_time, X[class_index, local_time]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color=marker_color)




            ax.set_title(f'Class: {class_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(alpha=0.5)
            # Colored rectangles legend
            colored_rectangles = [plt.Line2D([0, 1], [0, 1], color=color, linewidth=2) for color in ['green', 'red']]
            legend_labels = ['Increasing', 'Decreasing']

            # Custom legend markers
            custom_markers = [plt.Line2D([0], [0], marker='*', color='r', label='Local Max', linestyle=''),
                                plt.Line2D([0], [0], marker='*', color='b', label='Local Min', linestyle='')]

            # Combine the legend patches and markers
            legend_patches = colored_rectangles + custom_markers
            legend_labels += ['Local Max', 'Local Min']

            # Create the legend
            ax.legend(legend_patches, legend_labels, loc='upper right')
            # ax.legend(loc='upper right')

        for i, ax in enumerate(axes):
            plot_single_class(i, ax)

        # Set the same x-axis label for both subplots
        ax.set_xlabel('Timesteps')

        # Create a legend for the entire figure
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(f'{self.base_dir}/important_features_line_plot.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.base_dir}/important_features_line_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')

        plt.show()


    
    def post_processed(self, events_kmeans, scaler, col_name):
        prefix = ""
        channel = col_name.split('_')[1]
        if "increasing" in col_name.lower():
            prefix = f"{col_name}_c"
            text =  "increases from time {:.0f} to {:.0f} with average value {:.2f}"
        elif "decreasing" in col_name.lower():
            prefix = f"{col_name}_c"   #"dcls"
            text =  "decreases from time {:.0f} to {:.0f} with average value {:.2f}"
        elif "localmax" in col_name.lower():
            prefix = f"{col_name}_c" #"maxcls"
            text =  "local maximum at time {:.0f} with value {:.2f}"
        elif "localmin" in col_name.lower():
            prefix = f"{col_name}_c" #"mincls"
            text =  "local minimum at time {:.0f} with value {:.2f}"

        processed_dict = {
            "{}{}".format(prefix, i + 1): text.format(center[0], center[0] + center[1], center[2])
            if "increasing" in col_name.lower() or "decreasing" in col_name.lower()
            else text.format(center[0], center[1])
            for i, center in enumerate(scaler.inverse_transform(events_kmeans.cluster_centers_))
        }

        centroid_dict = {"{}{}".format(prefix, i + 1): center for i, center in enumerate(scaler.inverse_transform(events_kmeans.cluster_centers_))}
        return processed_dict, centroid_dict
    def update_master_dict(self, master_dict, global_df):
        # Convert the column names to a dictionary
        column_dict = {col: f"{col.split('_')[2]} global feature" for i, col in enumerate(global_df.columns)}

        # Print the dictionary
        # print(column_dict)
        master_dict.update(column_dict)
        return master_dict