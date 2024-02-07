# import all necessary libraries

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from utils.helper_class import *
from metrics import *



class GlobXplain4TSC:
  def __init__(self, base_dir):
    self.base_dir= base_dir
    self.helper_instance = HelperClass(base_dir=base_dir) 

  # Turn the data of 2D shape into 3D
  def preprocessing_data(self, X):
    print(f'Shape of data : {X.shape}')

    # Create sample input data
    # data = np.random.rand(90, 24, 51)

    # Reshape input data into a 2D array
    reshaped_data = np.empty((X.shape[0], X.shape[1]), dtype=np.ndarray)
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
          reshaped_data[i, j] = X[i, j, :]

    # Create a list of column names for the DataFrame
    col_names = [f"ch{i+1}" for i in range(X.shape[1])]


    # Create the DataFrame
    df = pd.DataFrame(reshaped_data, columns=col_names)

    # Print the resulting DataFrame
    # print(df)
    return df


  def global_feature_extraction(self, df):
    # Compute the mean of each cell
    # mean_df = df.applymap(lambda x: np.mean(x))
    # mean_df = mean_df.round(7)
    reshaped_data = df.reshape((df.shape[0], df.shape[2]))

  # Compute the mean across all time series for each time step
    mean_df = np.mean(reshaped_data, axis=0)
    # Create a DataFrame with dynamic column names for each channel
    num_channels = df.shape[1]
    channel_columns = [f'ch{i+1}' for i in range(num_channels)]
    mean_df = pd.DataFrame(mean_df, columns=channel_columns)
    # Define the dynamic column name pattern
    col_name = 'global_feature'

    # Create a dictionary to map the original column names to the new column names
    column_mapping = {col: f'{col_name}_{col}' for col in mean_df.columns}

    # Rename the columns using the dictionary
    mean_df = mean_df.rename(columns=column_mapping)

    return mean_df



  def extract_inc_dec_events(self, data):
    # Reshape input data into a 2D array
    reshaped_data = np.empty((data.shape[0], data.shape[1]), dtype=np.ndarray)
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
          reshaped_data[i, j] = data[i, j, :]

    # Create a list of column names for the DataFrame
    col_names = [f"ch{i+1}" for i in range(data.shape[1])]


    # Create the DataFrame
    df = pd.DataFrame(reshaped_data, columns=col_names)

    # Initialize the result DataFrame
    result_df = pd.DataFrame(index=df.index)
    events_dict = {}
    # Extract increasing and decreasing events for each column of each instance
    for col_name in df.columns:
      increasing_events = []
      decreasing_events = []

      for instance_idx in range(len(df)):
          col_values = df.loc[instance_idx, col_name]
          inc_start_time = 0
          dec_start_time = 0
          inc_duration = 0
          dec_duration = 0
          inc_events = []
          dec_events = []
          inc_sum_values = 0
          dec_sum_values = 0
          for i in range(1, len(col_values)):

              if col_values[i] > col_values[i-1]:
                  if dec_duration > 0:
                      dec_avg_value = dec_sum_values / dec_duration
                      dec_events.append([dec_start_time, dec_duration, dec_avg_value])
                      dec_duration = 0
                      dec_sum_values = col_values[i]
                  if inc_duration == 0:
                      inc_start_time = i
                  inc_duration += 1
                  inc_sum_values += col_values[i]
              elif col_values[i] < col_values[i-1]:
                  if inc_duration > 0:
                      inc_avg_value = inc_sum_values / inc_duration
                      inc_events.append([ inc_start_time, inc_duration, inc_avg_value])
                      inc_duration = 0
                      inc_sum_values = col_values[i]
                  if dec_duration == 0:
                      dec_start_time = i
                  dec_duration += 1
                  dec_sum_values += col_values[i]
          if inc_duration > 0:
              inc_avg_value = inc_sum_values / inc_duration
              inc_events.append([inc_start_time, inc_duration, inc_avg_value])
          if dec_duration > 0:
              dec_avg_value = dec_sum_values / dec_duration
              dec_events.append([dec_start_time, dec_duration, dec_avg_value])
          increasing_events.append(inc_events)
          decreasing_events.append(dec_events)
      # print(pd.Series(inc_events))
      result_df[f"Increasing_{col_name}"] = increasing_events
      result_df[f"Decreasing_{col_name}"] = decreasing_events

    # Display the resulting DataFrame
    # print(type(result_df))
    return result_df


  def extract_local_max_min_events(self, data):
    # Reshape input data into a 2D array
    reshaped_data = np.empty((data.shape[0], data.shape[1]), dtype=np.ndarray)
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
          reshaped_data[i, j] = data[i, j, :]

    # Create a list of column names for the DataFrame
    col_names = [f"ch{i+1}" for i in range(data.shape[1])]


    # Create the DataFrame
    df = pd.DataFrame(reshaped_data, columns=col_names)

    # Initialize the result DataFrame
    result_df = pd.DataFrame(index=df.index)
    events_dict = {}
    # Extract local max and local min events for each column of each instance
    for col_name in df.columns:
      local_max_events = []
      local_min_events = []
      for instance_idx in range(len(df)):
          col_values = df.loc[instance_idx, col_name]
          max_events = []
          min_events = []
          for i in range(1, len(col_values)-1):
              if col_values[i] > col_values[i-1] and col_values[i] > col_values[i+1]:
                  max_events.append([i, col_values[i]])
              elif col_values[i] < col_values[i-1] and col_values[i] < col_values[i+1]:
                  min_events.append([i, col_values[i]])

          local_max_events.append(max_events)
          local_min_events.append(min_events)

      result_df[f"LocalMax_{col_name}"] = local_max_events
      result_df[f"LocalMin_{col_name}"] = local_min_events

    # Display the resulting DataFrame
    # print(result_df)
    return result_df

  def flatten_nested_events(self, events_list):
    inner_values = [inner for row in events_list for inner in row]
    inner_values_2d = np.array(inner_values, dtype=object).tolist()
    return inner_values_2d

  def cluster_events(self, all_events, k=12, col_name=""):

      silhouette_scores = []
      sse = []
      data = np.array(all_events)
      scaler= StandardScaler()
      data_transformed = scaler.fit_transform(data)

      for n_clusters in range(2,k):


        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_transformed)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(data_transformed, labels)
        silhouette_scores.append(silhouette_avg)
        optimal_k = np.argmax(silhouette_scores) + 2

        # print(optimal_k)
        sse.append(kmeans.inertia_)
        # print(labels)
        centroids = kmeans.cluster_centers_
        # Plot silhouette scores

      # Fit the kmeans with optimal K value
      kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(data_transformed)
      labels = kmeans.labels_
      centroids = kmeans.cluster_centers_

      plt.plot(range(2, k), silhouette_scores, marker='o')
      plt.xlabel('Number of clusters', fontsize=12)
      plt.ylabel('Silhouette score', fontsize=12)
      plt.title('Silhouette Method', fontsize=12)
      # Add grid lines
      plt.grid(True)

      # Customize tick labels
      plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)


      plt.savefig(f'{self.base_dir}/{col_name}_silhouette_plot.png', dpi=300)
      # Save the plot to a PDF file (vector format)
      plt.savefig(f'{self.base_dir}/{col_name}_silhouette_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
      plt.show()

      # Plot the sum of squared distances against k
      plt.plot(range(2, k), sse)
      plt.xlabel('Number of clusters', fontsize=12)
      plt.ylabel('Sum of squared distances', fontsize=12)
      plt.title('Elbow Method', fontsize=12)
      # Add grid lines
      plt.grid(True)

      # Customize tick labels
      plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
      plt.savefig(f'{self.base_dir}/{col_name}_elbow_plot.png')
      plt.savefig(f'{self.base_dir}/{col_name}_silhouette_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')

      plt.show()
      return kmeans, scaler

  # Function to perform event attribution and mapping the extracted events to the clusters.
  def event_attribution(self, kmeans, parametrized_events, scaler):
      # Initialize  the clusters
    n_clusters = len(kmeans.cluster_centers_)

    # clusters = {i: [] for i in range(n_clusters)}
    rows = {}
    for i,row in enumerate(parametrized_events):
        # clusters.clear()
        # key = "cls_{}".format(i)
        clusters = {i: [] for i in range(n_clusters)}
        for event in row:

            event_scaled = scaler.transform([event])
            # print(f"{event}  -> {event_scaled}")
            label = kmeans.predict(event_scaled)[0]
            # print(label)
            for cluster_label, cluster_list in clusters.items():
                if cluster_label == label:
                    cluster_list.append('yes')
                    # print(clusters)

                else:
                    cluster_list.append('no')
                    # print(clusters)
            rows[i] = clusters




    cluster_event_counts = {}
    for key, values in rows.items():
      yes_prob = []
      for k, value in values.items():
        counter_list  = Counter(value)
        count_no = counter_list['no']
        count_yes = counter_list['yes']
        # print((count_yes/count_no))
        # print(counter_list)
        # yes_prob.append(round(count_yes/(count_yes + count_no),5))
        yes_prob.append(count_yes)
      cluster_event_counts[key] = yes_prob
    return cluster_event_counts

  def merge_event_df(self, df_inc_dec, df_max_min):

    # Merge the DataFrames by index
    merged_df = df_inc_dec.merge(df_max_min, left_index=True, right_index=True)
    merged_df.head()
    # merged_df = df_max_min.copy()
    return merged_df

  def prepare_data4DT(self, merged_df, kmeans_dict=None, scaler_dict=None, for_eval=False):
  #   helper_instance = HelperClass(base_dir="results")

    appended_df = pd.DataFrame()
    count = 0
    master_dict = {}
    cluster_centroids = {}
    if kmeans_dict is None:
      kmeans_dict = {}
      scaler_dict = {}
    for col_name in merged_df.columns:
      parametrized_events = merged_df[col_name]
      # print(type(col_name))
      flatten_data =  self.flatten_nested_events(parametrized_events)
      # Extract the part of the column name before the second underscore
      col_prefix = "_".join(col_name.split("_")[:2])
      if for_eval:
        # print(kmeans_dict[col_name])
        # print(kmeans_dict[f'{col_name}'])
        kmeans = kmeans_dict[col_name]
        scaler = scaler_dict[col_name]

        attributed_data = self.event_attribution(kmeans, parametrized_events, scaler)
      else:
        kmeans, scaler = self.cluster_events(flatten_data, col_name=col_name)
        kmeans_dict[col_name] = kmeans
        scaler_dict[col_name] = scaler
        attributed_data = self.event_attribution(kmeans, parametrized_events, scaler)
      # Determine the maximum length of the values
      max_length = max(len(values) for values in attributed_data.values())

      # Generate dynamic column names
      column_names = [f"{col_name}_c{i+1}" for i in range(max_length)]

      # Convert the dictionary to DataFrame with dynamic column names
      df = pd.DataFrame(attributed_data.values(), columns=column_names)
      appended_df = pd.concat([appended_df, df], axis=1)
      post_processed_col_name, cluster_centroid = self.helper_instance.post_processed(kmeans, scaler, col_name)
      master_dict.update(post_processed_col_name)
      cluster_centroids.update(cluster_centroid)
      if 'increasing' in col_name.lower() or 'decreasing' in col_name.lower():
        self.helper_instance.plot_3D(kmeans=kmeans, flatten_events=flatten_data, scaler=scaler, col_name=col_name)
      elif 'localmax' in col_name.lower() or 'localmin' in col_name.lower():
        self.helper_instance.plot_2D(kmeans=kmeans, flatten_events=flatten_data, scaler=scaler, col_name=col_name)

    return appended_df, master_dict, cluster_centroids, kmeans_dict, scaler_dict



  def combine_data(self, X_test, kmeans_dict=None, scaler_dict=None, for_eval=False):
    df = self.preprocessing_data(X_test)
    df_inc_dec = self.extract_inc_dec_events(X_test)
    df_max_min = self.extract_local_max_min_events(X_test)
    merged_df = self.merge_event_df(df_inc_dec=df_inc_dec, df_max_min=df_max_min)
    if for_eval:
      # print(kmeans_dict)
      kmeans_dict = kmeans_dict
      scaler_dict = scaler_dict
      appended_df, master_dict, cluster_centroids, kmeans_dict, scaler_dict = self.prepare_data4DT(merged_df, kmeans_dict=kmeans_dict, scaler_dict=scaler_dict, for_eval=True)
      # print(kmeans_dict)
    else:
      appended_df, master_dict, cluster_centroids, kmeans_dict, scaler_dict = self.prepare_data4DT(merged_df)
    # master_dict = self.helper_instance.update_master_dict(master_dict)
    full_data = appended_df.copy()

    # Convert 0 to False and non-zero to True
    # full_data = full_data.astype(bool)

    # Convert False to 0 and True to 1
    # full_data = full_data.astype(int)
    # Global feature calculation
    # mean_df = global_feature_extraction(X_test)

    # full_data = pd.concat([full_data, mean_df], axis=1)
    # full_data['y'] = lstm_preds
    # tree_model = apply_dt(full_data)

    # # objective evaluation
    # data = full_data.drop('y', axis=1)
    # tree_preds = tree_model.predict(data)
    # objective_evaluation(tree_model, lstm_preds, tree_preds)
    return full_data, kmeans_dict, scaler_dict, cluster_centroids, master_dict


  def apply_dt_gs(self, data, target, class_names, master_dict):
      # Load data
      input_data = data.copy()
      # input_data = final_data.drop('y', axis=1)
      # target = final_data['y']

      # Split data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(input_data, target, test_size=0.3, stratify = target, random_state=12)

      # Train the model using C4.5 algorithm
      model = DecisionTreeClassifier(criterion='entropy', random_state=12)
      model.fit(X_train, y_train)

      # Use k-fold cross-validation to evaluate the model's performance on multiple splits of the data
      scores = cross_val_score(model, input_data, target, cv=5)  # 5-fold cross-validation

      # Print the scores for each fold
      print("Cross-validation scores:", scores)

      # Compute the mean and standard deviation of the scores
      mean_score = scores.mean()
      std_score = scores.std()
      print(f"Mean score: {mean_score*100:.1f}")
      print("Standard deviation:", std_score)

      # Evaluate the model on the test set
      y_preds = model.predict(X_test)
      train_preds = model.predict(X_train)
      train_accuracy = accuracy_score(y_train, train_preds)
      test_accuracy = accuracy_score(y_test, y_preds)
      print(f"Train Accuracy: {train_accuracy*100:.1f}")
      print(f"Test Accuracy: {test_accuracy*100:.1f}")

      # Define a grid of hyperparameters to search over
      param_grid = {
          'criterion': ['gini', 'entropy'],
          'max_depth': [2, 3, 4, 6, 8, 10, None],
          'min_samples_split': [2, 4, 6, 8, 10],
          'min_samples_leaf': [1, 2, 4, 6, 8],
          'max_features': ['sqrt', None]
      }

      # Use grid search to find the best hyperparameters
      grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
      grid_search.fit(X_train, y_train)
      line = 5* '='
      print(f' {line}Finding optimal hyperparameters using grid search{line}')
      # Print the best hyperparameters found by grid search
      print("Best hyperparameters:", grid_search.best_params_)

      # Evaluate the performance of the best model on the test set
      best_model = grid_search.best_estimator_
      gs_y_pred = best_model.predict(X_test)
      gs_train_pred = best_model.predict(X_train)
      train_accuracy = accuracy_score(y_train, gs_train_pred)
      test_accuracy = accuracy_score(y_test, gs_y_pred)
      print(f"Train Accuracy: {train_accuracy*100:.1f}")
      print(f"Test Accuracy: {test_accuracy*100:.1f}")

      # # Get feature importances from the trained decision tree classifier
      feature_importances = best_model.feature_importances_

      class_names= class_names
      tree_rules = []
      def recurse(node, depth, rule):
        if best_model.tree_.feature[node] != -2:
            feature = X_train.columns.tolist()[best_model.tree_.feature[node]]
            threshold = best_model.tree_.threshold[node]
            rule.append('({} <= {})'.format(master_dict[feature], threshold))
            recurse(best_model.tree_.children_left[node], depth + 1, rule)
            rule.pop()
            rule.append('({} > {})'.format(master_dict[feature], threshold))
            recurse(best_model.tree_.children_right[node], depth + 1, rule)
            rule.pop()
        else:
            tree_rules.append(' and '.join(rule) + ' => {}'.format(class_names[best_model.tree_.value[node].argmax()]))

      recurse(0, 0, [])
      # for rule in tree_rules:
      #     print(rule)
      fig, ax = plt.subplots(figsize=(8, 8))

      # Plot the rules as text
      for i, rule in enumerate(tree_rules):
          ax.text(0.1, 0.9 - i*0.1, rule, transform=ax.transAxes, fontsize=12)

      # Remove axis ticks and labels
      ax.axis('off')

      # Save the figure as an image
      plt.savefig(f'{self.base_dir}/rules.png', dpi=300, bbox_inches='tight')
      # Plot the decision tree
      plt.figure(figsize=(25, 20), dpi=300)
      plot_tree(best_model, feature_names=[master_dict.get(col) for col in X_train.columns], class_names= class_names)
      plt.savefig(f'{self.base_dir}/dt_graph.png', dpi=300, bbox_inches='tight')
      # Save the plot in PDF format
      plt.savefig(f'{self.base_dir}/dt_graph.pdf', format='pdf', dpi=300, bbox_inches='tight')
      plt.show()




      # # Export decision tree graph to DOT format
      # dot_data = export_graphviz(best_model, out_file=None, feature_names=[master_dict.get(col) for col in X_train.columns],
      #                           class_names=class_names, filled=True, rounded=True, special_characters=True)

      # # Modify the DOT format to reduce arrow size
      # dot_data = dot_data.replace('arrowtail=none', 'arrowtail=dot')


      # graph = graphviz.Source(dot_data)
      # graph.format = 'png'
      # graph.render(f'{self.base_dir}/dt_gs_gviz', view=True, format='png')

      # Plot feature importance
      plt.figure(figsize=(10,6), dpi=300)
      plt.bar(range(X_train.shape[1]), best_model.feature_importances_, color='skyblue')
      plt.xticks(range(X_train.shape[1]), [master_dict.get(col) for col in X_train.columns], rotation=45, ha="right")
      plt.xlabel('Feature', fontsize=12)
      plt.ylabel('Importance', fontsize=12)
      plt.title('Decision Tree Feature Importance', fontsize=12, fontweight='bold')
      plt.tight_layout()
      # Add grid lines for clarity
      plt.grid(axis='y', linestyle='--', alpha=0.7)
      # Save the plot in high quality
      plt.savefig(f'{self.base_dir}/dt_fi_plot.png', dpi=300, bbox_inches='tight')
      # Save the plot in PDF format
      plt.savefig(f'{self.base_dir}/dt_fi_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
      plt.show()
      # # Get feature importances from the trained decision tree classifier
      feature_importances = best_model.feature_importances_

      # Filter non-zero features
      important_features_dt = [X_train.columns[feature_index] for feature_index, importance in enumerate(feature_importances) if importance !=0]
      (f' {line}Logistic Regression Model {line}')
      # Train a logistic regression model
      model_lr = LogisticRegression(max_iter=1000, random_state=42)
      model_lr.fit(X_train, y_train)

      # Make predictions on the test data using the Random Forest model
      lr_y_pred = model_lr.predict(X_test)
      lr_train_pred = model_lr.predict(X_train)

      star = 5 * '*'
      print(f' {star}LR Model Accuracy {star}')
      # Evaluate model accuracy
      lr_test_accuracy = accuracy_score(y_test, lr_y_pred)
      lr_train_accuracy = accuracy_score(y_train, lr_train_pred)
      print(f"LR Train Accuracy: {lr_train_accuracy*100:.1f}")
      print(f"LR Test Accuracy: {lr_test_accuracy*100:.1f}")

      # Get feature importance scores
      feature_importance = model_lr.coef_[0]
      feature_names = [master_dict.get(col) for col in X_train.columns]

      # Binary classification
      if len(model_lr.classes_)==2:
        feature_importance_class_1 = model_lr.coef_[0]
        feature_importance_class_0 = -feature_importance_class_1
        selected_feature_names_0 = {X_train.columns[i]: importance for i, importance in enumerate(feature_importance_class_0) if importance != 0}
        selected_feature_names_1 = {X_train.columns[i]: importance for i, importance in enumerate(feature_importance_class_1) if importance != 0}
        importances = {
            model_lr.classes_[0]: selected_feature_names_0,
            model_lr.classes_[1]: selected_feature_names_1
        }
      else:
        importances = {}
        for i, class_name in enumerate(model_lr.classes_):
            class_coefficients = model_lr.coef_[i]
            selected_feature_names = {X_train.columns[j]: importance for j, importance in enumerate(class_coefficients) if importance != 0}
            importances[class_name] = selected_feature_names



      # Plot the feature importance
      plt.figure(figsize=(10, 6), dpi=300)
      plt.barh(feature_names, feature_importance, color='skyblue')
      plt.xlabel('Feature Importance', fontsize=12)
      plt.ylabel('Features', fontsize=12)
      plt.title('Logistic Regression Feature Importance', fontsize=14)
      plt.xticks(fontsize=10)  # Adjust font size of x-axis labels
      plt.yticks(fontsize=10)  # Adjust font size of y-axis labels
      plt.gca().invert_yaxis()  # Invert y-axis for better readability
      # Add grid lines for clarity
      plt.grid(axis='x', linestyle='--', alpha=0.7)

      # Save the plot in high quality
      plt.savefig(f'{self.base_dir}/lr_fi_plot.png', dpi=300, bbox_inches='tight')
      # Save the plot in PDF format
      plt.savefig(f'{self.base_dir}/lr_fi_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
      plt.show()

      return best_model, importances, important_features_dt, gs_y_pred  #selected_feature_names


  def plot_confusionmatrix(self, y_train_pred,y_train, classes, dom):
      print(f'{dom} Confusion matrix')
      cf = confusion_matrix(y_train_pred,y_train)
      sns.heatmap(cf,annot=True,yticklabels=classes
                ,xticklabels=classes,cmap='Blues', fmt='g')
      plt.tight_layout()
      plt.show()

  def apply_dt_ccp(self, data, target, class_names, master_dict, alpha=None):
      # Load data
      final_data = data.copy()
      # input_data = final_data.drop('y', axis=1)
      # target = final_data['y']

      # Split data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(final_data, target, test_size=0.3, random_state=12)

      # Train the model using C4.5 algorithm
      clf = DecisionTreeClassifier()
      path = clf.cost_complexity_pruning_path(X_train, y_train)
      ccp_alphas, impurities = path.ccp_alphas, path.impurities
      print(ccp_alphas)

      # For each alpha we will append our model to a list
      clfs = []
      for ccp_alpha in ccp_alphas:
          clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
          clf.fit(X_train, y_train)
          clfs.append(clf)

      clfs = clfs[:-1]
      ccp_alphas = ccp_alphas[:-1]
      node_counts = [clf.tree_.node_count for clf in clfs]
      depth = [clf.tree_.max_depth for clf in clfs]
      plt.figure(figsize=(5, 3))  # Increase figure size
      plt.scatter(ccp_alphas, node_counts, marker='o', color='green')
      plt.scatter(ccp_alphas, depth, marker='s', color='purple')

      plt.plot(ccp_alphas, node_counts, label='Number of Nodes', drawstyle="steps-post", linestyle='-', color='green')
      plt.plot(ccp_alphas, depth, label='Depth', drawstyle="steps-post", linestyle='-', color='purple')

      plt.legend()
      plt.title('Number of Nodes and Depth vs alpha')
      plt.xlabel('Alpha')
      plt.ylabel('Count')

      # Add grid lines for clarity
      plt.grid(axis='y', linestyle='--', alpha=0.7)

      # Save the plot in high quality
      plt.savefig(f'{self.base_dir}/nodes_and_depth_vs_alpha_plot_ccp.png', dpi=300, bbox_inches='tight')

      # Save the plot in PDF format
      plt.savefig(f'{self.base_dir}/nodes_and_depth_vs_alpha_plot_ccp.pdf', format='pdf', dpi=300, bbox_inches='tight')

      plt.show()

      train_acc = []
      test_acc = []
      for c in clfs:
          y_train_pred = c.predict(X_train)
          y_test_pred = c.predict(X_test)
          train_acc.append(accuracy_score(y_train_pred,y_train))
          test_acc.append(accuracy_score(y_test_pred,y_test))


      plt.figure(figsize=(5, 3))  # Increase figure size
      plt.scatter(ccp_alphas, train_acc, marker='o', color='blue')
      plt.scatter(ccp_alphas, test_acc, marker='s', color='orange')

      plt.plot(ccp_alphas, train_acc, label='Train Accuracy', drawstyle="steps-post", linestyle='-', color='blue')
      plt.plot(ccp_alphas, test_acc, label='Test Accuracy', drawstyle="steps-post", linestyle='-', color='orange')

      plt.legend()
      plt.title('Accuracy vs alpha')
      plt.xlabel('Alpha')
      plt.ylabel('Accuracy')

      # Add grid lines for clarity
      plt.grid(axis='y', linestyle='--', alpha=0.7)

      # Save the plot in high quality
      plt.savefig(f'{self.base_dir}/accuracy_vs_alpha_plot_ccp.png', dpi=300, bbox_inches='tight')

      # Save the plot in PDF format
      plt.savefig(f'{self.base_dir}/accuracy_vs_alpha_plot_ccp.pdf', format='pdf', dpi=300, bbox_inches='tight')

      plt.show()
      if alpha is None:
        if test_acc:
            optimal_alpha = ccp_alphas[test_acc.index(max(test_acc))]
        else:
          print("Warning: test_acc is empty. Setting default alpha.")
          optimal_alpha = 0
      else:
        optimal_alpha = alpha
      print(f'Optimal alpha: {optimal_alpha}')

      clf_ = DecisionTreeClassifier(random_state=0,ccp_alpha=optimal_alpha)
      clf_.fit(X_train,y_train)
      y_train_pred = clf_.predict(X_train)
      y_test_pred = clf_.predict(X_test)
      
      train_accuracy = accuracy_score(y_train_pred,y_train)
      test_accuracy = accuracy_score(y_test_pred,y_test)

      print(f'Train score {accuracy_score(y_train_pred,y_train):.2f}')
      print(f'Test score {accuracy_score(y_test_pred,y_test):.2f}')
      self.plot_confusionmatrix(y_train_pred,y_train, class_names, dom='Train')
      self.plot_confusionmatrix(y_test_pred,y_test, class_names, dom='Test')

      # Extract feature importances
      feature_importances = clf_.feature_importances_

      # Filter non-zero features
      important_features = [X_train.columns[feature_index] for feature_index, importance in enumerate(feature_importances) if importance !=0]

      # Plot feature importance
      plt.figure(figsize=(10,6), dpi=300)
      plt.bar(range(X_train.shape[1]), clf_.feature_importances_, color='skyblue')
      plt.xticks(range(X_train.shape[1]), [master_dict.get(col) for col in X_train.columns], rotation=45, ha="right")
      plt.xlabel('Feature', fontsize=12)
      plt.ylabel('Importance', fontsize=12)
      plt.title('Decision Tree Feature Importance', fontsize=12, fontweight='bold')
      plt.tight_layout()
      # Add grid lines for clarity
      plt.grid(axis='y', linestyle='--', alpha=0.7)
      # Save the plot in high quality
      plt.savefig(f'{self.base_dir}/dt_fi_plot_ccp.png', dpi=300, bbox_inches='tight')
      # Save the plot in PDF format
      plt.savefig(f'{self.base_dir}/dt_fi_plot_ccp.pdf', format='pdf', dpi=300, bbox_inches='tight')
      plt.show()


      plt.figure(figsize=(20,20))
      features = [master_dict[feature] for feature in X_train.columns]
      plot_tree(clf_,feature_names=features,class_names=class_names,filled=True)
      plt.savefig(f'{self.base_dir}/dt_graph_ccp.png', dpi=300, bbox_inches='tight')
      # Save the plot in PDF format
      plt.savefig(f'{self.base_dir}/dt_graph_ccp.pdf', format='pdf', dpi=300, bbox_inches='tight')
      plt.show()

      # Export decision tree graph to DOT format
      dot_data = export_graphviz(clf_, out_file=None, feature_names=features,
                                class_names=class_names, filled=True, rounded=True, special_characters=True)

      # Modify the DOT format to reduce arrow size
      dot_data = dot_data.replace('arrowtail=none', 'arrowtail=dot')


      graph = graphviz.Source(dot_data)
      graph.format = 'png'
      graph.render(f'{self.base_dir}/dt_gviz', view=False, format='png')

      tree_rules = []
      max_line_length = 200  # Adjust this value based on the desired maximum line

      def format_rule_line(rule_line):
          return '\n'.join([rule_line[i:i + max_line_length] for i in range(0, len(rule_line), max_line_length)])

      def recurse(node, depth, rule):
          if clf_.tree_.feature[node] != -2:
              feature = X_train.columns.tolist()[clf_.tree_.feature[node]]
              threshold = clf_.tree_.threshold[node]

              left_rule = '{} <= {}'.format(master_dict[feature], threshold)
              rule.append(left_rule)
              recurse(clf_.tree_.children_left[node], depth + 1, rule)
              rule.pop()

              right_rule = '{} > {}'.format(master_dict[feature], threshold)
              rule.append(right_rule)
              recurse(clf_.tree_.children_right[node], depth + 1, rule)
              rule.pop()
          else:
              class_prediction = class_names[clf_.tree_.value[node].argmax()]
              formatted_rule = ' => {} \n'.format(class_prediction)
              tree_rules.append(' and '.join(rule) + formatted_rule)

      recurse(0, 0, [])

      fig, ax = plt.subplots(figsize=(8, 8))

      for i, rule in enumerate(tree_rules):
          ax.text(0.1, 0.9 - i * 0.1, rule, transform=ax.transAxes, fontsize=12, fontweight='bold', family='monospace')

      ax.axis('off')
      plt.savefig(f'{self.base_dir}/rules.png', dpi=300, bbox_inches='tight')
      return clf_, round(test_accuracy, 2), important_features, y_test_pred


  def main(self, full_data, lstm_preds, class_names, X_test, cluster_centroids, master_dict, only_dt=False, ccp_alpha=None):
    # full_data['y'] = lstm_preds
    if not only_dt:
      tree_model, important_features_lr, important_features_dt, y_preds = self.apply_dt_gs(full_data, lstm_preds, class_names, master_dict)
      # objective evaluation
      # data = full_data.drop('y', axis=1)
      tree_preds = tree_model.predict(full_data)
      fidelity_score, depth, n_nodes = objective_evaluation(tree_model, lstm_preds, tree_preds)
      self.helper_instance.plot_events_on_timeseries(X_test, lstm_preds, important_features_lr, cluster_centroids, class_names)
      self.helper_instance.plot_events_as_line_on_timeseries(X_test, lstm_preds, important_features_lr, cluster_centroids, class_names)
      # self.helper_instance.plot_events_as_line_on_timeseries1(X_test, lstm_preds, important_features_lr, cluster_centroids, class_names)
      print(important_features_dt)
      return important_features_dt, y_preds, fidelity_score, depth, n_nodes

    else:
      tree_model, test_acc, important_features, y_preds = self.apply_dt_ccp(full_data, lstm_preds, class_names, master_dict, ccp_alpha)
      # objective evaluation
      # data = full_data.drop('y', axis=1)
      tree_preds = tree_model.predict(full_data)
      fidelity_score, depth, n_nodes = objective_evaluation(tree_model, lstm_preds, tree_preds)
      return important_features, y_preds, fidelity_score, depth, n_nodes
    
    
  def ts_global_rule_explanation(self, X_test, model_preds, class_names, ccp_alpha= None, kmeans_dict=None, scaler_dict=None, for_eval=False):
    df = self.preprocessing_data(X_test)
    df_inc_dec = self.extract_inc_dec_events(X_test)
    df_max_min = self.extract_local_max_min_events(X_test)
    merged_df = self.merge_event_df(df_inc_dec=df_inc_dec, df_max_min=df_max_min)
    if for_eval:
      # print(kmeans_dict)
        kmeans_dict = kmeans_dict
        scaler_dict = scaler_dict
        final_data, master_dict, cluster_centroids, kmeans_dict, scaler_dict = self.prepare_data4DT(merged_df, kmeans_dict=kmeans_dict,
                                                                                                  scaler_dict=scaler_dict, for_eval=True)
      # print(kmeans_dict)
    else:
        final_data, master_dict, cluster_centroids, kmeans_dict, scaler_dict = self.prepare_data4DT(merged_df)
    
    tree_model, test_acc, important_features, y_preds = self.apply_dt_ccp(final_data, model_preds, class_names, master_dict, ccp_alpha)
    # objective evaluation
    tree_preds = tree_model.predict(final_data)
    fidelity_score, depth, n_nodes = objective_evaluation(tree_model, model_preds, tree_preds)
    return  test_acc, fidelity_score, depth, n_nodes



