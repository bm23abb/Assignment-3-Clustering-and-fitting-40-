import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.optimize import curve_fit

# Function to read both data files
def both_data():
    """
    read both data file
     
    Return
    -------
    data1 : pandas DataFrame, the first dataset
    data2 : pandas DataFrame, the second dataset

    """
    data1 = pd.read_csv("Forest area.csv", skiprows=4)
    data2 = pd.read_csv("Population growth.csv", skiprows=4)
    
    
    return data1, data2 

# Function to clean and merge data for clustering
def prepare_data_for_clustering(data1, data2):
    """
    Clean and merge data for clustreing

    Parameters
    ----------
    data1 : pandas DataFrame, the first dataset
    data2 : pandas DataFrame, the second dataset
    Returns
    -------
    TYPE
        pandas DataFrame, Merged and cleaned dataset
        for clustering

    """
    forest_data = data1[['Country Name', 'Country Code', '2002']].copy()
    population_data = data2[['Country Name', 'Country Code', '2002']].copy()

    data_of = pd.merge(forest_data, population_data, on="Country Name", how="outer")
    data_of = data_of.dropna()
    data_of = data_of.rename(columns={"2002_x": "forest_data", "2002_y": "population_data"})

    # Select only the necessary columns for clustering
    selected_columns = ['forest_data', 'population_data']
    return data_of[selected_columns]

# Function to normalize data
def normalize_data(data):
    """
    Normalize of the data

    Parameters
    ----------
    data : pandas.DataFrame,
    the dataset
     

    Returns
    -------
    normalized_data_df : pandas.DataFrame,
    Normalize dataset

    """
    numeric_col = data . select_dtypes (include=[np.number]). columns
    Simple = SimpleImputer(strategy='mean')
    imputed_data = Simple.fit_transform(data[numeric_col])
    normalized_datas = StandardScaler().fit_transform(imputed_data)
    normalized_data_df = pd.DataFrame(normalized_datas, columns=['forest_data', 'population_data'])
    return normalized_data_df

# Function to perform k-means clustering
def perform_clustering(data, n_clusters=6):
    """
   Performing k-means cluster.

  
   data : pandas DataFrame
       The dataset for cluster.
   n_clusters : int, optional
       The number of clusters

   Returns
   -------
   cluster_labels : numpy ndarray
       lable for each other data point.
   centers : numpy.ndarray
       Cluster centers.
   """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")
    return cluster_labels, kmeans.cluster_centers_

# Function to plot clustering results
def plot_clusters(data, cluster_labels, centers):
    """
   Plot clustering results.

   Parameters
   ----------
   data : pandas.DataFrame
       The dataset.
   cluster_labels : numpy.ndarray
       Labels for each data point.
   centers : numpy.ndarray
       Cluster centers.
   """
    plt.figure(figsize=(6, 5))
    cm = plt.cm.get_cmap('tab10')
    for i, label in enumerate(np.unique(cluster_labels)):
        cluster_data = data[cluster_labels == label]
        plt.scatter(cluster_data['forest_data'], cluster_data['population_data'],
                    10, label="Cluster {}".format(label), cmap=cm, alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], 50, "k", marker="D",
                label="Cluster centers")
    plt.xlabel("Forest Area")
    plt.ylabel("Population Growth")
    plt.title("Kmeans Clustering")
    plt.legend()
    plt.savefig("Kmeans Clustering.png", dpi=300)
    plt.show()
    
def generate_prediction():
    """
    Generate CO2 emission predictions.

    Returns
    -------
    data_co2 : pandas.DataFrame
        CO2 emission data.
    """
    co2 = pd.read_csv("CO2 emissions.csv", skiprows=4)
    co2 = co2.set_index('Country Name', drop=True)
    co2 = co2.loc[:, '1990':'2021']
    co2 = co2.transpose()
    co2 = co2.loc[:, 'United States']
    data = co2.dropna(axis=0, how="all")
    
    data_co2 = pd.DataFrame()
    data_co2["Year"] = pd.DataFrame(data.index)
    data_co2["CO2 emissions.csv"] = pd.DataFrame(data.values)
    
    return data_co2

data_co2 = generate_prediction()

def Expo(t, n0, r):
    """
   Exponential function for curve fitting.

   Parameters
   ----------
   t : numpy.ndarray
       Time points.
   n0 : float
       Initial value.
   r : float
       Growth rate.

   Returns
   -------
   numpy.ndarray
       Exponential values.
   """
    return n0 * np.exp(r * (t - data_co2["Year"].min()))

data_co2["Year"] = pd.to_numeric(data_co2["Year"])

param, covariance = curve_fit(Expo, data_co2["Year"], data_co2["CO2 emissions.csv"], p0=(1.2e12, 0.03))
n0_fit, r_fit = param

prediction_years = np.arange(1990, 2022)
prediction_values = Expo(prediction_years, *param)

stderr = np.sqrt(np.diag(covariance))
conf_interval = 1.96 * stderr
upper = Expo(prediction_years, *(param + conf_interval))
lower = Expo(prediction_years, *(param - conf_interval))

future_years = np.arange(2021, 2027)
prediction_future = Expo(future_years, *param)

plt.figure(figsize=(10, 6))
plt.plot(data_co2["Year"], data_co2["CO2 emissions.csv"], label="Historical Data", marker='o', linestyle='-', color='blue')
plt.plot(prediction_years, prediction_values, label="prediction", marker='o', linestyle='--', color='orange')
plt.plot(future_years, prediction_future, label="Next 6 Years prediction", marker='o', linestyle='--', color='green')
plt.fill_between(prediction_years, upper, lower, color='purple', alpha=0.2, label="95% Confidence Interval")
plt.xlabel("Year")
plt.ylabel("CO2 emissions")
plt.title("Exponential Predictions for CO2 Emissions in the United States")
plt.legend()
plt.savefig("CO2 emissions.png")
plt.show()

# Function to use the elbow approach to determine the best clusters
def find_optimal_clusters_elbow(data, max_clusters=10):
    inertias = []

    transposed_data = data.transpose()

    num_samples = transposed_data.shape[0]
    num_clusters = min(num_samples, max_clusters)

    for i in range(1, num_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(transposed_data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, num_clusters + 1), inertias, marker='o')
    plt.title('Elbow method for Optimal k')
    plt.xlabel('Number of cluster')
    plt.ylabel('Inertia')
    plt.show()

# Main code
data1, data2 = both_data()
concat_data = prepare_data_for_clustering(data1, data2)
normalized_data_df = normalize_data(concat_data)
find_optimal_clusters_elbow(normalized_data_df)

# Choose the ideal cluster size determined by the elbow method
optimal_clusters = 3
cluster_labels, centers = perform_clustering(normalized_data_df, n_clusters=optimal_clusters)

# Add cluster labels to the DataFrame
data_of_with_labels = data2.copy()
data_of_with_labels["labels"] = np.nan
data_of_with_labels.loc[:len(cluster_labels) - 1, "labels"] = cluster_labels

# Reset the index if 'Country Name' is used as an index
if 'Country Name' in data_of_with_labels.index:
    data_of_with_labels.reset_index(inplace=True)

# Plot clustering results
plot_clusters(normalized_data_df, cluster_labels, centers)