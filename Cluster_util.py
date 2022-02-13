from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from statistics import mean, mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal 
from itertools import permutations
from sklearn.metrics import silhouette_score

def df_col_diff(df, columns):
    """Places the difference betweeen to rows of dataframe in a new column named with '_diff' """
    for i in range(len(columns)): 
        col = columns[i]
        name = col+'_diff'
        df[name] = df[col].diff()
    return df

def align_df(QOS_df, usage_df):
    """Aligns the  QOS and Usage dataframe to neasrest match in time stamps row for row"""
    time_diff = []
    shift = len(QOS_df[QOS_df['QOS'] == 0]) #variable to skip rows in QOS data with zero value

    if QOS_df['QOS'][shift] > 5 * QOS_df['QOS'][shift:].mean(): #At times the first QOS obtained is an outlier, so this condition removes it if the first value is 5 times above the average
        QOS_df = QOS_df[shift+1:] #skip rows in QOS data with zero value plus the first value if it is an outlier
    else:
        QOS_df = QOS_df[shift:] #skip rows in QOS data with zero value

    usage_df_temp = usage_df[shift:] # create copy of usage df as a reference to measure timing mismatch
    usage_df_temp = usage_df_temp[:len(QOS_df)] # shorten copy of usage df to length of QOS df
    min_len = min(len(QOS_df),len(usage_df_temp)) # get length to use for timing mismatch calculation in following loop, they should be the same length anyway, but just a precaution
    
    for i in range(min_len-1):  #timing mismatch calculation
        diff = QOS_df['time'].iloc[i] - usage_df_temp['time'].iloc[i]
        time_diff.append(diff)
    
    # select shift value that resolves timing mismatch
    # there's probably a better way, but meh ... it works well enough, unless the mismatch is more than 30 sec, which shouldn't happen at all
    if mean(time_diff) < -30:
        usage_df = usage_df[shift-7:]
    elif -30 <= mean(time_diff) < -25:
        usage_df = usage_df[shift-6:]
    elif -25 <= mean(time_diff) < -20:
        usage_df = usage_df[shift-5:]
    elif -20 <= mean(time_diff) < -15:
        usage_df = usage_df[shift-4:]
    elif -15 <= mean(time_diff) < -10:
        usage_df = usage_df[shift-3:]
    elif -10 <= mean(time_diff) < -5:
        usage_df = usage_df[shift-2:]
    elif -5 <= mean(time_diff) < -2.5:
        usage_df = usage_df[shift-1:]
    elif -2.5 <= mean(time_diff) <= 2.5:
        usage_df = usage_df[shift:]
    elif 2.5 < mean(time_diff) <= 5:
        usage_df = usage_df[shift+1:]
    elif 5 < mean(time_diff) <= 10:
        usage_df = usage_df[shift+2:]
    elif 10 < mean(time_diff) <= 15:
        usage_df = usage_df[shift+3:]
    elif 15 < mean(time_diff) <= 20:
        usage_df = usage_df[shift+4:]
    elif 20 < mean(time_diff) <= 25:
        usage_df = usage_df[shift+5:]
    elif 25 < mean(time_diff) <= 30:
        usage_df = usage_df[shift+6:]
    else:
        usage_df = usage_df[shift+7:]

    return QOS_df, usage_df[:len(QOS_df)]

def dataPrep(rpi_name, data_name, cols):
    """Obtaines data and prepares it for use"""
    # read data from csvs in their respecive directories
    usage_data = pd.read_csv(r"C:/Users/USER/Desktop/Queen's University/Research/rpi_benchmark_profile/data/{}/usage_data_{}_{}.csv".format(rpi_name,rpi_name,data_name), index_col = 'time_stamp')
    QOS_data = pd.read_csv(r"C:/Users/USER/Desktop/Queen's University/Research/rpi_benchmark_profile/data/{}/QOS_data_{}_{}.csv".format(rpi_name,rpi_name,data_name), index_col = 'time_stamp')
    RCoin = pd.read_csv(r"C:/Users/USER/Desktop/Queen's University/Research/rpi_benchmark_profile/data/{}/RCoin_{}_{}.csv".format(rpi_name,rpi_name,data_name))

    # align the QOS and usage data in terms of time stamp row for row and total length
    QOS_data, usage_data = align_df(QOS_data, usage_data)

    # add differece values to the dataframe for two consecutive rows for features specified in 'cols'
    usage_data = df_col_diff(usage_data, cols)

    usage_data['QOS'] = QOS_data['QOS'].tolist() # place QOS measurements in usage data
    QOS_data['model_number']= QOS_data['model_number'].astype(int)
    usage_data['model_number'] = QOS_data['model_number'].tolist() # place QOS average measurements in usage data
    
    try:
        usage_data['QOS_avg'] = QOS_data['QOS_avg'].tolist() # place QOS measurements in usage data
        usage_data['task'] = QOS_data['task'].tolist() # place QOS average measurements in usage data
    except:
        pass

    usage_data.fillna(method ='bfill', inplace = True) # fill nan values with next value

    for i in range(len(RCoin)):
        usage_data.loc[usage_data.model_number == i+1, "Training Time"] = RCoin['Training Times'][i]

    return [usage_data, rpi_name +'_'+ data_name, RCoin]

def dbscan(usage_QOS_listofLists, epsilon, num_samples, num_PCA_comp, columns):
    """Clusters data using DBSCAN method"""
    
    scaler = StandardScaler()
    usage_normalized_list = []

    for usage_QOS_list in usage_QOS_listofLists: # for each set of QOS and Usage data
        
        usage_QOS_list[0]['QOS'] = usage_QOS_list[1]['QOS'].tolist() # place QOS measurements in usage data
        usage_QOS_list[0]['QOS_avg'] = usage_QOS_list[1]['QOS_avg'].tolist() # place QOS average measurements in usage data

        usage_QOS_list[0].fillna(method ='bfill', inplace = True) # fill nan values with next value
   
        usage_normalized_list.append(normalize(scaler.fit_transform(usage_QOS_list[0][columns]))) # Scale usage data for specified features only

    

    if num_PCA_comp == 0: # cluster based on density without using Prinicipal Component Analysis (PCA)
        
        X_normalized = np.concatenate(tuple(usage_normalized_list), axis=0)

        X_principal = pd.DataFrame(X_normalized)
    else:

        X_normalized = np.concatenate(tuple(usage_normalized_list), axis=0)

        X_normalized = pd.DataFrame(X_normalized)

        pca = PCA(n_components = num_PCA_comp) 
        X_principal = pca.fit_transform(X_normalized) # Obtain 'num_PCA_comp' Principal Components 
        X_principal = pd.DataFrame(X_principal)

    db_default = DBSCAN(eps = epsilon, min_samples = num_samples).fit(X_principal) # Compute DBSCAN Clusters according to radius epsilon with num_samples within radius
    labels = db_default.labels_ # gets cluster labels

    return list(labels) 

def generate_grid(eps_min,eps_max, eps_step, samples_min, samples_max, samples_step,pc_min, pc_max, pc_step):
    """ Generate radii, nuumber of samples. and number of principal components to be usge to obstain clusters with DBSCAN"""
    # gets the number of decimal points of step size to get rounding scale
    d = Decimal(str(eps_step))
    roundingNum = - d.as_tuple().exponent
    
    grid_search_list = []

    for ep in np.arange(start=eps_min, stop=eps_max, step=eps_step):
        for sample in np.arange(start=samples_min, stop=samples_max, step=samples_step):
            for num_pc in  np.arange(start=pc_min, stop=pc_max, step=pc_step):
                grid_search_list.append([round(ep, roundingNum),sample,num_pc])

    return grid_search_list
  

def slice_labels(data_list, labels):
    """Slice labels list according to length of each dataset"""
    labels_copy = labels.copy()
    
    for data in data_list:
        data.append(labels_copy[:len(data[0])])
        labels_copy = labels_copy[len(data[0]):]

    return  data_list  
        
def print_slice_labels(data_list, labels):
    labels_copy = labels.copy()
    
    for data in data_list:
        printable = labels_copy[:len(data[0])]
        mostCommonElement = mode(printable)
        print(data[-1])
        print((printable.count(mostCommonElement) / len(data[0])) * 100, '%', mostCommonElement)
        print(printable)
        data.append(printable)
        labels_copy = labels_copy[len(data[0]):]

def check_slice_labels(data_list, labels):
    labels_copy = labels.copy()
    total_diff = 0
    for data in data_list:
        printable = labels_copy[:len(data[0])]
        mostCommonElement = mode(printable)
        total_diff += len(data[0]) - printable.count(mostCommonElement)
        labels_copy = labels_copy[len(data[0]):]
    return total_diff

def run_KMeans(data_list, feature_columns, grid, num_PCA_comp = 0):
    

    usage_normalized_list = []

    for usage_QOS_list in data_list: # for each set of QOS and Usage data
        
        usage_QOS_list[0]['QOS'] = usage_QOS_list[1]['QOS'].tolist() # place QOS measurements in usage data
        usage_QOS_list[0]['QOS_avg'] = usage_QOS_list[1]['QOS_avg'].tolist() # place QOS average measurements in usage data

        usage_QOS_list[0].fillna(method ='bfill', inplace = True) # fill nan values with next value
   
        usage_normalized_list.append(normalize(usage_QOS_list[0][feature_columns])) # Scale usage data for specified features only

    if num_PCA_comp == 0: # cluster based on density without using Prinicipal Component Analysis (PCA)
        
        X_normalized = np.concatenate(tuple(usage_normalized_list), axis=0)

        X_principal = pd.DataFrame(X_normalized)
    else:

        X_normalized = np.concatenate(tuple(usage_normalized_list), axis=0)

        X_normalized = pd.DataFrame(X_normalized)

        pca = PCA(n_components = num_PCA_comp) 
        X_principal = pca.fit_transform(X_normalized) # Obtain 'num_PCA_comp' Principal Components 
        X_principal = pd.DataFrame(X_principal)

    total_diff_list = []
    for num_clust in grid:
        kmeans = KMeans(n_clusters = num_clust, init = 'k-means++').fit(X_principal) # Compute DBSCAN Clusters according to radius epsilon with num_samples within radius
        total_diff = check_slice_labels(data_list, list(kmeans.labels_))
        total_diff_list.append(total_diff)
    
    minIndex = total_diff_list.index(min(total_diff_list))
    kmeans = KMeans(n_clusters = grid[minIndex], init = 'k-means++').fit(X_principal)
    kmeans.predict
    print('silhouette coefficient:', silhouette_score(X_principal, kmeans.labels_))

    return list(kmeans.labels_), list(kmeans.cluster_centers_)

    
