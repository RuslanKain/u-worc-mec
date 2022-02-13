from numpy.core.fromnumeric import std
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from statistics import mean, mode
import numpy as np
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
import itertools

class cluster():
    
    def __init__(self, rpi_names, data_names, diff_columns, feature_columns, centorid_type = 'median', norm = True, wait_interval = 0):
        
        self.cluster_RPi_data_list, self.test_RPi_data_list, self.centroids = [], [], []
        self.labeled_centroids = {}
        self.diff_columns = diff_columns
        self.feature_columns = feature_columns

        self.rpi_names = rpi_names
        self.data_names = data_names

        for rpi in self.rpi_names:
            for dt in self.data_names: 

                cluster_RPi_data = self.dataPrep(rpi, dt, self.diff_columns)
                test_RPi_data = self.dataPrep(rpi, dt+'_2', self.diff_columns)
                
                self.cluster_RPi_data_list.append(cluster_RPi_data)
                self.test_RPi_data_list.append(test_RPi_data)
        

        self.centorid_type = centorid_type
        self.norm = norm
        self.wait_interval = wait_interval
    
    def _df_col_diff(self, df, columns):
        """Places the difference betweeen to rows of dataframe in a new column named with '_diff' """
        for i in range(len(columns)): 
            col = columns[i]
            name = col+'_diff'
            df[name] = df[col].diff()
        return df

    def _align_df(self, QOS_df, usage_df):
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

    def dataPrep(self, rpi_name, data_name, cols):
        """Obtaines data and prepares it for use"""
        # read data from csvs in their respecive directories
        usage_data = pd.read_csv('data/{}/usage_data_{}_{}.csv'.format(rpi_name,rpi_name,data_name), index_col = 'time_stamp')
        QOS_data = pd.read_csv('data/{}/QOS_data_{}_{}.csv'.format(rpi_name,rpi_name,data_name), index_col = 'time_stamp')
        RCoin = pd.read_csv('data/{}/RCoin_{}_{}.csv'.format(rpi_name,rpi_name,data_name))

        # align the QOS and usage data in terms of time stamp row for row and total length
        QOS_data, usage_data = self._align_df(QOS_data, usage_data)

        # add differece values to the dataframe for two consecutive rows for features specified in 'cols'
        usage_data = self._df_col_diff(usage_data, cols)

        # calculate % cpu usage for process running the task 
        usage_data['cpu_times_ipykernel_percent'] = ((usage_data['cpu_times_user_ipykernel_diff']+usage_data['cpu_times_system_ipykernel_diff'])/(usage_data['cpu_user_time_diff']+usage_data['cpu_system_time_diff']))*100

        usage_data['QOS'] = QOS_data['QOS'].tolist() # place QOS measurements in usage data
        QOS_data['model_number']= QOS_data['model_number'].astype(int)
        usage_data['model_number'] = QOS_data['model_number'].tolist() # place model number (representing benchmark run in usage data
        
        try:
            usage_data['QOS_avg'] = QOS_data['QOS_avg'].tolist() # place average QOS measurements in usage data
            usage_data['task'] = QOS_data['task'].tolist() # place benchmark  task type name in usage data
        except:
            pass

        usage_data.fillna(method ='bfill', inplace = True) # fill nan values with next value
    
        for i in range(len(RCoin)):
            usage_data.loc[usage_data.model_number == i+1, "Training Time"] = RCoin['Training Times'][i]

        return [usage_data, rpi_name +'_'+ data_name, RCoin]       

    def _run_KMeans_multiClusters(self, data_list, num_PCA_comp = 0):
        
        data_normalized_list = []

        for data in data_list: # for each set of QOS and Usage data
            
            data_normalized_list.append(normalize(data[0][self.feature_columns])) # Scale usage data for specified features only

        if num_PCA_comp == 0: # cluster based on density without using Prinicipal Component Analysis (PCA)
            
            X_normalized = np.concatenate(tuple(data_normalized_list), axis=0)

            X_principal = pd.DataFrame(X_normalized)
        else:

            X_normalized = np.concatenate(tuple(data_normalized_list), axis=0)

            X_normalized = pd.DataFrame(X_normalized)

            pca = PCA(n_components = num_PCA_comp) 
            X_principal = pca.fit_transform(X_normalized) # Obtain 'num_PCA_comp' Principal Components 
            X_principal = pd.DataFrame(X_principal)

        kmeans = KMeans(n_clusters = len(data_list), init = 'k-means++').fit(X_principal)

        print('silhouette coefficient:', silhouette_score(X_principal, kmeans.labels_))

        return kmeans.cluster_centers_.tolist()

    def _get_centroid(self, data, runs, norm = True, type = 'median'):

        if norm == True:
            X = pd.DataFrame(normalize(data[self.feature_columns].loc[data['model_number']<=runs]))
        else:
            X = data[self.feature_columns].loc[data['model_number']<=runs]

        if type == 'median':
            centroid = X.median()
        elif type == 'mean':
            centroid = X.mean()

        return centroid.tolist()

    def _distance_to_centroid(self, row, centroids, norm):
        centroid_dist = []
        if norm == False:
            for centroid in centroids:
                centroid_dist.append(np.linalg.norm([row.tolist(),centroid]))
        else:
            for centroid in centroids:
                centroid_dist.append(cosine_similarity(np.array(row.tolist()).reshape(1, -1),np.array(centroid).reshape(1, -1)).tolist()[0][0])
            
        return centroid_dist

    def _get_accuracy(self, verbose):
        
        data_batch, acc_list = [], []

        for test_data in self.test_RPi_data_list:

            clusterd_new_data = {'time_stamp':[],'data':[],'predicted_cluster':[], 'actual_cluster':[]}
            
            for index, row in test_data[0].iterrows():
                row = row[self.feature_columns]
                if self.norm == True:
                    data_batch.append(row.tolist())
                    clusterd_new_data['time_stamp'].append(index)
                    clusterd_new_data['data'].append(row.tolist())
                    norm_row = normalize(np.array(row.tolist()).reshape(1, -1))[0]

                    centroid_similatiry = self._distance_to_centroid(norm_row, self.centroids, self.norm)
                    assigned_cluster = self.cluster_RPi_data_list[centroid_similatiry.index(max(centroid_similatiry))][1] 
                    
                    clusterd_new_data['predicted_cluster'].append( assigned_cluster )
                    clusterd_new_data['actual_cluster'].append( test_data[1] )
                    
                    if self.wait_interval != 0:
                        print(index, assigned_cluster)
                        sleep(self.wait_interval)

                else:

                    clusterd_new_data['time_stamp'].append(index)
                    clusterd_new_data['data'].append(row.tolist())

                    centroid_distance = self._distance_to_centroid(row, self.centroids, self.norm)
                    assigned_cluster = self.cluster_RPi_data_list[centroid_distance.index(min(centroid_distance))][1] 
                    clusterd_new_data['predicted_cluster'].append( assigned_cluster )
                    clusterd_new_data['actual_cluster'].append( test_data[1] )

                    
                    if self.wait_interval != 0:
                        print(index, assigned_cluster)
                        sleep(self.wait_interval)

            
            new_data = pd.DataFrame(clusterd_new_data).set_index('time_stamp')


            accuracy = (len(new_data.loc[new_data.predicted_cluster == test_data[1].split('_2')[0]])/len(new_data['predicted_cluster'] ))*100

            print('{}% accuracy for cluster'.format(round(accuracy,2)), test_data[1].split('_2')[0])
            acc_list.append(round(accuracy,2))

        
            if verbose == True:
                print(new_data['predicted_cluster'].value_counts())
        
        avg_acc = round(mean(acc_list),2)
        std_acc = round(std(acc_list),2)

        print('Average accuracy: {}%, Standard Dev.: {}%'.format(avg_acc,std_acc))

        return new_data, avg_acc, std_acc

    def run_clustering(self, runs, verbose = False):
        """Accuracy of resource state clustering test"""
        print('\nU-WORC values for {} runs'.format(runs))
        if self.centorid_type == 'kmeans':
        
            centroidss = self._run_KMeans_multiClusters(self.cluster_RPi_data_list)
            a = 0
            for RPi_data in self.cluster_RPi_data_list:
                self.centroids.append(centroidss[a])
                self.labeled_centroids[RPi_data[1]] = centroidss[a]
                a =+ 1
        else:
            for RPi_data in self.cluster_RPi_data_list:
                new_centroid = self._get_centroid(RPi_data[0], runs, self.norm, type = self.centorid_type)
                self.centroids.append(new_centroid)
                self.labeled_centroids[RPi_data[1]] = new_centroid

        return self._get_accuracy(verbose)

    

    def select_using_RCoin(self, device_state_comb, RCoins):
              
        RCoin_device_best = max(RCoins, key=RCoins.get)
        
        for device in device_state_comb:
            if RCoin_device_best.split("_")[0] in device:
                return device 
        
    def select_using_cluster(self, device_state_comb, states, verbose = False):
        
        states_subset = {key: states[key] for key in device_state_comb}

        if verbose == True:
            print('RS QOS:',states_subset)

        return min(states_subset, key=states_subset.get)

    def task_alloc_eval(self, RCoin_states, verbose = False):
        """Task allocation evaluation for the specified initial state in RCoin_states"""
        combos = []
        resource_states, RCoin_devices = {}, {}

        for data_list in self.cluster_RPi_data_list: # create dicts lists 1) resource state with QOS 2) devices with RCoinP
            
            device_state = data_list[1]
            avg_QOS = round(data_list[0]['QOS'].median(),2)
            resource_states[device_state] = avg_QOS
        
            # create RCoin dict based on chosen resource state per device
            if "RPi8_1500_{}".format(RCoin_states[0]) in data_list[1]:
                RCoinP = int(data_list[2]['RCoinP'].iloc[-2])
                RCoin_devices[device_state]= RCoinP
            elif "RPi4_1000_{}".format(RCoin_states[1]) in data_list[1]:
                RCoinP = int(data_list[2]['RCoinP'].iloc[-2])
                RCoin_devices[device_state]= RCoinP
            elif "RPi2_600_{}".format(RCoin_states[2]) in data_list[1]:
                RCoinP = int(data_list[2]['RCoinP'].iloc[-2])
                RCoin_devices[device_state]= RCoinP

        if verbose == True:
            print('RCoinP:', RCoin_devices,'\n')

        for rpi_name in self.rpi_names: # create list of lists, where each list contains a permutation of the three devices and different states (devices don't repeat)
            combo = []
            for data_name in self.data_names:
                combo.append(rpi_name+"_"+data_name)
            combos.append(combo)

        all_combos = list(itertools.product(*combos)) # get all possible permutations of the lists of 3 lists created 

        RCoin_expected_avg_training_times_list, cluster_expected_avg_training_times_list = [], []

        for comb in all_combos: # run task allocation evaluation 
            if verbose == True:
                print('RPi State Combination:',comb)

            RCoin_choice = self.select_using_RCoin(comb, RCoin_devices)
            cluster_choice = self.select_using_cluster(comb, resource_states, verbose)

            for data_list in self.test_RPi_data_list: # get average training time from 2nd dataset based on choices
                if data_list[1] == RCoin_choice+"_2":
                    RCoin_expected_avg_training_time = round(data_list[2]['Training Times'][:].mean(),2)
                    RCoin_expected_avg_training_times_list.append(RCoin_expected_avg_training_time)

                    if verbose == True:
                        print('RCoinP Choice:', RCoin_choice.split("_")[:2], 'Avg. ML Training Time:',RCoin_expected_avg_training_time)

                if data_list[1] == cluster_choice+"_2":
                    cluster_expected_avg_training_time = round(data_list[2]['Training Times'][:].mean(),2)
                    cluster_expected_avg_training_times_list.append(cluster_expected_avg_training_time)

                    if verbose == True:
                        print('RS QOS Choice:', cluster_choice.split("_")[:2], 'Avg. ML Training Time:',cluster_expected_avg_training_time,'\n')    


        print('RCoin choice average ML model Training Time', round(mean(RCoin_expected_avg_training_times_list),2), 'seconds')
        print('RS QOS choice average ML model Training Time', round(mean(cluster_expected_avg_training_times_list),2), 'seconds')

        
        data_plot = [RCoin_expected_avg_training_times_list,cluster_expected_avg_training_times_list]

        plt.rcParams.update({'font.size': 15})

        plt.boxplot(data_plot, meanprops = dict(linestyle='-', linewidth=2.5, color='red'),  medianprops= dict(linestyle='-', linewidth=2.5, color='blue'),showmeans = True, meanline=True)
        plt.xticks([1, 2], ['RCoinP', 'U-WORC'])
        plt.ylabel('ML Model Training Time (sec.)')
        plt.show()

    def plot_acc_per_runs(self, accuracy_list, stdv_list, runs, type = 'line'):
        
        if type == 'line':
            plt.plot(runs,accuracy_list, linewidth = 3, color='b',label='Average')
            plt.plot(runs,stdv_list, linewidth = 3, color='orange',label='Standard Deviation')
            plt.ylabel('Resource Usage State Characterization Accuracy (%)')
            plt.xlabel('Number of Benchmark Runs')
            plt.axvline(3,color='green', linewidth = 3, label = 'Acceptable Accuracy')
            plt.axvline(5,color='red', linewidth = 3, label = 'RCoinP Minimum')
            plt.legend(loc='best')
            plt.ylim([0,100])
            plt.xticks(runs)
            plt.yticks(np.arange(0,110,10))
            plt.grid()
            plt.show()

        else:
            
            plt.bar(runs,accuracy_list, width = 0.3, yerr = stdv_list, color=['blue', 'blue', 'green', 'blue', 'red','blue', 'blue','blue', 'blue', 'blue'], error_kw=dict(lw=2, capsize=2, capthick=2))
            plt.ylabel('Resource Usage State Characterization Accuracy (%)')
            plt.xlabel('Number of Benchmark Runs')
            plt.show()
        
if __name__ == '__main__':

    # features to obtain difference between to consequtive measurements (measurements accumulate)
    diff_columns = ['cpu_user_time', 'cpu_system_time','cpu_idle_time', 'net_sent', 'net_recv', 'io_counters_read_bytes_ipykernel', 'io_counters_write_bytes_ipykernel', 'cpu_times_user_ipykernel', 'cpu_times_system_ipykernel']
    
    # features to be used in resource state clustering
    feature_columns = ['cpu_user_time_diff', 'cpu_system_time_diff','cpu_idle_time_diff','memory','memory_percent_ipykernel','cpu_times_user_ipykernel_diff', 'cpu_times_system_ipykernel_diff','net_sent_diff','QOS']
    
    # available devices;  RPi8_1500: High Device, RPi4_1000: Mid Device, RPi2_600: Low Device
    rpi_names = ['RPi8_1500','RPi4_1000','RPi2_600']
    
    # resource state used for calculating RCoinP value, order in relation to devices same as rpi_names
    # possible values are: 'Stationary_WIFI','Stationary_Streaming', 'Stationary_Games_WIFI'
    RCoin_list = ["Stationary_WIFI","Stationary_WIFI","Stationary_WIFI"]
    
    # resource state used used for accuracy test and task evaluation test
    # an additional state not used in published resutls is 'Stationary_Streaming_WIFI'
    data_names = ['Stationary', 'Stationary_WIFI','Stationary_Streaming', 'Stationary_Games_WIFI']

    avg_acc_list, std_acc_list = [], []
    
    runs = range(1,11)
    
    for r in runs:
        # creates cluster class object with features, devices, and data specfied in the lists above
        # centroid type defines what clustering technique is used, either K-Means or the median (used in paper)
        # wait interval specifies experiment period, the experiment that specifies the resource state of the test data
        # if wait interval is 0, only the final results are obtained, without showing the state of each datapoint
        c = cluster(rpi_names, data_names, diff_columns, feature_columns, centorid_type='median', wait_interval= 0) # centorid types are 'median' and 'k-means'

        # shows accuracy of resource state clustering test, if verbose is True, shows the count of all assigned labels per resource state 
        data, avg_acc, std_acc = c.run_clustering(runs = r, verbose = True)

        avg_acc_list.append(avg_acc)
        std_acc_list.append(std_acc)
    
    c.plot_acc_per_runs(avg_acc_list, std_acc_list, runs, 'line') # possible plot types 'line' or 'bar' plot


    
    # runs task allocation evaluation for the specified initial state when calculating RCoinP for each device
    
    #c.task_alloc_eval(RCoin_list, verbose = False)

