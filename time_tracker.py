# this file holds class to get usage statistics


from time import time, sleep, localtime
from queue import LifoQueue
from threading  import Thread
import pandas as pd
from statistics import mean
pd.options.mode.chained_assignment = None  # default='warn'

class time_tracker():

    def __init__(self, interval = 2):
    
        self.interval = interval
        self.q = LifoQueue()
        self.kill_monitor_thread = False # unsets flag that kills monitoring thread
        self.get_current = False # unsets flag to place usage info in queue from thread
        self.model_num = None 
        self.QOS = 0
        self.QOS_list = []
        self.task = 'None'


    def get_time(self):
        """Obtains one instance of usage information using the PsUtil Library at current time"""
        curr_time = time() # gets curretn time
        timeObj = localtime(curr_time)
        # creats time stamp Day-Month-Year Hour:Minute:Second
        time_stamp = str(timeObj.tm_mday) + '-' + str(timeObj.tm_mon) + '-' + str(timeObj.tm_year) + ' ' + str(timeObj.tm_hour) + ':' + str(timeObj.tm_min) + ':' + str(timeObj.tm_sec)

        return time_stamp, curr_time

  

    def get_time_df(self, model_num, QOS, QOS_list, task):
        """Gets time infomation and converts them into a pandas Dataframe"""
        time_stamp, current_time= self.get_time()

        # creats dictionary with time information to be added to the dataframe
        time_dict = {'time_stamp': [time_stamp], 'time': [current_time], 'model_number': [model_num], 'QOS': [QOS], 'QOS_avg': [mean(QOS_list) if len(QOS_list) >= 1 else 0], 'task': [task]}
        self.QOS_list = []

        return pd.DataFrame(time_dict).set_index('time_stamp') # returns dataframe with time stamp as index
  

    def run_monitor(self, interval=2):
        """Gets time information according to the interval set, to be used in a seprate thread so as not to stop code flow, returns time information as a Dataframe when stop_monitor_thread is set"""
        get_loop_time = 0
        time_dataframe = self.get_time_df(self.model_num, self.QOS, self.QOS_list, self.task) # initialize usage information dataframe
        while True:
            if self.kill_monitor_thread == True: # kill thread when stop thread flag is set externally
                self.q.put(time_dataframe) # place dataframe in queue enabling access from outside of thread
                break
            elif self.get_current == True: # place latest usage info data in queue
                self.q.put(time_dataframe) # place dataframe in queue enabling access from outside of thread
            sleep(interval-get_loop_time) # used to ensure interval timing is accurate, taking out time to obtain info
            start_loop_time = time()
            time_dataframe = time_dataframe.append(self.get_time_df(self.model_num, self.QOS, self.QOS_list, self.task)) # update usage information dataframe
            get_loop_time = time() - start_loop_time

    def run_monitor_thread(self):
        """ Runs monitoring thread """
        self.kill_monitor_thread = False
        self.t1 = Thread(target=self.run_monitor, args=[self.interval]) # creates monitoring thread and passes queue to return results in, with specified info retrieval interval
        self.t1.daemon = True
        self.t1.start()

    def get_current_usage(self):
        """ returns dataframe with current worker usage information """
        self.get_current = True
        while len(self.q.queue) == 0: # wait until queue not empty 
            pass
        time_data = self.q.get()
        self.get_current = False
        return time_data

    def stop_monitor_thread(self):    
        """ stops monitoring thread and returns last worker usage information dataframe """
        self.kill_monitor_thread = True

        while len(self.q.queue) == 0: # wait until queue not empty 
            pass
        
        time_data = self.q.get()

        self.t1.join()  # kills monitoring thread
        return time_data # returns last usage information dataframe stored in queue

