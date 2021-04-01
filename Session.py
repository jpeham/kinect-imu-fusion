# program to create a session from path for CSVs

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime #J: for unix timestamps
import os # for creating directories

#sns.set() # for nicer looking sns plots (gray backgound, lines,...)
pd.options.display.float_format = '{:.2f}'.format # to show timestamp as float and not e-notation
#%config InlineBackend.figure_format = 'retina' #for higher plot resolution


import tools
import Slot
debug = False
debug_prep = False
debug_concat = False

bool_loading = False # for showing loading process

class Session: 
    def __init__(self, dir_path): 
        self.dir_path = dir_path
        if debug:
            print('start preparing data')
        
        #Distances to tires
        #TODO auslagern in tools
        self.dist = pd.read_csv(("../Both/samples/z-final/" + dir_path + "/kin-sample-distances_all" + ".csv"))
        
        #self.distance = self.distance[self.distance.tire2 != "nan"] # get nans out
        #self.distance.dropna(how='any')    #to drop if any value in the row has a nan
        
        self.dist['timestamp1'] /= 1000 # to get milliseconds behind decimal point
        self.dist['timestamp2'] /= 1000 # to get milliseconds behind decimal point
        self.dist['timestamp_field1'] /= 1000 # to get milliseconds behind decimal point
        self.dist['timestamp_field2'] /= 1000 # to get milliseconds behind decimal point
        #self.distance.pop('timestamp2')
        #self.distance.pop('tire1')
        #self.distance.pop('tire2')
        
        if debug_concat:
            print('-'*30)
            print('dist from csv')
            print(self.dist.iloc[-5:])
            print('-'*40)
        
        #--------------------------------------------------------------------------------------------------
        
        # Targets
        
        full_path = ("../Both/samples/z-final/" + dir_path + "/targets.csv")
        self.targets = pd.read_csv(full_path)
        self.targets = self.targets.drop([self.targets.columns[0]], axis='columns')
        
        if debug_prep:
            print('targets shape: {}'.format(self.targets.shape))
        
        
        #Kinect
        self.kin = tools.get_data(dir_path, "kin-sample-hand") # get kinect dataframe from csv
        
        # pos-Daten normalisieren, damit sie sich um 0 bewegen
        self.kin['pos_x'] = self.kin['pos_x'] - self.kin['pos_x'].mean() # TODO kürzer (-=)
        self.kin['pos_y'] = self.kin['pos_y'] - self.kin['pos_y'].mean()
        self.kin['pos_z'] = self.kin['pos_z'] - self.kin['pos_z'].mean()

        # Daten = kininert generell normalisieren?
        # -> damit sie sich zwischen -1 und 1 bewegen?

        if debug_concat:
            print('-'*30)
            print('kin from csv')
            print(self.kin.iloc[-5:])
            print('-'*40)
        
        if False: # Sicherung
            # pos-Daten normalisieren, damit sie sich um 0 bewegen
            self.temp_pos = self.kin
            self.temp_pos['pos_x'] = self.kin['pos_x'] - self.kin['pos_x'].mean()
            self.temp_pos['pos_y'] = self.kin['pos_y'] - self.kin['pos_y'].mean()
            self.temp_pos['pos_z'] = self.kin['pos_z'] - self.kin['pos_z'].mean()

            self.pos = pd.DataFrame({'counter':self.temp_pos['counter'],
                             'timestamp':self.temp_pos['timestamp'],
                             'pos_x':self.temp_pos['pos_x'],
                             'pos_y':self.temp_pos['pos_y'],
                             'pos_z':self.temp_pos['pos_z']})
            self.pos123 = self.temp_pos[['counter','timestamp', 'pos_x', 'pos_y', 'pos_z']] # TODO kann weg?

            self.kin = self.temp_pos
        
        
        #--------------------------------------------------------------------------------------------------
        #                              CONCAT KIN AND INERT
        #--------------------------------------------------------------------------------------------------
        assert len(self.kin) == len(self.dist), 'positions-samples und distances-samples sind NICHT GLEICH LANG!!'
        # positions-samples und distances-samples sind gleich lang und dürfen zusammengefügt werden
        if debug_concat:
            print('positions-samples und distances-samples sind gleich lang und dürfen zusammengefügt werden')
        
        #gleiche indices!!
        self.kin = pd.concat([self.kin, self.dist], axis=1, sort=False)
        
        if debug_concat:
            print('-'*30)
            print('kin concat with dist')
            print(self.kin.iloc[-5:])
            print('-'*40)
        
        self.kin = self.kin[['counter','timestamp', 
                             'pos_x', 'pos_y', 'pos_z', 
                             'distance1', 'distance2', 'distance_field1', 'distance_field2']]
        self.kin = self.kin.rename(index=str, columns={'distance1': 'distance_tire1', 'distance2': 'distance_tire2'})
        
        if debug_concat:
            print('-'*30)
            print('kin concat with dist, aussortiert, umbenannt')
            print(self.kin.iloc[-5:])
            print('-'*40)
        
        ####doppelte counter Spalten dropen
        ###assert self.kin.counter.all() == self.kin.counter_dist.all(), 'counter of kinect data and distance are not the same, something went wrong!'
        ###if debug_concat:
        ###    print('counter columns are the same -> drop one')
        ###self.kin = self.kin.drop(columns=['counter_dist'])
            
            
            
            
        if False: # Sicherung
            assert len(self.pos ) == len(self.dist), 'positions-samples und distances-samples sind NICHT GLEICH LANG!!'
        
            # positions-samples und distances-samples sind gleich lang und dürfen zusammengefügt werden
            if debug_concat:
                print('positions-samples und distances-samples sind gleich lang und dürfen zusammengefügt werden')
            self.kin = pd.DataFrame({'counter':self.pos['counter'],
                                 'timestamp':self.pos['timestamp'],
                                 'pos_x':self.pos['pos_x'],
                                 'pos_y':self.pos['pos_y'],
                                 'pos_z':self.pos['pos_z'],
                                 'counter_dist':self.dist['Unnamed: 0'],
                                 'distance_tire1':self.dist['distance1'],
                                 'distance_tire2':self.dist['distance2'],
                                 'distance_field1':self.dist['distance_field1'],
                                 'distance_field2':self.dist['distance_field2']})

            #doppelte counter Spalten dropen
            assert self.kin.counter.all() == self.kin.counter_dist.all(), 'counter of kinect data and distance are not the same, something went wrong!'
            if debug_concat:
                print('counter columns are the same -> drop one')
            self.kin = self.kin.drop(columns=['counter_dist'])
        
        #--------------------------------------------------------------------------------------------------
        
        #Inertial
        self.inert = tools.get_data(dir_path, "inert-sample") # get inertial dataframe from csv
        self.inert = pd.DataFrame({'counter':self.inert['counter'],
                                 'timestamp':self.inert['timestamp'],
                                 'acc_x':self.inert['acc_x'],
                                 'acc_y':self.inert['acc_y'],
                                 'acc_z':self.inert['acc_z'],
                                 'gyro_x':self.inert['gyro_x'],
                                 'gyro_y':self.inert['gyro_y'],
                                 'gyro_z':self.inert['gyro_z']})
        
        #--------------------------------------------------------------------------------------------------
        
        #Events
        self.events = tools.get_data(dir_path, "kin-sample-events")
        #filter out nonsense event_types (16, 17, 4,... )
        # --> get events to minimum event_types --> graph is minimal height (readability)
        #--------------------------------------------------------------------------------------------------
        
        #Timeslots from Event-Keys
        self.prepare_timeslots(dir_path)
        
        #--------------------------------------------------------------------------------------------------
        
        self.inert_old = self.inert # zur Sicherung
        self.inert['timestamp'] = pd.to_datetime(self.inert['timestamp'], unit='s') # unix timestamp to datetime
        
        debug_uniquify = False
        if debug_uniquify:
            print('starting uniquify with inert data')
            print(''*50)
        self.inert = tools.uniquify_timestamp(self.inert)
        
        self.inert = self.inert.set_index('timestamp')
        self.inert = self.inert.resample('20ms').pad()
        #inert_pad.iloc[-5:]
        
        #---
        
        self_kin_old = self.kin # Zur Sicherung
        
        self.kin['timestamp'] = pd.to_datetime(self.kin['timestamp'], unit='s') # unix timestamp to datetime
        #kin = kin.set_index('timestamp')
        
        if debug_uniquify:
            print('starting uniquify with kin data')
            print(''*50)
        self.kin = tools.uniquify_timestamp(self.kin)
        self.kin = self.kin.set_index('timestamp')
        self.kin = self.kin.resample('20ms').pad()
        #kin_pad.iloc[-5:]
        
        #######################################################
        
        self.kin = self.kin.reset_index()
        self.inert = self.inert.reset_index()

        session_beginning = max(self.kin.iloc[0,0], self.inert.iloc[0,0])
        session_end = min(self.kin.iloc[-1,0], self.inert.iloc[-1,0])

        if debug_prep:
            print('session {} from {} to {}'.format(dir_path, session_beginning, session_end))

        # kin und inert so schneiden, dass sie sich überdecken und nie alleine laufen
        self.kin_short = self.kin.loc[(self.kin['timestamp'] >= session_beginning) & (self.kin['timestamp'] <= session_end)]
        self.inert_short = self.inert.loc[(self.inert['timestamp'] >= session_beginning) & (self.inert['timestamp'] <= session_end)]

        self.kin_test = self.kin_short
        self.inert_test = self.inert_short
        
        assert self.kin_short.iloc[0,0] == self.inert_short.iloc[0,0], 'erster timestamp nicht gleich!'
        assert self.kin_short.iloc[0,0] == self.inert_short.iloc[0,0], 'letzter timestamp nicht gleich!'
        assert len(self.kin_short) == len(self.inert_short), 'dfs nicht gleich lang!!'
        # kin und inert haben die gleiche Sampling-Rate und die selben ersten und letzten Timestamps
        # --> dürfen gemerged werden

        if debug_concat:
            print('-'*100)
            print('-'*100)
            print('-'*30)
            print('self.kin_short before set_index and column_rename')
            print(self.kin_short[-5:])
            print('-'*40)
            print('self.kin_short before set_index, timestamp type = ' + str(type(self.kin_short.iloc[0,0])))
        
        self.kin_short = self.kin_short.set_index('timestamp')
        self.kin_short = pd.DataFrame({'counter_kin':self.kin_short['counter'], 
                                       'pos_x':self.kin_short['pos_x'], 
                                       'pos_y':self.kin_short['pos_y'], 
                                       'pos_z':self.kin_short['pos_z'], 
                                       'distance_tire1':self.kin_short['distance_tire1'], 
                                       'distance_tire2':self.kin_short['distance_tire2'], 
                                       'distance_field1':self.kin_short['distance_field1'], 
                                       'distance_field2':self.kin_short['distance_field2']})
        
        if debug_concat:
            print('-'*30)
            print('self.kin_short')
            print(self.kin_short[-5:])
            print('-'*40)
            print('self.kin_short after set_index, index type'+ str(self.kin_short.index.dtype))
        
        #########
        
        if debug_concat:
            print('-'*30)
            print('self.inert_short before set_index and column_rename')
            #print(self.inert_short[-5:])
            print('-'*40)
            print('self.inert_short before set_index, ' 
                  + 'timestamp = ' + str(self.inert_short.iloc[0,0]) 
                  + ', timestamp type = ' + str(type(self.inert_short.iloc[0,0])))
        
        self.inert_short = self.inert_short.set_index('timestamp')
        self.inert_short = pd.DataFrame({'counter_inert':self.inert_short['counter'], 
                                         'acc_x':self.inert_short['acc_x'], 
                                         'acc_y':self.inert_short['acc_y'], 
                                         'acc_z':self.inert_short['acc_z'],
                                         'gyro_x':self.inert_short['gyro_x'],
                                         'gyro_y':self.inert_short['gyro_y'],
                                         'gyro_z':self.inert_short['gyro_z']})
        
        if debug_concat:
            print('-'*30)
            print('self.inert_short')
            #print(self.inert_short[-5:])
            print('-'*40)
            print('self.inert_short after set_index, index type'+ str(self.inert_short.index.dtype))

        # ----------------------------concat kin and inert to kininert
        
        self.kininert = pd.concat([self.kin_short, self.inert_short], axis=1, sort=False)

        if debug_concat:
            print('-'*30)
            print('self.kininert')
            print(self.kininert[-5:])
            print('-'*40)
        
        if debug_concat:
            print('kin und inert concateniert')
        
        ####################################################################################
        
        #make slot object for every slot in slots
        self.slots = []
        if False:
            print('start making slots')
            print(self.slots_short)
        
        for i in range(len(self.slots_short)):
            #if i == 0: # only for DEBUG
            slot = self.slots_short.iloc[i]
            #unicode = slot.unicode
            tsf = slot.timestamp_from
            tst = slot.timestamp_to
            s = Slot.Slot(i, tsf, tst, self.kininert) # create new slot
            self.slots.append(s) # save slot in slots-list
            
            if bool_loading:
                print('.', end='')
       
        if debug:
            print('preparing data done')
            
        if debug_prep:
            print('targets shape: {}'.format(self.targets.shape[1]))
            print('slots length: {}'.format(len(self.slots)))
        
        assert self.targets.shape[1] == len(self.slots), ' Anzahl Slots = ' + str(len(self.slots))\
                                                         + ' und Anzahl Targets = ' + str(self.targets.shape[1]) \
                                                         + ' von session ' + str(self.dir_path) + ' sind NICHT GLEICH LANG!'
        
        
    def prepare_timeslots(self, dir_path): # get keys and get timeslot-segments out
        debug_slots = False

        self.keys = pd.read_csv(("../Both/samples/z-final/" + dir_path + "/kin-sample-events-keys" + ".csv"))
        #print("../Both/samples/" + dir_path + "/kin-sample-events-keys" + ".csv")
        #self.keys = pd.read_csv(("../Both/samples/sample-20190220-183508/kin-sample-events-keys" + ".csv"))

        self.keys['timestamp'] /= 1000 # to get milliseconds behind decimal point
        self.keys['timestamp'] = pd.to_datetime(self.keys['timestamp'], unit='s') # unix timestamp to datetime
        
        if debug_slots:
            print('start preparing timeslots without markers')
            print('original keys')
            print(self.keys)
            print('.')
        
        try:
            self.keys = self.keys[self.keys.scancode != 30] # get a out
        except:
            print('no a')
        
        # TODO delete everything after z?
        
        try:
            self.keys = self.keys[self.keys.scancode != 21] # get z out
        except: print('no z')
        
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
                   'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # take unneccesary button-ups from number-and-letter input and delete it
        # we only need number and letter buttom-downs and buttom-down and buttom-up from spaces
        for i in range(len(self.keys)):
            if self.keys.iloc[i]["unicode"] == "b": #################################
                self.keys.iat[i,2] = "beginning"
                if self.keys.iloc[i+1]["scancode"] == self.keys.iloc[i]["scancode"] and \
                   self.keys.iloc[i+1]["unicode"] == "none": # i+1 is button-up for i
                    self.keys.iat[i+1,2] = "end"
            elif self.keys.iloc[i]["unicode"] in numbers or \
               self.keys.iloc[i]["unicode"] in letters: #############################
                if self.keys.iloc[i+1]["scancode"] == self.keys.iloc[i]["scancode"] and \
                   self.keys.iloc[i+1]["unicode"] == "none": # i+1 is button-up for i
                    self.keys.iat[i+1,2] = "delete"

        self.keys = self.keys[self.keys.unicode != 'delete'] # delete all rows marked with delete
        self.keys = self.keys[~self.keys['unicode'].isin(numbers)] # delete all rows with numbers in unicode == POIs
        
        if debug_slots:
            print('preparing timeslots, get beginning and end, delete up for keys, delete digit_rows')
            print(self.keys)
            print('.')
        
        ###print(self.keys)
            
            
        # TODO!
        # test dass rows am Ende Format: letter - beginning -  end hat??
        # und dass scancode von 2&3 der selbe ist.
        self.slots_long = pd.DataFrame({'event_from':self.keys['event_type'].iloc[0::2].values,
                         'event_to':self.keys['event_type'].iloc[1::2].values,
                         'unicode_from':self.keys['unicode'].iloc[0::2].values,
                         'unicode_to':self.keys['unicode'].iloc[1::2].values,
                         'scancode_from':self.keys['scancode'].iloc[0::2].values,
                         'scancode_to':self.keys['scancode'].iloc[1::2].values,
                         'counter_from':self.keys['counter'].iloc[0::2].values,
                         'counter_to':self.keys['counter'].iloc[1::2].values,
                         'timestamp_from':self.keys['timestamp'].iloc[0::2].values,
                         'timestamp_to':self.keys['timestamp'].iloc[1::2].values})

        self.slots_short = pd.DataFrame({'counter_from':self.keys['counter'].iloc[0::2].values,
                         'counter_to':self.keys['counter'].iloc[1::2].values,
                         'timestamp_from':self.keys['timestamp'].iloc[0::2].values,
                         'timestamp_to':self.keys['timestamp'].iloc[1::2].values})

        #return TODO
    
    def plot_slot(self, i):
        # build plots
        fig, ax =plt.subplots(4,1)
        fig.subplots_adjust(hspace=0.6)

        dist_ax = sns.lineplot(x=self.slots[i].dist_melt.index, y="value", hue="variable", legend = "full", data=self.slots[i].dist_melt, ax=ax[0]).set_title("distance " + str(self.slots[i].counter))
        pos_ax = sns.lineplot(x=self.slots[i].pos_melt.index, y="value", hue="variable", legend = "full", data=self.slots[i].pos_melt, ax=ax[1]).set_title("position " + str(self.slots[i].counter))
        acc_ax = sns.lineplot(x=self.slots[i].acc_melt.index, y="value", hue="variable", legend = "full", data=self.slots[i].acc_melt, ax=ax[2]).set_title("accelerometer " + str(self.slots[i].counter))
        gyro_ax = sns.lineplot(x=self.slots[i].gyro_melt.index, y="value", hue="variable", legend = "full", data=self.slots[i].gyro_melt, ax=ax[3]).set_title("gyroscop " + str(self.slots[i].counter))


        custom_dir = "../Both/samples/z-final/" + self.dir_path + "/plots/plot_" + str(i) + ".png"

        ###custom_dir = "plots-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # set name for directory
        ###os.makedirs(custom_dir) # create directory
        pos_ax.figure.set_size_inches(18.5, 10.5)
        fig.savefig(custom_dir) ###  + "/" + self.dir_path + "/plot_" + str(i) + ".png")
        #open("%s/kin-sample-events.csv" % custom_dir, "w")








