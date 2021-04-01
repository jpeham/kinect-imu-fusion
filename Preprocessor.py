#  A program to prepare data from kinect and imu for classificator
import pandas as pd
import Session
import datetime

bool_loading = False # for showing loading process
debug_prep = False
debug = False

class Preprocessor:
    def __init__(self, path_list, data_list):
        if bool_loading:
            print('preprocessor started', end='')
        self.sessions = []

        if bool_loading:
            print('creating sessions ', end='')
        # get Sessions from path_list
        for i in range(len(path_list)):
            # go through path_list, create Session for every path and append it to sessions
            
            #if bool_loading:
                #print('session {} '.format(i), end='')
            if True:
                print('.', end='')
            
            session = Session.Session(path_list[i]) # get data from samples
            self.sessions.append(session)
            
        if bool_loading:
            #print(' created')
            print('^')
        
        #assert ob alle sessions gleich viele slots haben --------------------------NOTWENDIG?!??
        if False:
            self.prev_len = 0
            for i in range(len(self.sessions)): # für jede Session
                #print('#'*30)
                #print('session {}: targets shape: {}'.format(i, self.sessions[i].targets.shape[1]))
                #print('session {}: slots length: {}'.format(i, len(self.sessions[i].slots)))

                if not (self.prev_len == 0):
                    #print('check if prev_len {} == slots length {}'.format(self.prev_len, len(self.sessions[i].slots)))
                    assert self.prev_len == len(self.sessions[i].slots), \
                        'session ' + self.sessions[i].dir_path + ' hat unterschiedlich viele slots! ' \
                        + str(len(self.sessions[i].slots)) + ' statt bisher ' + str(self.prev_len)

                #else:
                    #print('dont check since prev_len is still 0: {}'.format(self.prev_len))
                self.prev_len = len(self.sessions[i].slots)
                #print('prev_len = {}'.format(self.prev_len))

                
                
                
                
        # alle slots von auf eine Länge bekommen
        # = samples von allen slots auffüllen, bis alle gleich lang
        # --> auslagern in eigene Methode?
        
        # find length of longest segment
        self.longest = 0
        for d in range(len(self.sessions)): # für jede Session
            for s in range(len(self.sessions[d].slots)): # für jeden Slot
                self.longest = max(self.longest, len(self.sessions[d].slots[s].kininert)) # die längste Länge speichern

        #print('longest segment = {}'.format(self.longest)) # Anzahl der längsten Sequenz-sampling-anzahl
        # --> alle anderen Samplings müssen so lange sein
        
        #find length of shortest segment
        self.shortest = 99990
        for d in range(len(self.sessions)):
            for s in range(len(self.sessions[d].slots)):
                self.shortest = min(self.shortest, len(self.sessions[d].slots[s].kininert))
                #print('{} - {} - {}'.format(d, s, shortest))
        #print('-')
        #print('shortest segment = {}'.format(self.shortest))
        
        
        
        #self.sessions[0].slots[0].kininert.iloc[-1:]
        
        
        # bei jedem Sampling wird das letzte Sample 'eingefrohen'
        # aber timestamp um 20ms und counter um 1 hochzählen
        if bool_loading:
                    print('bringing slots to same length ', end='') # Segmente auffüllen
        print('|', end='')
        for d in range(len(self.sessions)): # für jede Session
            if True:
                print('.', end='')
            if bool_loading:
                print('|', end='') # für jede Session
                
            for s in range(len(self.sessions[d].slots)): # für jeden Slot
                if bool_loading:
                    print('.', end='') # für jeden Slot
                    
                # differenz herausfinden zwischen pos und self.longest
                length = len(self.sessions[d].slots[s].kininert)
                diff = self.longest - length
                #print('sampling length = {}'.format(diff))
                #print('diff between sampling and diff = {}'.format(length))

                last_sample = self.sessions[d].slots[s].kininert.iloc[-1:] # letztes Sample im Sampling
                ##print('###last sample = {}'.format(last_sample))
                #print('###last sample index = {}'.format(last_sample.index))
                #print('###last sample counter = {}'.format(last_sample.iloc[0][0]))

                # genauso oft letzte Zeile in kininert appenden
                for i in range(diff):
                    #print('trying')
                    # increment counter by one
                    last_sample.iloc[0][0] += 1
                    # and
                    # add 20 ms to timestamp
                    last_sample.index += datetime.timedelta(milliseconds=20) # adds 20 ms

                    #print('~~~~last sample index = {}'.format(last_sample.index))
                    #print('~~~~last sample counter = {}'.format(last_sample.iloc[0][0]))

                    self.sessions[d].slots[s].kininert = self.sessions[d].slots[s].kininert.append(last_sample)
                    #print('---' + str(d) + '-' + str(s) + '-' + str(i) 
                    #      + ' appended - ' + str(len(self.sessions[d].slots[s].kininert)))
                    #print('--')
        if bool_loading:
                print(' - done') # für jeden Slot
        
        
        
        # check if length of shortest segment is now length of longest segment
        self.shortest = 99990
        for d in range(len(self.sessions)):
            for s in range(len(self.sessions[d].slots)):
                self.shortest = min(self.shortest, len(self.sessions[d].slots[s].kininert))
                #print('{} - {} - {}'.format(d, s, shortest))
        #print('-')
        #print('shortest segment = {}'.format(self.shortest))
        
        #--------------------------------------------------------------------------- assert if not
        assert self.shortest == self.longest, 'kininerts aller Segmente sind NICHT GLEICH LANG!!'
        
        #####################################################
        #bis hier hin "alle slots von auf eine Länge bekommen" # -------> auslagern in eigene Methode? TODO
        
        # counter_kin und counter_inert rausschmeißen 
        # -------------------TODO: schon früher raus schmeißen? vor Übergabe in Slots?
        for d in range(len(self.sessions)): # für jede Session
            for s in range(len(self.sessions[d].slots)): # für jeden Slot
                self.sessions[d].slots[s].kininert.pop('counter_kin')
                self.sessions[d].slots[s].kininert.pop('counter_inert')        
        
        
        #if dist nicht in list -> raus schmeißen
        if 'dist' not in data_list:
            #print('dist not in data_list -> rausschmeißen')
            # dist und gyro raus schmeißen
            for d in range(len(self.sessions)): # für jede Session
                for s in range(len(self.sessions[d].slots)): # für jeden Slot
                    self.sessions[d].slots[s].kininert.pop('distance_tire1')
                    self.sessions[d].slots[s].kininert.pop('distance_tire2') 
                    self.sessions[d].slots[s].kininert.pop('distance_field1')
                    self.sessions[d].slots[s].kininert.pop('distance_field2')

                    #print(self.sessions[d].slots[s].kininert.head(3))
        # -------------------
        
        #if pos nicht in list -> raus schmeißen
        if 'pos' not in data_list:
            #print('pos not in data_list -> rausschmeißen')
            # pos raus schmeißen
            for d in range(len(self.sessions)): # für jede Session
                for s in range(len(self.sessions[d].slots)): # für jeden Slot
                    self.sessions[d].slots[s].kininert.pop('pos_x')
                    self.sessions[d].slots[s].kininert.pop('pos_y') 
                    self.sessions[d].slots[s].kininert.pop('pos_z') 

                    #print(self.sessions[d].slots[s].kininert.head(3))
        # -------------------
        
        #if acc nicht in list -> raus schmeißen
        if 'acc' not in data_list:
            #print('acc not in data_list -> rausschmeißen')
            # acc raus schmeißen
            for d in range(len(self.sessions)): # für jede Session
                for s in range(len(self.sessions[d].slots)): # für jeden Slot
                    self.sessions[d].slots[s].kininert.pop('acc_x')
                    self.sessions[d].slots[s].kininert.pop('acc_y') 
                    self.sessions[d].slots[s].kininert.pop('acc_z')

                    #print(self.sessions[d].slots[s].kininert.head(3))
        # -------------------
        
        #if gyro nicht in list -> raus schmeißen
        if 'gyro' not in data_list:
            #print('gyro not in data_list -> rausschmeißen')
            # gyro raus schmeißen
            for d in range(len(self.sessions)): # für jede Session
                for s in range(len(self.sessions[d].slots)): # für jeden Slot
                    self.sessions[d].slots[s].kininert.pop('gyro_x')
                    self.sessions[d].slots[s].kininert.pop('gyro_y')
                    self.sessions[d].slots[s].kininert.pop('gyro_z')

                    #print(self.sessions[d].slots[s].kininert.head(3))
        
        if bool_loading:
                print('preparation done')
        #print('0-0-kininert head5:')
        #print(self.sessions[0].slots[0].kininert.head(5))
        
        
        # make sessions_list with samples of all sesstions
        # --------------------------------------------------TODO make list of one session first, and later concat?
        self.sessions_list = []
        #print('sessions_list-----------')
        #self.sessions_list

        # jedes segment = slot zu array machen (.values) und flatten
        # und dann an session_list anhängen

        for d in range(len(self.sessions)): # für jede Session
            for s in range(len(self.sessions[d].slots)): # für jeden Slot
                seg = self.sessions[d].slots[s].kininert # kininert dataframe für aktuellen Slot
                seg = seg.values # numpy array aus kininert dataframe machen
                seg = seg.flatten() # numpy array zu liste flatten
                self.sessions_list.append(seg) # liste als Zeile an session_list anfügen
                
                if debug:
                    print(' {} - {} kininert.values.flattened appended to sessions_list, sessions_list is not {} long'.format(d, s, len(self.sessions_list)))

                    
        # TODO assert, der schaut ob Länge von session_list übereinstimmt mit logischer Zahl
        # Länge sollte ja Anzahl an Sessions x Anzahl an Segmenten sein
        # - Anzahl an Segmente sollte bei jeder Session gleich groß sein!
                
        if False:
            if debug:
                        print('sessions_list created, length should be: {} * {} = {}, is {}'.format(len(self.sessions), self.prev_len, len(self.sessions)*self.prev_len, len(self.sessions_list))) 
            # TODO assert daraus machen:

            assert len(self.sessions)*self.prev_len == len(self.sessions_list), 'sessions_list length should be number of sessions times number of slots, but isnt!'
        
        
        # TODO assert, der schaut ob zeile.shape ((5512, )) übereinstimmt mit allen
        
        # TODO generell assert, der schaut ob Größe stimmt


        self.targets_list = []
        # targets_list machen wie sessions_list
        # TODO sessions_list umbenennen?
        
        if True:
            for d in range(len(self.sessions)): # für jede Session
                #print(self.sessions[d].targets)
                #print('do it {} times'.format(int(self.sessions[d].targets.shape[1])))
                for t in range(int(self.sessions[d].targets.shape[1])):
                    tar = self.sessions[d].targets.iloc[0,t]
                    #print(tar)
                    self.targets_list.append(tar)
                #print('---')
                #print(self.targets_list)



            #print(self.sessions[d].targets)
            #print('---')

            #tar = self.sessions[d].targets # targets von aktueller session
            ######## nicht notwendig, da schon nur 1 lang... tar = tar.values # numpy array aus tar 
            ##########tar = tar.flatten() # numpy array zu liste flatten

            ####self.targets_list.append(tar) # liste als Zeile an session_list anfügen
            #print('targets_list:')
            #print(self.targets_list)

        if False:
            self.all_targets = self.sessions[-1].targets  # get row with all targety (1-21) to initialize dataframe (prbly lazy)
            for i in range(len(self.sessions)):
                # type(self.sessions[i].targets.tail(1))

                session_tmp = self.sessions[i]  # get current session
                df_tmp = session_tmp.targets  # get current targets
                df_tmp['name'] = session_tmp.dir_path[7:]  # append sample name to targets values
                all_targets = all_targets.append(df_tmp)  # appending current targets to list with all targets

            all_targets = all_targets[
                ['name', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '12', '13', '14', '15', '16', '17',
                 '18', '19', '20', '21']] # --------------------------------------------------TODO geht nur bei 21 Segmenten nicht bei 19!
            all_targets = all_targets[1:]  # drop first row (that we only needed for initializind dataframe)
            # all_targets#.head(5)
        
    
    def make_session_list(self):
        asdf = 0
        
        # -----------------------------------------------------------------------TODO muss noch gemacht werden
        
        #####    # jedes segment = slot zu array machen (.values) und flatten
        #####    # und dann an session_list anhängen
        #####    d = 0
        #####    for s in range(len(self.sessions[d].slots)): # für jeden Slot
        #####        seg = self.sessions[d].slots[s].kininert # kininert dataframe für aktuellen Slot
        #####        seg = seg.values # numpy array aus kininert dataframe machen
        #####        seg = seg.flatten() # numpy array zu liste flatten
        #####        self.sessions_list.append(seg) # liste als Zeile an session_list anfügen
    
    def plot(self):
        # ---------------------------------------------------------------------------------------------TODO
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # #import datetime
        #
        # sns.set() # for nicer looking sns plots (gray backgound, lines,...)
        # #pd.options.display.float_format = '{:.2f}'.format # to show timestamp as float and not as e-notation
        # %config InlineBackend.figure_format = 'retina' #for higher plot resolution
        
        for i in range(len(self.session1.slots)):
            self.session1.plot_slot(i)

        for i in range(len(self.session2.slots)):
            self.session2.plot_slot(i)
        
        
        
