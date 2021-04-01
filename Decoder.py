import pandas as pd
import tools
from pandas import DataFrame

import datetime

debug = False

# dont have Objects with Methods
# but Py Skript with Funktions to call later

#

class Decoder:
    slots_short: DataFrame # TODO --------------------------------- typeHint

    def __init__(self):

        self.path_list = []
        self.sensors_list = []
        self.sessions = [] # TODO wie und wo? nachdem Methoden aufgerufen, direkt in self.sessions speichern, oder jedesmal nur als RETURN??
        # self.kininert

        #print('hello, im alive')
        # TODO do me
        import datetime
        self.start_time = datetime.datetime.now()
        #print('Start time = {}'.format(self.start_time))

    def extract_paths(self):
        # get list of folder_names from z-final dir
        from os import listdir
        from os.path import isfile, join
        folder_names = [f for f in listdir("../../Fusion/Recorder/recordings/z-final") if
                        not isfile(join("../../Fusion/Recorder/recordings/z-final", f))]
        path_list = folder_names

        return path_list

    def fill(self, path_list = None, sensors_list = None):

        # if no paths are given, take all paths from z-final
        if path_list is None:
            self.path_list = self.extract_paths()
            if debug:
                print('extracted paths myself')
        else:
            self.path_list = path_list
            if debug:
                print('took your path_list')

        # if no sensors_list are given, take this default list
        if sensors_list is None:
            self.sensors_list = ['dist', 'pos', 'acc', 'gyro']
            if debug:
                print('took default sensor_list')
        else:
            self.sensors_list = sensors_list
            if debug:
                print('took your sensor_list')

        #return path_list, sensors_list

    def start(self):  # TODO -------------------------------------------------------------------------------------------
        # durch path_list iterieren und für jeden Path:
        # eigene Session erstellen
        #   aber nicht als Objekt sondern als Liste von Slot-DFs
        # alle Sessions in einer Liste von Sessions speichern

        sessions = []
        all_targets = []

        paths = self.path_list
        print('-getting data', end = '')
        time = datetime.datetime.now()
        for path in paths:
            print('making session: {}'.format(path))
            data = self.extract_data(path)
            slot_borders = self.prepare_timeslots(path)
            slots = self.make_slots(data, slot_borders)
            sessions.append(slots)
            all_targets.append(self.extract_targets(path))
            print('.', end = '')

        print(' done, {}'.format(datetime.datetime.now() - time))

        print('-filling slots ', end='')
        time = datetime.datetime.now()
        #sessions = self.same_length(sessions) # bring slots to same length #FROZEN
        print(' done, {}'.format(datetime.datetime.now() - time))

        print('-deleting counter ', end='')
        time = datetime.datetime.now()
        sessions = self.delete_counter(sessions) # delete columns 'counter_kin' and'counter_inert'
        print(' done, {}'.format(datetime.datetime.now() - time))

        print('-cut sensor data ', end='')
        time = datetime.datetime.now()
        sessions = self.cut_sensor_data(sessions, self.sensors_list)
        print(' done, {}'.format(datetime.datetime.now() - time))

        print('-concat data ', end='')
        time = datetime.datetime.now()
        concat_data = self.make_concat_data(sessions)
        print(' done, {}'.format(datetime.datetime.now() - time))

        print('-concat targets ', end='')
        time = datetime.datetime.now()
        concat_targets = self.make_concat_targets(all_targets)
        print(' done, {}'.format(datetime.datetime.now() - time))

        assert len(concat_data) == len(concat_targets), 'something went wrong: concat_data and concat_target does not have same length'

        self.sessions = sessions # TODO einheitlich
        return sessions, all_targets, concat_data, concat_targets

    def extract_data(self, dir_path):
        # take path and extract hand_samples, distances, inert, event_keys, targets

        self.dist = pd.read_csv(("../../Fusion/Recorder/recordings/z-final/" + dir_path + "/kin-sample-distances_all" + ".csv"))
        self.dist['timestamp1'] /= 1000  # to get milliseconds behind decimal point
        self.dist['timestamp2'] /= 1000  # to get milliseconds behind decimal point
        self.dist['timestamp_field1'] /= 1000  # to get milliseconds behind decimal point
        self.dist['timestamp_field2'] /= 1000  # to get milliseconds behind decimal point

        # Kinect ------------------------------------------------------------------------------------------------------
        self.kin = tools.get_data(dir_path, "kin-sample-hand")  # get kinect dataframe from csv

        # pos-Daten normalisieren, damit sie sich um 0 bewegen
        self.kin['pos_x'] = self.kin['pos_x'] - self.kin['pos_x'].mean()  # TODO kürzer (-=)
        self.kin['pos_y'] = self.kin['pos_y'] - self.kin['pos_y'].mean()
        self.kin['pos_z'] = self.kin['pos_z'] - self.kin['pos_z'].mean()

        assert len(self.kin) == len(self.dist), 'positions-samples und distances-samples sind NICHT GLEICH LANG!!'

        self.kin = pd.concat([self.kin, self.dist], axis=1, sort=False)

        self.kin = self.kin[['counter', 'timestamp',
                             'pos_x', 'pos_y', 'pos_z',
                             'distance1', 'distance2', 'distance_field1', 'distance_field2']]
        self.kin = self.kin.rename(index=str, columns={'distance1': 'distance_tire1', 'distance2': 'distance_tire2'})

        # Events ------------------------------------------------------------------------------------------------------
        self.events = tools.get_data(dir_path, "kin-sample-events")

        # Timeslots from Event-Keys -----------------------------------------------------------------------------------
        self.prepare_timeslots(dir_path)

        # Inertial ----------------------------------------------------------------------------------------------------
        self.inert = tools.get_data(dir_path, "inert-sample")  # get inertial dataframe from csv
        self.inert = pd.DataFrame({'counter': self.inert['counter'],
                                   'timestamp': self.inert['timestamp'],
                                   'acc_x': self.inert['acc_x'],
                                   'acc_y': self.inert['acc_y'],
                                   'acc_z': self.inert['acc_z'],
                                   'gyro_x': self.inert['gyro_x'],
                                   'gyro_y': self.inert['gyro_y'],
                                   'gyro_z': self.inert['gyro_z']})

        # TODO beschreiben
        self.inert['timestamp'] = pd.to_datetime(self.inert['timestamp'], unit='s')  # unix timestamp to datetime
        self.inert = tools.uniquify_timestamp(self.inert)
        self.inert = self.inert.set_index('timestamp')
        self.inert = self.inert.resample('20ms').pad()

        self.kin['timestamp'] = pd.to_datetime(self.kin['timestamp'], unit='s')  # unix timestamp to datetime
        self.kin = tools.uniquify_timestamp(self.kin)
        self.kin = self.kin.set_index('timestamp')
        self.kin = self.kin.resample('20ms').pad()

        ###############################################################################################################

        self.kin = self.kin.reset_index()
        self.inert = self.inert.reset_index()

        session_beginning = max(self.kin.iloc[0, 0], self.inert.iloc[0, 0])
        session_end = min(self.kin.iloc[-1, 0], self.inert.iloc[-1, 0])

        # kin und inert so schneiden, dass sie sich überdecken und nie alleine laufen
        self.kin_short = self.kin.loc[
            (self.kin['timestamp'] >= session_beginning) & (self.kin['timestamp'] <= session_end)]
        self.inert_short = self.inert.loc[
            (self.inert['timestamp'] >= session_beginning) & (self.inert['timestamp'] <= session_end)]

        # TODO kann weg?
        self.kin_test = self.kin_short
        self.inert_test = self.inert_short

        assert self.kin_short.iloc[0, 0] == self.inert_short.iloc[0, 0], 'erster timestamp nicht gleich!'
        assert self.kin_short.iloc[0, 0] == self.inert_short.iloc[0, 0], 'letzter timestamp nicht gleich!'
        assert len(self.kin_short) == len(self.inert_short), 'dfs nicht gleich lang!!'
        # kin und inert haben die gleiche Sampling-Rate und die selben ersten und letzten Timestamps
        # --> dürfen gemerged werden

        self.kin_short = self.kin_short.set_index('timestamp')
        self.kin_short = pd.DataFrame({'counter_kin': self.kin_short['counter'],
                                       'pos_x': self.kin_short['pos_x'],
                                       'pos_y': self.kin_short['pos_y'],
                                       'pos_z': self.kin_short['pos_z'],
                                       'distance_tire1': self.kin_short['distance_tire1'],
                                       'distance_tire2': self.kin_short['distance_tire2'],
                                       'distance_field1': self.kin_short['distance_field1'],
                                       'distance_field2': self.kin_short['distance_field2']})

        #########

        self.inert_short = self.inert_short.set_index('timestamp')
        self.inert_short = pd.DataFrame({'counter_inert': self.inert_short['counter'],
                                         'acc_x': self.inert_short['acc_x'],
                                         'acc_y': self.inert_short['acc_y'],
                                         'acc_z': self.inert_short['acc_z'],
                                         'gyro_x': self.inert_short['gyro_x'],
                                         'gyro_y': self.inert_short['gyro_y'],
                                         'gyro_z': self.inert_short['gyro_z']})

        # TODO counter_kin und counter_inert noch weg?
        # ----------------------------concat kin and inert to kininert # TODO name it data instead of kininert?
        self.kininert = pd.concat([self.kin_short, self.inert_short], axis=1, sort=False)

        return self.kininert  # TODO was wird returned?! "Method" ?? kinert dataframe zurückgeben

    def prepare_timeslots(self, dir_path):  # get keys and get timeslot-segments out # TODO umbenennen: slot_borders # TODO SEGMENTS!!!!
        # get event_keys and pair them into button_down and button_up

        self.keys = pd.read_csv(("../../Fusion/Recorder/recordings/z-final/" + dir_path + "/kin-sample-events-keys" + ".csv"))
        self.keys['timestamp'] /= 1000  # to get milliseconds behind decimal point
        self.keys['timestamp'] = pd.to_datetime(self.keys['timestamp'], unit='s')  # unix timestamp to datetime

        try:
            self.keys = self.keys[self.keys.scancode != 30]  # get a out
        except:
            print('no a')

        # TODO delete everything after z?

        try:
            self.keys = self.keys[self.keys.scancode != 21]  # get z out
        except:
            print('no z')

        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                   'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # take unneccesary button-ups from number-and-letter input and delete it
        # we only need number and letter buttom-downs and buttom-down and buttom-up from spaces
        for i in range(len(self.keys)):
            if self.keys.iloc[i]["unicode"] == "b":  # ################################
                self.keys.iat[i, 2] = "beginning"
                if self.keys.iloc[i + 1]["scancode"] == self.keys.iloc[i]["scancode"] and \
                        self.keys.iloc[i + 1]["unicode"] == "none":  # i+1 is button-up for i
                    self.keys.iat[i + 1, 2] = "end"
            elif self.keys.iloc[i]["unicode"] in numbers or \
                    self.keys.iloc[i]["unicode"] in letters:  # ############################
                if self.keys.iloc[i + 1]["scancode"] == self.keys.iloc[i]["scancode"] and \
                        self.keys.iloc[i + 1]["unicode"] == "none":  # i+1 is button-up for i
                    self.keys.iat[i + 1, 2] = "delete"

        self.keys = self.keys[self.keys.unicode != 'delete']  # delete all rows marked with delete
        self.keys = self.keys[~self.keys['unicode'].isin(numbers)]  # delete all rows with numbers in unicode == POIs

        # TODO!
        # test dass rows am Ende Format: letter - beginning -  end hat??
        # und dass scancode von 2&3 der selbe ist.
        self.slots_long = pd.DataFrame({'event_from': self.keys['event_type'].iloc[0::2].values,
                                        'event_to': self.keys['event_type'].iloc[1::2].values,
                                        'unicode_from': self.keys['unicode'].iloc[0::2].values,
                                        'unicode_to': self.keys['unicode'].iloc[1::2].values,
                                        'scancode_from': self.keys['scancode'].iloc[0::2].values,
                                        'scancode_to': self.keys['scancode'].iloc[1::2].values,
                                        'counter_from': self.keys['counter'].iloc[0::2].values,
                                        'counter_to': self.keys['counter'].iloc[1::2].values,
                                        'timestamp_from': self.keys['timestamp'].iloc[0::2].values,
                                        'timestamp_to': self.keys['timestamp'].iloc[1::2].values})

        self.slots_short = pd.DataFrame({'counter_from': self.keys['counter'].iloc[0::2].values,
                                         'counter_to': self.keys['counter'].iloc[1::2].values,
                                         'timestamp_from': self.keys['timestamp'].iloc[0::2].values,
                                         'timestamp_to': self.keys['timestamp'].iloc[1::2].values})

        return self.slots_short

    def make_slots(self, data, slot_borders):
        # gets kininert data and slot_borders and returns slices of kininert data according to slot_borders
        # returns list 'slots' with dataframe with each slot

        # old:      slot object for every slot in slots
        # now:      make list of dataframes, one df per slot with sliced data in it
        slots = []

        for index, row in slot_borders.iterrows():
            # TODO slots[index] oder slots.append??
            slot = self.make_single_slot(data, row['timestamp_from'], row['timestamp_to'])
            slots.append(slot)
            # TODO keine eigene Methode für make_single_slots, sondern direkt die paar Zeilen hier rein?

        assert slot_borders.shape[0] == len(slots), 'something went wrong, nmb of slots != nmb of slot_borders'

        return slots

    def make_single_slot(self, data, ts_from, ts_to):

        data = data.reset_index()

        # kininert slicen auf timeslot from current slot
        data = data[(data['timestamp'] >= ts_from) & (data['timestamp'] <= ts_to)]

        # set timestamp back as index
        data = data.set_index('timestamp')

        # TODO hier wenn notwendig noch aufteilen in dist, pos, acc, gyro
        # um sie dann zu melten, um sie zu plotten

        return data

    def same_length(self, sessions): # TODO do it simpler!!!! -> get longest list in list, with yield!
        # alle slots von auf eine Länge bekommen
        #   = samples von allen slots auffüllen, bis alle gleich lang

        #def lengths(x):
        #    if isinstance(x, list):
        #        yield len(x)
        #        for y in x:
        #            for z in lengths(y):
        #                yield z#

        # find length of longest segment
        longest = 0
        for session in sessions:
            for slot in session:
                longest = max(longest, len(slot))
        #print('longest = {}'.format(longest))

        # --> alle anderen Samplings müssen so lange sein
        #    -> bei jedem Sampling wird das letzte Sample 'eingefrohen'
        shortest = 9999
        for session in sessions:
            for slot in session:
                shortest = min(shortest, len(slot))
        #print('shortest = {}'.format(shortest))

        #fill slots with frozen last sample
        #print('filling slots ', end = '')
        for i_session, session in enumerate(sessions):
            for i_slot, slot in enumerate(session):
                length = len(slot)
                diff = longest - length
                last_sample = slot.iloc[-1:]  # last Sample in current Slot
                #print('len before = {}'.format(len(slot)), end='')

                # fill slot with frozen samples
                for i in range(diff):
                    last_sample.iloc[0][0] += 1  # increase counter
                    last_sample.index += datetime.timedelta(milliseconds=20)  # add 20 ms
                    slot = slot.append(last_sample)
                    # print(' new length = {}'.format(len(slot)))

                # save filled-up slot back into sessions[i_session][i_slot]
                sessions[i_session][i_slot] = slot
                #print(', len after = {}'.format(len(slot)))
            print('.', end ='')

        shortest = 9999
        for session in sessions:
            for slot in session:
                shortest = min(shortest, len(slot))
        #print('shortest = {}'.format(shortest))

        assert longest == shortest, 'filling of slots with frozen last sample did not work'

        if debug:
            if longest == shortest:
                print('slots now filled with frozen last sample -> now all slots have the same length')

        return sessions

    def delete_counter(self, sessions):
        # schmeißt counter_kin und counter_inert raus
        # -------------------TODO: schon früher raus schmeißen? vor Übergabe in Slots?

        for i_session, session in enumerate(sessions):
            for i_slot, slot in enumerate(session):
                slot.pop('counter_kin')
                slot.pop('counter_inert')
                sessions[i_session][i_slot] = slot
            print('.', end ='')


        #for d in range(len(self.sessions)):  # für jede Session
        #    for s in range(len(self.sessions[d].slots)):  # für jeden Slot
        #
        #        #self.sessions[d].slots[s].kininert.pop('counter_kin')
        #        #self.sessions[d].slots[s].kininert.pop('counter_inert')

        return sessions

    def cut_sensor_data(self, sessions, sensors_list):

        # TODO nicht sessions cutten sondern data, bevor es in die sessions geht?
        # TODO mergen mit extract_data Methode?
        # -> direkt dort einfach Abfragen ob Sensorkanal in sensors_list und dann entweder hinzufügen oder nicht

        # TODO alle Abfragen in loop-loop rein?
        # also nicht 4 Abfragen mit looploop
        # oder looploop mit 4 Abfragen drinnen?

        # if dist nicht in list -> raus schmeißen
        if 'dist' not in sensors_list:
            # dist und gyro raus schmeißen
            for i_session, session in enumerate(sessions):  # für jede Session
                for i_slot, slot in enumerate(session):  # für jeden Slot
                    slot.pop('distance_tire1')
                    slot.pop('distance_tire2')
                    slot.pop('distance_field1')
                    slot.pop('distance_field2')
                    sessions[i_session][i_slot] = slot
                print('.', end='')
        # -------------------

        # if pos nicht in list -> raus schmeißen
        if 'pos' not in sensors_list:
            # dist und gyro raus schmeißen
            for i_session, session in enumerate(sessions):  # für jede Session
                for i_slot, slot in enumerate(session):  # für jeden Slot
                    slot.pop('pos_x')
                    slot.pop('pos_y')
                    slot.pop('pos_z')
                    sessions[i_session][i_slot] = slot
                print('.', end='')
        # -------------------

        # if acc nicht in list -> raus schmeißen
        if 'acc' not in sensors_list:
            # print('dist not in data_list -> rausschmeißen')
            # dist und gyro raus schmeißen
            for i_session, session in enumerate(sessions):  # für jede Session
                for i_slot, slot in enumerate(session):  # für jeden Slot
                    slot.pop('acc_x')
                    slot.pop('acc_y')
                    slot.pop('acc_z')
                    sessions[i_session][i_slot] = slot
                print('.', end='')
        # -------------------

        # if gyro nicht in list -> raus schmeißen
        if 'gyro' not in sensors_list:
            # dist und gyro raus schmeißen
            for i_session, session in enumerate(sessions):  # für jede Session
                for i_slot, slot in enumerate(session):  # für jeden Slot
                    slot.pop('gyro_x')
                    slot.pop('gyro_y')
                    slot.pop('gyro_z')
                    sessions[i_session][i_slot] = slot
                print('.', end='')

        return sessions

    def make_concat_data(self, sessions):
        # make sessions_list with samples of all sessions
        # --------------------------------------------------TODO make list of one session first, and later concat?
        # get concat_data list for all sessions (all slots in all sessions)
        #
        concat_data = []
        #concat_segments = []
        nmb_slots = 0

        for i_session, session in enumerate(sessions):  # für jede Session
            nmb_slots += len(session)
            for i_slot, slot in enumerate(session):  # für jeden Slot
                seg = slot  # aktueller Slot
                seg = seg.values  # numpy array aus aktuellem Slot
                seg = seg.flatten()  # numpy array zu liste flatten
                concat_data.append(seg)  # liste als Zeile an session_list anfügen
            # print('.', end='')

                # print(' {} - {} slot.values.flattened appended to sessions_list, concat_data is now {} long'.format(i_session, i_slot, len(concat_data)))
        #print('concat_data is now {} long'.format(len(concat_data)))
        #print('number of all slots (in all sessions) = {}'.format(nmb_slots))
        assert len(
            concat_data) == nmb_slots, 'concat_data didnt work properly, length does not fit: len(concat_data) = {}, nmb_slots = {}'.format(
            len(concat_data), nmb_slots)

        # TODO assert, der schaut ob Länge von session_list übereinstimmt mit logischer Zahl
        # Länge sollte ja Anzahl an Sessions x Anzahl an Segmenten sein
        # - Anzahl an Segmente sollte bei jeder Session gleich groß sein? - nein!
        # -> also nicht Anzahl_sessions x Anzahl_Segmente, sondern durch iterieren und jeweils Anzahl_segments pro session zusammenzählen


        if False:
            for i_session, session in enumerate(sessions):  # für jede Session
                print('length of session {} = {}'.format(i_session, len(session)))
                print('-lengths of slot = nmbs of segments in slot = ', end='')
                for i_slot, slot in enumerate(session):  # für jeden Slot
                    print(', {}'.format(len(slot)), end='')
                print('')

            nmbs_slots = 0
            for i_session, session in enumerate(sessions):  # für jede Session
                nmbs_slots += len(session)
            print('number of all slots (in all sessions) = {}'.format(nmbs_slots))

        # TODO assert, der schaut ob zeile.shape ((5512, )) übereinstimmt mit allen

        # TODO generell assert, der schaut ob Größe stimmt

        return concat_data#, concat_segments

    def extract_targets(self, dir_path):
        # Targets -----------------------------------------------------------------------------------------------------
        full_path = ("../../Fusion/Recorder/recordings/z-final/" + dir_path + "/targets.csv")
        targets = pd.read_csv(full_path)
        targets = targets.drop([targets.columns[0]], axis='columns')
        return targets

    def make_concat_targets(self, all_targets):
        # concat targets into one list for clf

        concat_targets = []
        # targets_list machen wie sessions_list
        # TODO sessions_list umbenennen?

        for i_targets, targets in enumerate(all_targets):
            for i_target in range(int(targets.shape[1])):
                tar = targets.iloc[0, i_target]
                concat_targets.append(tar)
            # print('---')
            # print(self.targets_list)
            print('.', end='')
        return concat_targets

    def TODO(self):

        assert self.targets.shape[1] == len(self.slots), ' Anzahl Slots = ' + str(len(self.slots)) \
                                                         + ' und Anzahl Targets = ' + str(self.targets.shape[1]) \
                                                         + ' von session ' + str(
            self.dir_path) + ' sind NICHT GLEICH LANG!'





























