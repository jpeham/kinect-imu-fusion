import pandas as pd
import datetime

import sklearn
from sklearn.metrics import accuracy_score

from rename_labels import label_dict # dict with names for labels (Stift rein = 10, Schrauben = 110/111/120/120,...)
from rename_labels import label_to_index_dict # dict with indices for labels (Schrauben, Stift rein,...)
from rename_labels import get_opposites_out, merge_variants_screw_lr, merge_variants_screw_abc


def get_data(data_path, data_type):
    
    #print("trying to add data from " + data_type)
    
    #import csv
    full_path = ("../Recorder/recordings/z-final/" + data_path + "/" + data_type + ".csv")
    data_df = pd.read_csv(full_path)
    
    if data_type != "inert-sample":
        data_df['timestamp'] /= 1000 # to get milliseconds behind decimal point
    else:
        data_df.pop('timestamp_sensor') # get rid of extra timestamp in front (only on inert data)
    if data_type == "kin-sample-events":
        #data_df = data_df[data_df.event_type != 4] # delete rows where event = 4 (mouse movement)
        # --> get events to minimum event_types --> graph is minimal height (readability)
        data_df = data_df[~data_df['event_type'].isin([1, 4, 12, 16, 17])] # delete all rows with numbers in unicode
        
    #data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], unit='s') # unix timestamp to datetime
    
    #print("---data from " + data_type + " added")
    
    return data_df


def diff(df, col_name):
    df1 = df[df.variable == col_name] # extract list of samples with variable of interest
    df1.pop('variable') # delete column with variable (not suitable for diff)
    df1_diff = df1.diff() # differentiate
    df1_diff['variable'] = col_name
    return df1_diff


def uniquify_timestamp(uniquify_df):
    debug_uniquify = False
    
    while not uniquify_df["timestamp"].is_unique:
        if debug_uniquify:
            print('df is not unique')
        duplicates = uniquify_df["timestamp"].duplicated() # bools if timestamps are duplicates
        if debug_uniquify:
            print('{} timestamps are duplicates'.format(len(uniquify_df.loc[duplicates, "timestamp"])))

        #print('list of duplicates:')
        #print(uniquify_df.loc[duplicates, "timestamp"])
        if debug_uniquify:
            print('-'*30)

        uniquify_df.loc[duplicates, "timestamp"] += datetime.timedelta(milliseconds=1) # adds 1 ms to duplicate timestamps

    if uniquify_df["timestamp"].is_unique:
        if debug_uniquify:
            print('df is unique')
    
    return uniquify_df


def make_name(train_subjects=None, subjects=None, sensors_list=None,
              pipe=None, klassifikator=None,
              diff_features_list=None, stat_features_list=None,
              fuse_slots=None,
              merge_opposites=False, merge_screw_lr=False, merge_screw_abc=False,
              with_date=True,
              show_time=False, show_dots=False, debug=False, deep_debug=False):
    """
    get name out of given parameters

    :param train_subjects: subjects to be trained on, of not None subjects = test_subjects
    :param subjects: list of subjects, not yet split up
    :param sensors_list:
    :param fuse_slots: list of slots to fuse, first start slot, then end slot, and names they should be given
    :param merge_opposites: bool whether to merge opposites of turning motions or not
    :param merge_screw_lr: bool whether to merge screwing motions into lr
    :param merge_screw_abc:
    :param with_date: bool if date should be in name (for id) (prevent overwriting)
    :param show_time: debug bool to show elapsed time
    :param show_dots: debug bool to show dots to not get impatient
    :param debug: debug bool to show first layer debug stuff
    :param deep_debug: debug bool to show deeper debug stuff
    :return:
    """

    # TODO if no train_subjects -> UI
    #  sonst UD, und extra Ausgabe für test und train subjects

    if train_subjects is not None:
        if debug: print('add train and test subjects TODO {}'.format(train_subjects))
        subjects = 'UD_' + train_subjects + '_' + subjects
    if debug: print('subjects = {}'.format(subjects))

    sensors_str = '_'.join(sensors_list)

    diff_features_list = '_'.join(diff_features_list)
    stat_features_list = '_'.join(stat_features_list)

    if sensors_list == ['dist', 'pos']:
        sensors_str = 'kinect'
    elif sensors_list == ['acc', 'gyro']:
        sensors_str = 'imu'
    elif sensors_list == ['dist', 'pos', 'acc', 'gyro']:
        sensors_str = 'fusion'

    merge_bools = ''
    if merge_opposites:
        merge_bools += 'oops'
    else:
        merge_bools += 'noops'

    if merge_screw_lr:
        merge_bools += 'nolr'

    if merge_screw_abc:
        merge_bools += 'noabc'

    fuse = ''
    if fuse_slots is None:
        fuse = ''
    elif fuse_slots == [[210, 310], [240, 340], [250, 350]]:
        fuse = '_fuseg'
    elif fuse_slots == [[210, 310], [230, 330], [235, 335]]:
        fuse = '_fusehs'
    elif fuse_slots == [[210, 310], [220, 320], [225, 325]]:
        fuse = '_fuseh'

    name = subjects + '_' + sensors_str
    if pipe: name = name + '_' + pipe
    if klassifikator: name = name + '_' + klassifikator
    name = name + '_' + diff_features_list + '_' + stat_features_list
    name = name + '_' + merge_bools + fuse  # concat info
    if with_date:
        name = name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # add time
    name = 'plots/' + name  # add directory

    return name


def save_infos(sensors_list=None, pipe=None, klassifikator=None,
               diff_features_list=None, stat_features_list=None,
               train_path_list=None, path_list=None,
               train_subjects=None, subjects=None,
               targets=None, predicted=None, rename=None, mode=None, name=None):

    if name is None:
        name = 'plots/test_1234'

    file = open(name + ".txt", "w")

    file.write('Name: {}\n'.format(name))
    file.write('Analysed sensors: {}\n'.format('_'.join(sensors_list)))
    if pipe: file.write('Pipe: {}\n'.format(pipe))
    if klassifikator: file.write('Klassifikator: {}\n'.format(klassifikator))
    file.write('Differential features: {}\n'.format('_'.join(diff_features_list)))
    file.write('Statistical features: {}\n'.format('_'.join(stat_features_list)))

    print('---train_subjects = {}'.format(train_subjects))

    if train_subjects is not None:
        file.write('Train on {} Sessions from {}\n'.format(len(train_path_list), '_'.join(train_subjects)))
        file.write('Test on {} Sessions from {}\n'.format(len(path_list), subjects))
    else:
        file.write('Number of Sessions = {}\n'.format(len(path_list)))
        file.write('Subjects = {}\n'.format('_'.join(subjects)))

    file.write('Accuracy = {}\n'.format(accuracy_score(targets, predicted) * 100))

    file.write('')

    report = sklearn.metrics.classification_report(targets, predicted)

    file.write(report)
    file.close()

    # -----------------------------------------------------

    # save accuracy alone in separate txt file
    file = open(name + "_accuracy.txt", "w")
    acc = (accuracy_score(targets, predicted) * 100)
    acc = round(acc, 2)
    file.write(str(acc))
    file.close()

    # -----------------------------------------------------

    # save accuracy of Schrauben actions
    indices = ([i for i, x in enumerate(targets) if x in [110, 111, 115, 120, 121, 125, 130, 131, 135, 140, 141]])
    print(indices)
    pred_schrauben = [x for i, x in enumerate(predicted) if i in indices]  # select predicted with schrauben index
    print(pred_schrauben)
    targets_schrauben = ([x for i, x in enumerate(targets) if i in indices])
    print(targets_schrauben)
    acc_schrauben = accuracy_score(targets_schrauben, pred_schrauben) * 100
    print(acc_schrauben)
    acc_schrauben = round(acc_schrauben, 2)
    print(acc_schrauben)
    file = open(name + "_accuracy_schrauben.txt", "w")
    file.write(str(acc_schrauben))
    file.close()











def get_report_from_txt(path, debug=False, deep_debug=False):
    assert path is not None, 'no path!'
    if debug: print(path)
    # open file and get txt
    myfile = open(path)
    txt = myfile.read()
    myfile.close()

    lines = txt.split('\n')  # cut txt in lines
    # print(lines)
    acc_str = lines[6][11:]  # take only lines with precisions
    acc = float(acc_str)  # get total accuracy from txt as well
    acc = round(acc, 2)  # and round it

    classes = []  # list for action numbers/names
    plotMat = []  # report

    for line in lines[9: (len(lines) - 5)]:  # iterate over lines with precisions
        if deep_debug: print(line)
        t = line.split()
        # print('t = {}'.format(t))
        classes.append(t[0])  # t[0] = Class names # get our class values
        v = [float(x) for x in t[1: len(t) - 1]]
        if deep_debug: print('v = {}'.format(v))
        plotMat.append(v)  # rest of t = (precision, recall, f1-score, support) values
        # precisions.append(t[1])
        if deep_debug: print('-' * 50)
    if deep_debug: print('plotMat = {}'.format(plotMat))
    precisions = [elem[0] for elem in plotMat]
    precisions = [elem * 100 for elem in precisions]
    precisions = [round(elem, 2) for elem in precisions]

    precisions.append(acc)

    if debug: print('precisions = {}'.format(precisions))
    return precisions


# go through all_slots list and append every slot to one_slot list for specific label
# -> all_slots is list with all slots from label 10 in all_slots[0], label 20 in all_slots[1],...
# -> all_slots[0] = list with dataframes for each slot

def get_all_slots(sessions, targets, debug=False):
    all_slots = [[] for _ in range(26)]  # list with 26 empty lists

    for i_session, session in enumerate(sessions):
        # path_name = path_list[i_session]  # path name for labeling samples
        # if debug: print('- session: {}, path: {}'.format(i_session, path_list[i_session]))
        # if debug: print('-' * 50)

        for i_slot, slot in enumerate(session):
            tmp_slot = slot[:]

            if debug: print('-- slot: {}'.format(i_slot), end=', ')

            if debug: print('get target from session {} - {}'.format(i_session, i_slot))
            target = targets[i_session].iloc[0, i_slot]  # target of this slot
            target = get_opposites_out[target]  # rename with get_opposites_out
            if debug: print('target: {}'.format(target))

            target_name = label_dict[target]
            target_index = label_to_index_dict[target]

            if debug: print('target Zahl: {}'.format(target), end=', ')
            if debug: print('Name: {}'.format(target_name), end=', ')
            if debug: print('Index: {}'.format(target_index))
            if debug: print('')
            ########################################################################################
            # tmp_slot['path'] = path_name  # extra column on slot with path_name
            # if debug: print('added pathname {}'.format(path_name))

            tmp_slot['target'] = target  # extra column on slot with target
            if debug: print('added target {}'.format(target))

            # normalisiseren der einzelnen Spalten, damit sie sich um 0 bewegen -> richtig? TODO JH

            # ACHTUNG Distanz darf natürlich nicht normalisiert werden, da interessiert es uns welche echt näher an null ist
            tmp_slot['pos_x'] -= tmp_slot['pos_x'].mean()
            tmp_slot['pos_y'] -= tmp_slot['pos_y'].mean()
            tmp_slot['pos_z'] -= tmp_slot['pos_z'].mean()

            tmp_slot['acc_x'] -= tmp_slot['acc_x'].mean()
            tmp_slot['acc_y'] -= tmp_slot['acc_y'].mean()
            tmp_slot['acc_z'] -= tmp_slot['acc_z'].mean()

            tmp_slot['gyro_x'] -= tmp_slot['gyro_x'].mean()
            tmp_slot['gyro_y'] -= tmp_slot['gyro_y'].mean()
            tmp_slot['gyro_z'] -= tmp_slot['gyro_z'].mean()

            i_allslots = label_to_index_dict[target]
            if debug: print('i_allslots = index of target = {}'.format(i_allslots))

            tmp_slot['i_target'] = target_index  # extra column on slot with target index
            if debug: print('added i_target {}'.format(target_index))
            ########################################################################################
            if debug: print('-' * 70)
            if debug: print('')
            i_allslots = target_index

            all_slots[i_allslots].append(tmp_slot)
            # all_slots[i_allslots].append(1)
            # test[i_allslots].append(1)
            if debug: print('appended slot to all_slots[{}]'.format(i_allslots))

            last_all_slot = all_slots[i_allslots][-1]
            ########################################################################################
            # Tests
            if debug:
                print('-' * 70)
                # if last_all_slot.iloc[-1, -3] == path_name:
                #     print('path stimmt: {}'.format(last_all_slot.iloc[-1, -3]))
                # else:
                #     print('path sollte {} sein, ist {}'.format(path_name, last_all_slot.iloc[-1, -3]))

                if last_all_slot.iloc[-1, -2] == target:
                    print('target stimmt: {}'.format(last_all_slot.iloc[-1, -2]))
                else:
                    print('target sollte {} sein, ist {}'.format(target, last_all_slot.iloc[-1, -2]))

                if last_all_slot.iloc[-1, -1] == target_index:
                    print('i_target index stimmt: {}'.format(last_all_slot.iloc[-1, -1]))
                else:
                    print('i_target index sollte {} sein, ist {}'.format(target_index, last_all_slot.iloc[-1, -1]))
            ########################################################################################

        if debug: print('-' * 100)


    ##############################################################

    # set timestamps to zero
    # and add column with counter per one_slot

    # single slots have to be kept separate from other slots in one_slot
    # every slot will have a line/color in lineplot

    for i_one_slot, one_slot in enumerate(all_slots):
        for i_slot, slot in enumerate(one_slot):
            if debug:
                print('--------slot {}--------'.format(i_slot))
                print(slot.iloc[0:3, 0:3])

            slot = slot.reset_index()  # reset timestamp index, so we can work with timestamp
            if debug:
                print('')
                print('reset index')
                print(slot.iloc[0:3, 0:3])

            slot['timestamp'] -= slot['timestamp'][0]  # substract first timestamp from
            if debug:
                print('')
                print('reset timestamp')
                print(slot.iloc[0:3, 0:3])

            slot = slot.set_index('timestamp')  # set timestamp back as index again
            if debug:
                print('')
                print('set index')
                print(slot.iloc[0:3, 0:3])

            # add column with i_slot
            slot['i_slot'] = i_slot  # extra column on slot with path_name
            if debug: print('added i_slot {}'.format(i_slot))

            one_slot[i_slot] = slot  # save slot in one_slot
            if debug:
                print('')
                print('saved slot in one_slot')
                print(one_slot[i_slot].iloc[0:3, 0:3])

            if debug: print('-' * 70)

        all_slots[i_one_slot] = one_slot  # save one_slot in all_slots

    ###############################################

    # put all one_slots into one list (all_slots2)

    # test = []
    # make empty temp list for all_slots because somehow it wouldnt take it if I change the original all_slots list....
    temp_all_slots = [[] for _ in range(26)]  # list with 26 empty lists

    # iterate through all_slots and concat dfs in one_slot and append to test
    for i_one_slot, one_slot in enumerate(all_slots):
        if debug: print('current slot = {}'.format(i_one_slot))
        temp_one_slot = []
        try:
            temp_one_slot = pd.concat(one_slot, axis=0)  # concat all one_slots
            if debug:
                print('--------temp head:')
                print(temp_one_slot.iloc[0:3, -4:])
                print('--------temp tail')
                print(temp_one_slot.iloc[0:3, -4:])
            # test.append(temp_one_slot)
            # print('asdf')
            # temp_all_slots[i_one_slot] = temp_one_slot
        except:
            print('not working for {}'.format(i_one_slot))
        # temp_one_slot = pd.concat(one_slot)#, axis=0)
        # temp_all_slots[i_one_slot] = temp_one_slot

        # test.append(temp_one_slot)
        print('asdf')
        temp_all_slots[i_one_slot] = temp_one_slot  # save temp_one_slot in temp_all_slots

        if debug: print('-' * 100)
    # all_slots2 = test
    all_slots = temp_all_slots  # save temp_all_slots in all_slots2


    return all_slots
