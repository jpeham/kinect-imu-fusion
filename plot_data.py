import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline # TODO wieso da? noch gebraucht? oder weg
import seaborn as sns
import datetime

import pandas
import sklearn

from sklearn.metrics import accuracy_score
from plot_cm import plot_cm # Confusion Matrix from JH

from rename_labels import label_dict, label_to_index_dict, color_dict

# TODO mods damit Hintergrund grau und Linien, und bessere Quali

def plot_slot(slot, target, rename):
    slot = slot.reset_index()

    dist_df = pd.melt(slot, id_vars=['timestamp'], value_vars=['distance_tire1', 'distance_tire2', 'distance_field1', 'distance_field2'])
    pos_df = pd.melt(slot, id_vars=['timestamp'], value_vars=['pos_x', 'pos_y', 'pos_z'])
    acc_df = pd.melt(slot, id_vars=['timestamp'], value_vars=['acc_x', 'acc_y', 'acc_z'])
    gyro_df = pd.melt(slot, id_vars=['timestamp'], value_vars=['gyro_x', 'gyro_y', 'gyro_z'])

    # build plots
    # TODO set Title with Names of Slots (with dict)

    fig, ax =plt.subplots(4,1)
    fig.subplots_adjust(hspace=0.6)

    dist_ax_ = sns.lineplot(x="timestamp",y="value", hue="variable", legend="full", data=dist_df, ax=ax[0]).set_title("{} - distance".format(rename[target]))
    pos_ax = sns.lineplot(x="timestamp", y="value", hue="variable", legend="full", data=pos_df, ax=ax[1]).set_title("{} - position".format(rename[target]))
    acc_ax = sns.lineplot(x="timestamp", y="value", hue="variable", legend="full", data=acc_df, ax=ax[2]).set_title("{} - accelerometer".format(rename[target]))
    gyro_ax = sns.lineplot(x="timestamp", y="value", hue="variable", legend="full", data=gyro_df, ax=ax[3]).set_title("{} - gyroscop".format(rename[target]))

    # Label the axes
    #ax.set(title='USA births by day of year (1969-1988)')  # ,ylabel='average daily births')

    #ax.text(0.85, 0.85, 'Text Here', fontsize=9)  # add text
    plt.xlabel('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    plt.ylabel('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    #ax.xaxis.set_label_position('top')  # move 'Hypothesis' label to top

    dist_ax_.figure.set_size_inches(18.5, 10.5)


def make_cm_tt(sensors_lists, diff_features_list, stat_features_list,
            train_path_list, test_path_list,
            train_subjects, test_subjects,
            targets, predicted, rename, name=None, pipe=None, klassifikator=None):
    # print infos of predicted data and plot confusion matrix with plot_cm

    print('Analysed sensors: ', *sensors_lists, sep=", ")
    if pipe: print('Pipe: {}'.format(pipe))
    if klassifikator: print('Klassifikator: {}'.format(klassifikator))
    print('Differential features: ', *diff_features_list, sep=", ", end=' - ')
    print('Statistical features: ', *stat_features_list, sep=", ")

    print('Number of Train Sessions = {}'.format(len(train_path_list)), end=' - ')
    print('Number of Test Sessions = {}'.format(len(test_path_list)))

    print('Train Subjects = ', *train_subjects, sep=", ")#, end=' - ')
    print('Test Subjects = ', *test_subjects, sep=", ")

    plot_cm(targets, predicted, rename=rename, name=name)
    print('Accuracy = {}'.format(accuracy_score(targets, predicted)*100))


def make_cm_all(sensors_list, diff_features_list, stat_features_list,
            path_list, subjects,
            targets, predicted, rename, mode, name=None, pipe=None, klassifikator=None):
    # print infos of predicted data and plot confusion matrix with plot_cm

    print('Analysed sensors: ', *sensors_list, sep=", ")
    if pipe: print('Pipe: {}'.format(pipe))
    if klassifikator: print('Klassifikator: {}'.format(klassifikator))
    print('Differential features: ', *diff_features_list, sep=", ", end=' - ')
    print('Statistical features: ', *stat_features_list, sep=", ")

    print('Number of Sessions = {}'.format(len(path_list)))
    print('Subjects = ', *subjects, sep=", ")

    if name is None:
        name = 'plots/test_1234' + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # add time

    plot_cm(targets, predicted, rename=rename, name=name)

    print('Accuracy = {}'.format(accuracy_score(targets, predicted)*100))
    print('Mode = {}'.format(mode))


def prep_one_slot(sessions, targets, path_list):
    # initialisieren mit concat_targets?
    # durchiterieren und pro target in concat_targets eine leere Liste anfügen?
    # Achtung, nicht jede Session hat gleich viele Slots?
    # --> anschauen welche Targets in concat_targets?
    # --> mal 21 slots machen, je mit Schlüssel (dict?) mit Index und label names?
    # -> durch all_slots iterieren und das nehmen mit Schlüssel von Label

    # go through all_slots list and append every slot to one_slot list for specific label
    # -> all_slots is list with all slots from label 10 in all_slots[0], label 20 in all_slots[1],...
    # -> all_slots[0] = list with dataframes for each slot

    all_slots = [[] for _ in range(26)]  # list with 26 empty lists

    debug = False

    for i_session, session in enumerate(sessions):
        path_name = path_list[i_session]  # path name for labeling samples
        if debug: print('- session: {}, path: {}'.format(i_session, path_list[i_session]))
        if debug: print('-' * 50)

        for i_slot, slot in enumerate(session):

            tmp_slot = slot[:]

            if debug: print('-- slot: {}'.format(i_slot), end=', ')

            if debug: print('get target from session {} - {}'.format(i_session, i_slot))
            target = targets[i_session].iloc[0, i_slot]  # target of this slot
            if debug: print('target: {}'.format(target))

            target_name = label_dict[target]
            target_index = label_to_index_dict[target]

            if debug: print('target Zahl: {}'.format(target), end=', ')
            if debug: print('Name: {}'.format(target_name), end=', ')
            if debug: print('Index: {}'.format(target_index))
            if debug: print('')
            ########################################################################################
            tmp_slot['path'] = path_name  # extra column on slot with path_name
            if debug: print('added pathname {}'.format(path_name))

            tmp_slot['target'] = target  # extra column on slot with target
            if debug: print('added target {}'.format(target))

            # normalisiseren der einzelnen Spalten, damit sie sich um 0 bewegen -> richtig? TODO JH
            tmp_slot['distance_tire1'] -= tmp_slot['distance_tire1'].mean()
            tmp_slot['distance_tire2'] -= tmp_slot['distance_tire2'].mean()
            tmp_slot['distance_field1'] -= tmp_slot['distance_field1'].mean()
            tmp_slot['distance_field2'] -= tmp_slot['distance_field2'].mean()

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
                if last_all_slot.iloc[-1, -3] == path_name:
                    print('path stimmt: {}'.format(last_all_slot.iloc[-1, -3]))
                else:
                    print('path sollte {} sein, ist {}'.format(path_name, last_all_slot.iloc[-1, -3]))

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

    # set timestamps to zero
    # and add column with counter per one_slot

    # single slots have to be kept separate from other slots in one_slot
    # every slot will have a line/color in lineplot

    debug = False

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

    # put all one_slots into one list (all_slots2)

    # test = []
    # make empty temp list for all_slots because somehow it wouldnt take it if I change the original all_slots list....
    temp_all_slots = [[] for _ in range(26)]  # list with 26 empty lists

    debug = False

    # iterate through all_slots and concat dfs in one_slot and append to test
    for i_one_slot, one_slot in enumerate(all_slots):
        if debug: print('current slot = {}'.format(i_one_slot))
        temp_one_slot = []
        #try:


        temp_one_slot = pd.concat(one_slot, axis=0)  # concat all one_slots
        if debug:
            print('--------temp head:')
            print(temp_one_slot.iloc[0:3, -4:])
            print('--------temp tail')
            print(temp_one_slot.iloc[0:3, -4:])
        # test.append(temp_one_slot)
        # print('asdf')
        # temp_all_slots[i_one_slot] = temp_one_slot


        #except:
        #    print('concating one_slot not working for one_slot {}'.format(i_one_slot))
        # temp_one_slot = pd.concat(one_slot)#, axis=0)
        # temp_all_slots[i_one_slot] = temp_one_slot

        # test.append(temp_one_slot)
        temp_all_slots[i_one_slot] = temp_one_slot  # save temp_one_slot in temp_all_slots

        if debug: print('-' * 100)
    # all_slots2 = test
    all_slots2 = temp_all_slots  # save temp_all_slots in all_slots2

    # all_slots ist liste aus dfs mit je allen slots einer Bewegung
    # --> jetzt können darauf einzelne Spalten entnommen, gemelted und geplottet werden

    one_slot_id = 11  # select one one_slot - welche Bewegung möchte ich plotten?
    # 11 = Schrauben_l_Stift
    # 17 = Hand zu Hebel
    # 20 = Hebel kippen
    # 24 = Schiene zickzack

    one_slot = all_slots2[one_slot_id]
    one_slot = one_slot.reset_index()

    plot_one_slot(one_slot, one_slot_id)
    #plot_one_slot_dist(one_slot, one_slot_id)


def plot_one_slot(one_slot, one_slot_id):
    dist_1_df = pd.concat([one_slot['timestamp'], one_slot['distance_tire1'], one_slot['i_slot']], axis=1)
    dist_2_df = pd.concat([one_slot['timestamp'], one_slot['distance_tire2'], one_slot['i_slot']], axis=1)
    dist_3_df = pd.concat([one_slot['timestamp'], one_slot['distance_field1'], one_slot['i_slot']], axis=1)
    dist_4_df = pd.concat([one_slot['timestamp'], one_slot['distance_field2'], one_slot['i_slot']], axis=1)
    pos_x_df = pd.concat([one_slot['timestamp'], one_slot['pos_x'], one_slot['i_slot']], axis=1)
    pos_y_df = pd.concat([one_slot['timestamp'], one_slot['pos_y'], one_slot['i_slot']], axis=1)
    pos_z_df = pd.concat([one_slot['timestamp'], one_slot['pos_z'], one_slot['i_slot']], axis=1)
    acc_x_df = pd.concat([one_slot['timestamp'], one_slot['acc_x'], one_slot['i_slot']], axis=1)
    acc_y_df = pd.concat([one_slot['timestamp'], one_slot['acc_y'], one_slot['i_slot']], axis=1)
    acc_z_df = pd.concat([one_slot['timestamp'], one_slot['acc_z'], one_slot['i_slot']], axis=1)
    gyro_x_df = pd.concat([one_slot['timestamp'], one_slot['gyro_x'], one_slot['i_slot']], axis=1)
    gyro_y_df = pd.concat([one_slot['timestamp'], one_slot['gyro_y'], one_slot['i_slot']], axis=1)
    gyro_z_df = pd.concat([one_slot['timestamp'], one_slot['gyro_z'], one_slot['i_slot']], axis=1)

    print(label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]])
    # sns.lineplot(x="timestamp", y="gyro_x", hue="i_slot", legend=False, data=gyro_x_df2)

    target = label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]]

    # build plots
    # TODO set Title with Names of Slots (with dict)

    fig, ax = plt.subplots(13, 1)
    fig.subplots_adjust(hspace=0.6)

    dist_1_ax = sns.lineplot(x="timestamp", y='distance_tire1', hue="i_slot", legend=False, data=dist_1_df,
                             ax=ax[0]).set_title("{} - dist_tire_links".format(target))
    dist_2_ax = sns.lineplot(x="timestamp", y='distance_tire2', hue="i_slot", legend=False, data=dist_2_df,
                             ax=ax[1]).set_title("{} - dist_tire_rechts".format(target))
    dist_3_ax = sns.lineplot(x="timestamp", y='distance_field1', hue="i_slot", legend=False, data=dist_3_df,
                             ax=ax[2]).set_title("{} - dist_schiene".format(target))
    dist_4_ax = sns.lineplot(x="timestamp", y='distance_field2', hue="i_slot", legend=False, data=dist_4_df,
                             ax=ax[3]).set_title("{} - dist_hebel".format(target))
    pos_x_ax = sns.lineplot(x="timestamp", y='pos_x', hue="i_slot", legend=False, data=pos_x_df, ax=ax[4]).set_title(
        "{} - pos_x".format(target))
    pos_y_ax = sns.lineplot(x="timestamp", y='pos_y', hue="i_slot", legend=False, data=pos_y_df, ax=ax[5]).set_title(
        "{} - pos_y".format(target))
    pos_z_ax = sns.lineplot(x="timestamp", y='pos_z', hue="i_slot", legend=False, data=pos_z_df, ax=ax[6]).set_title(
        "{} - pos_z".format(target))
    acc_x_ax = sns.lineplot(x="timestamp", y='acc_x', hue="i_slot", legend=False, data=acc_x_df, ax=ax[7]).set_title(
        "{} - acc_x".format(target))
    acc_y_ax = sns.lineplot(x="timestamp", y='acc_y', hue="i_slot", legend=False, data=acc_y_df, ax=ax[8]).set_title(
        "{} - acc_y".format(target))
    acc_z_ax = sns.lineplot(x="timestamp", y='acc_z', hue="i_slot", legend=False, data=acc_z_df, ax=ax[9]).set_title(
        "{} - acc_z".format(target))
    gyro_x_ax = sns.lineplot(x="timestamp", y='gyro_x', hue="i_slot", legend=False, data=gyro_x_df,
                             ax=ax[10]).set_title("{} - gyro_x".format(target))
    gyro_y_ax = sns.lineplot(x="timestamp", y='gyro_y', hue="i_slot", legend=False, data=gyro_y_df,
                             ax=ax[11]).set_title("{} - gyro_y".format(target))
    gyro_z_ax = sns.lineplot(x="timestamp", y='gyro_z', hue="i_slot", legend=False, data=gyro_z_df,
                             ax=ax[12]).set_title("{} - gyro_z".format(target))

    dist_1_ax.figure.set_size_inches(18.5, 50.5)

    name = str(list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)])
    name = 'plots/' + name

    # bbox_inches = "tight" needed to not cut off xlabels
    plt.savefig(name + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(name + ".png", format="png", bbox_inches="tight")
    plt.savefig(name + ".pdf", format="pdf", bbox_inches="tight")


def plot_one_slot_dist(subjects_list, all_slots, one_slot_id, with_date=True):
    one_slot = all_slots[one_slot_id]
    one_slot = one_slot.reset_index()

    schriftgroesse = 13
    sns.set_context("paper", rc={"font.size": schriftgroesse, "axes.titlesize": schriftgroesse,
                                 "axes.labelsize": schriftgroesse})

    dist_1_df = pd.concat([one_slot['timestamp'], one_slot['distance_tire1'], one_slot['i_slot']], axis=1)
    dist_2_df = pd.concat([one_slot['timestamp'], one_slot['distance_tire2'], one_slot['i_slot']], axis=1)
    dist_3_df = pd.concat([one_slot['timestamp'], one_slot['distance_field1'], one_slot['i_slot']], axis=1)
    dist_4_df = pd.concat([one_slot['timestamp'], one_slot['distance_field2'], one_slot['i_slot']], axis=1)

    print(label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]])
    # sns.lineplot(x="timestamp", y="gyro_x", hue="i_slot", legend=False, data=gyro_x_df2)

    target = label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]]

    # build plots
    # TODO set Title with Names of Slots (with dict)

    fig, ax = plt.subplots(4, 1)
    fig.subplots_adjust(hspace=0.6)

    dist_1_ax = sns.lineplot(x="timestamp", y='distance_tire1', hue="i_slot", legend=False, data=dist_1_df,
                             ax=ax[0]).set_title("{} - dist_tire_links".format(target))
    dist_2_ax = sns.lineplot(x="timestamp", y='distance_tire2', hue="i_slot", legend=False, data=dist_2_df,
                             ax=ax[1]).set_title("{} - dist_tire_rechts".format(target))
    dist_3_ax = sns.lineplot(x="timestamp", y='distance_field1', hue="i_slot", legend=False, data=dist_3_df,
                             ax=ax[2]).set_title("{} - dist_schiene".format(target))
    dist_4_ax = sns.lineplot(x="timestamp", y='distance_field2', hue="i_slot", legend=False, data=dist_4_df,
                             ax=ax[3]).set_title("{} - dist_hebel".format(target))


    dist_1_ax.figure.set_size_inches(18.5, 10.5)

    name = str(list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)])
    name = 'plots/' + subjects_list + '_' + name
    name = name + '_dist'
    if with_date:
        name = name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # add time

    # bbox_inches = "tight" needed to not cut off xlabels
    plt.savefig(name + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(name + ".png", format="png", bbox_inches="tight")
    plt.savefig(name + ".pdf", format="pdf", bbox_inches="tight")


def plot_one_slot_dist_tire(subjects_list, all_slots, one_slot_id, with_date=True):
    one_slot = all_slots[one_slot_id]
    one_slot = one_slot.reset_index()

    schriftgroesse = 13
    sns.set_context("paper", rc={"font.size": schriftgroesse, "axes.titlesize": schriftgroesse,
                                 "axes.labelsize": schriftgroesse})

    dist_1_df = pd.concat([one_slot['timestamp'], one_slot['distance_tire1'], one_slot['i_slot']], axis=1)
    dist_2_df = pd.concat([one_slot['timestamp'], one_slot['distance_tire2'], one_slot['i_slot']], axis=1)

    print(label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]])
    # sns.lineplot(x="timestamp", y="gyro_x", hue="i_slot", legend=False, data=gyro_x_df2)

    target = label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]]

    # build plots
    # TODO set Title with Names of Slots (with dict)

    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.6)

    dist_1_ax = sns.lineplot(x="timestamp", y='distance_tire1', hue="i_slot", legend=False, data=dist_1_df,
                             ax=ax[0]).set_title("{} - dist_tire_links".format(target))
    dist_2_ax = sns.lineplot(x="timestamp", y='distance_tire2', hue="i_slot", legend=False, data=dist_2_df,
                             ax=ax[1]).set_title("{} - dist_tire_rechts".format(target))

    dist_1_ax.figure.set_size_inches(18.5, 5.25)

    name = str(list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)])
    name = 'plots/' + subjects_list + '_' + name
    name = name + '_dist_tire'
    if with_date:
        name = name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # add time

    # bbox_inches = "tight" needed to not cut off xlabels
    plt.savefig(name + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(name + ".png", format="png", bbox_inches="tight")
    plt.savefig(name + ".pdf", format="pdf", bbox_inches="tight")


def plot_one_slot_dist_box(subjects_list, all_slots, one_slot_id, with_date=True):
    one_slot = all_slots[one_slot_id]
    one_slot = one_slot.reset_index()

    schriftgroesse = 13
    sns.set_context("paper", rc={"font.size": schriftgroesse, "axes.titlesize": schriftgroesse,
                                 "axes.labelsize": schriftgroesse})

    dist_3_df = pd.concat([one_slot['timestamp'], one_slot['distance_field1'], one_slot['i_slot']], axis=1)
    dist_4_df = pd.concat([one_slot['timestamp'], one_slot['distance_field2'], one_slot['i_slot']], axis=1)

    print(label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]])
    # sns.lineplot(x="timestamp", y="gyro_x", hue="i_slot", legend=False, data=gyro_x_df2)

    target = label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]]

    # build plots
    # TODO set Title with Names of Slots (with dict)

    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.6)

    dist_3_ax = sns.lineplot(x="timestamp", y='distance_field1', hue="i_slot", legend=False, data=dist_3_df,
                             ax=ax[0]).set_title("{} - dist_schiene".format(target))
    dist_4_ax = sns.lineplot(x="timestamp", y='distance_field2', hue="i_slot", legend=False, data=dist_4_df,
                             ax=ax[1]).set_title("{} - dist_hebel".format(target))

    dist_3_ax.figure.set_size_inches(18.5, 5.25)

    name = str(list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)])
    name = 'plots/' + subjects_list + '_' + name
    name = name + '_dist_box'
    if with_date:
        name = name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # add time

    # bbox_inches = "tight" needed to not cut off xlabels
    plt.savefig(name + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(name + ".png", format="png", bbox_inches="tight")
    plt.savefig(name + ".pdf", format="pdf", bbox_inches="tight")


def plot_one_slot_pos(subjects_list, all_slots, one_slot_id, with_date=True):
    one_slot = all_slots[one_slot_id]
    one_slot = one_slot.reset_index()

    schriftgroesse = 13
    sns.set_context("paper", rc={"font.size": schriftgroesse, "axes.titlesize": schriftgroesse,
                                 "axes.labelsize": schriftgroesse})

    pos_1_df = pd.concat([one_slot['timestamp'], one_slot['pos_x'], one_slot['i_slot']], axis=1)
    pos_2_df = pd.concat([one_slot['timestamp'], one_slot['pos_y'], one_slot['i_slot']], axis=1)
    pos_3_df = pd.concat([one_slot['timestamp'], one_slot['pos_z'], one_slot['i_slot']], axis=1)

    print(label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]])
    # sns.lineplot(x="timestamp", y="gyro_x", hue="i_slot", legend=False, data=gyro_x_df2)

    target = label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]]

    # build plots
    # TODO set Title with Names of Slots (with dict)

    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.6)

    pos_x_ax = sns.lineplot(x="timestamp", y='pos_x', hue="i_slot", legend=False, data=pos_1_df,
                             ax=ax[0]).set_title("{} - pos_x".format(target))
    pos_y_ax = sns.lineplot(x="timestamp", y='pos_y', hue="i_slot", legend=False, data=pos_2_df,
                             ax=ax[1]).set_title("{} - pos_y".format(target))
    pos_z_ax = sns.lineplot(x="timestamp", y='pos_z', hue="i_slot", legend=False, data=pos_3_df,
                             ax=ax[2]).set_title("{} - pos_z".format(target))

    pos_x_ax.figure.set_size_inches(18.5, 7.5)

    name = str(list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)])
    name = 'plots/' + subjects_list + '_' + name
    name = name + '_pos'
    if with_date:
        name = name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # add time

    # bbox_inches = "tight" needed to not cut off xlabels
    plt.savefig(name + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(name + ".png", format="png", bbox_inches="tight")
    plt.savefig(name + ".pdf", format="pdf", bbox_inches="tight")


def plot_one_slot_acc(subjects_list, all_slots, one_slot_id, with_date=True):
    one_slot = all_slots[one_slot_id]
    one_slot = one_slot.reset_index()

    acc_1_df = pd.concat([one_slot['timestamp'], one_slot['acc_x'], one_slot['i_slot']], axis=1)
    acc_2_df = pd.concat([one_slot['timestamp'], one_slot['acc_y'], one_slot['i_slot']], axis=1)
    acc_3_df = pd.concat([one_slot['timestamp'], one_slot['acc_z'], one_slot['i_slot']], axis=1)

    print(label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]])
    # sns.lineplot(x="timestamp", y="gyro_x", hue="i_slot", legend=False, data=gyro_x_df2)

    target = label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]]

    # build plots
    # TODO set Title with Names of Slots (with dict)

    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.6)

    acc_x_ax = sns.lineplot(x="timestamp", y='acc_x', hue="i_slot", legend=False, data=acc_1_df,
                             ax=ax[0]).set_title("{} - acc_x".format(target))
    acc_y_ax = sns.lineplot(x="timestamp", y='acc_y', hue="i_slot", legend=False, data=acc_2_df,
                             ax=ax[1]).set_title("{} - acc_y".format(target))
    acc_z_ax = sns.lineplot(x="timestamp", y='acc_z', hue="i_slot", legend=False, data=acc_3_df,
                             ax=ax[2]).set_title("{} - acc_z".format(target))

    acc_x_ax.figure.set_size_inches(18.5, 7.5)

    name = str(list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)])
    name = 'plots/' + subjects_list + '_' + name
    name = name + '_acc'
    if with_date:
        name = name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # add time

    # bbox_inches = "tight" needed to not cut off xlabels
    plt.savefig(name + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(name + ".png", format="png", bbox_inches="tight")
    plt.savefig(name + ".pdf", format="pdf", bbox_inches="tight")


def plot_one_slot_gyro(subjects_list, all_slots, one_slot_id, with_date=True):
    one_slot = all_slots[one_slot_id]
    one_slot = one_slot.reset_index()

    gyro_1_df = pd.concat([one_slot['timestamp'], one_slot['gyro_x'], one_slot['i_slot']], axis=1)
    gyro_2_df = pd.concat([one_slot['timestamp'], one_slot['gyro_y'], one_slot['i_slot']], axis=1)
    gyro_3_df = pd.concat([one_slot['timestamp'], one_slot['gyro_z'], one_slot['i_slot']], axis=1)

    print(label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]])
    # sns.lineplot(x="timestamp", y="gyro_x", hue="i_slot", legend=False, data=gyro_x_df2)

    target = label_dict[list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)]]

    # build plots
    # TODO set Title with Names of Slots (with dict)

    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.6)

    gyro_x_ax = sns.lineplot(x="timestamp", y='gyro_x', hue="i_slot", legend=False, data=gyro_1_df,
                             ax=ax[0]).set_title("{} - gyro_x".format(target))
    gyro_y_ax = sns.lineplot(x="timestamp", y='gyro_y', hue="i_slot", legend=False, data=gyro_2_df,
                             ax=ax[1]).set_title("{} - gyro_y".format(target))
    gyro_z_ax = sns.lineplot(x="timestamp", y='gyro_z', hue="i_slot", legend=False, data=gyro_3_df,
                             ax=ax[2]).set_title("{} - gyro_z".format(target))

    gyro_x_ax.figure.set_size_inches(18.5, 7.5)

    name = str(list(label_to_index_dict.keys())[list(label_to_index_dict.values()).index(one_slot_id)])
    name = 'plots/' + subjects_list + '_' + name
    name = name + '_gyro'
    if with_date:
        name = name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # add time

    # bbox_inches = "tight" needed to not cut off xlabels
    plt.savefig(name + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(name + ".png", format="png", bbox_inches="tight")
    plt.savefig(name + ".pdf", format="pdf", bbox_inches="tight")


def plot_precisions_old(report, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues, name=None, acc=0, debug=False):

    # TODO alles runden, weil precisions von einzelnen Klassen ohne Nachkommastelle? aber gesamt Accuracy mit?

    if name is None:
        name = 'plots/precisions_1234'

    lines = report.split('\n')

    classes = []
    plotMat = []

    precisions = []

    for line in lines[2: (len(lines) - 5)]:
        # print(line)
        t = line.split()
        # print('t = {}'.format(t))
        classes.append(t[0])  # t[0] = Class names
        v = [float(x) for x in t[1: len(t) - 1]]
        # print('v = {}'.format(v))
        plotMat.append(v)  # rest of t = (precision, recall, f1-score, support) values
        # precisions.append(t[1])

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)

    x = False # kann weg? !
    if x:
        plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        x_tick_marks = np.arange(3)
        y_tick_marks = np.arange(len(classes))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('Classes')
        plt.xlabel('Measures')

    if debug: print(plotMat)
    precisions = [line[0] for line in plotMat]
    precisions = [elem * 100 for elem in precisions]
    if debug: print(precisions)

    classes = [label_dict[int(item)] for item in classes]

    classes.append('total')
    # acc = 50
    precisions.append(acc)

    prec_plot = sns.barplot(x=classes, y=precisions, order=classes, palette="Blues_d")
    prec_plot.figure.set_size_inches(15, 2)
    # ax1.axhline(0, color="k", clip_on=False)
    # prec_plot.set_xlabel("Bewegungen")
    prec_plot.set_ylabel("Akkuratheit")
    # TODO - ax.set_xticklabels(label_text['labels'], rotation='vertical', fontsize=10)
    # prec_plot.xticks(rotation=30)
    # prec_plot.setp(labels, rotation=45)

    for index, row in enumerate(precisions):
        if debug: print('index = {}, class = {}, row = {}'.format(index, classes[index], row))
        prec_plot.text(index, row, row, color='black', ha="center")

    # Finalize the plot
    sns.despine()  # remove the top and right spines from plot
    # plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=4)  # limit height

    # bbox_inches = "tight" needed to not cut off xlabels
    plt.savefig(name + '_precisions' + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(name + '_precisions' + ".png", format="png", bbox_inches="tight")
    plt.savefig(name + '_precisions' + ".pdf", format="pdf", bbox_inches="tight")


def plot_precisions_old3(report, schriftgroesse=10, name=None, acc=0, custom_color=False, custom_colors=False, debug=False):
    lines = report.split('\n')

    classes = []
    plotMat = []

    # get precisions out of classification report
    for line in lines[2: (len(lines) - 5)]:
        # print(line)
        t = line.split()
        # print('t = {}'.format(t))
        classes.append(t[0])  # t[0] = Class names
        v = [float(x) for x in t[1: len(t) - 1]]
        # print('v = {}'.format(v))
        plotMat.append(v)  # rest of t = (precision, recall, f1-score, support) values
        # precisions.append(t[1])

    if debug: print(plotMat)
    precisions = [elem[0] for elem in plotMat]
    precisions = [elem * 100 for elem in precisions]
    precisions = [round(elem, 2) for elem in precisions]
    if debug: print(precisions)

    if custom_colors: # if custom_colors is True, get colors from dict
        colors = [color_dict[int(item)] for item in classes]
        colors.append('darkgray')  # append total accuracy to precision results

    classes = [label_dict[int(item)] for item in classes]

    classes.append('gesamt')  # add total accuracy as last bar
    precisions.append(round(acc, 2))

    if not custom_colors:
        colors = "Blues_d"  # if no custom_colors, take built-in palette

    if custom_color: plt = sns.barplot(x=classes, y=precisions, order=classes, color=custom_color)
    else: plt = sns.barplot(x=classes, y=precisions, order=classes, palette=colors)
    plt.set_xticklabels(plt.get_xticklabels(), rotation=45, ha="right", size=schriftgroesse)
    plt.figure.set_size_inches(15, 2)
    plt.set_ylabel("Akkuratheit")
    for index, row in enumerate(precisions):
        if debug: print('index = {}, class = {}, row = {}'.format(index, classes[index], row))
        plt.text(index, row, row, color='black', ha="center")  # get them to be aligned

    # make line with average accuracy
    plt.axhline(y=round((sum(precisions[:-1]) / len(precisions[:-1])), 2), linewidth=0.5, color='k')

    # Finalize the plot
    sns.despine()  # remove the top and right spines from plot

    fig = plt.get_figure()
    # bbox_inches = "tight" needed to not cut off xlabels
    fig.savefig(name + '_precisions' + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(name + '_precisions' + ".png", format="png", bbox_inches="tight")
    fig.savefig(name + '_precisions' + ".pdf", format="pdf", bbox_inches="tight")


def plot_precisions(report, schriftgroesse=10, name=None, acc=0, custom_color=False, custom_colors=["#64b75f", "#a8d7a6"], debug=False):
    lines = report.split('\n')

    classes = []
    plotMat = []

    # get precisions out of classification report
    for line in lines[2: (len(lines) - 5)]:
        # print(line)
        t = line.split()
        # print('t = {}'.format(t))
        classes.append(t[0])  # t[0] = Class names
        v = [float(x) for x in t[1: len(t) - 1]]
        # print('v = {}'.format(v))
        plotMat.append(v)  # rest of t = (precision, recall, f1-score, support) values
        # precisions.append(t[1])

    if debug: print(plotMat)
    precisions = [elem[0] for elem in plotMat]
    precisions = [elem * 100 for elem in precisions]
    precisions = [round(elem, 2) for elem in precisions]
    if debug: print(precisions)
    recall = [elem[1] for elem in plotMat]
    recall = [elem * 100 for elem in recall]
    recall = [round(elem, 2) for elem in recall]
    if debug: print(recall)

    old_classes = classes
    classes = [label_dict[int(item)] for item in classes]

    raw_df = {'Bewegung': classes,
              'Precisions': precisions,
              'Recall': recall}
    df = pd.DataFrame(raw_df, columns=['Bewegung', 'Precisions', 'Recall'])
    # precisions
    df_melt = pd.melt(df, id_vars=['Bewegung'], value_vars=['Precisions', 'Recall'])

    # Draw a nested barplot to show survival for class and sex
    plt = sns.catplot(x="Bewegung", y="value", hue="variable", data=df_melt,
                      height=3, aspect=3.5, kind="bar", palette=custom_colors)

    plt.set_xticklabels(classes, rotation=45, ha='right', size=13)  # rename and position ticklabels

    # plt.set_size_inches(15, 2)
    plt._legend.set_title(' ')
    plt.despine(left=True)
    plt.set_ylabels("Erkennungsrate")
    # for index, row in enumerate(precisions):
    #     if debug: print('index = {}, class = {}, row = {}'.format(index, classes[index], row))
    #     plt.text(index, row, row, color='black', ha="center")  # get them to be aligned

    # make line with average accuracy
    # plt.axhline(y=round((sum(precisions[:-1]) / len(precisions[:-1])), 2), linewidth=0.5, color='k')

    plt.savefig(name + '_precisions' + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(name + '_precisions' + ".png", format="png", bbox_inches="tight")
    plt.savefig(name + '_precisions' + ".pdf", format="pdf", bbox_inches="tight")


# ohne schriftgroesse als variable TODO variable in einer Funktion machen mit None und checken
def plot_precisions_old(report, name=None, acc=0, custom_colors=False, debug=False):
    lines = report.split('\n')

    classes = []
    plotMat = []

    # get precisions out of classification report
    for line in lines[2: (len(lines) - 5)]:
        # print(line)
        t = line.split()
        # print('t = {}'.format(t))
        classes.append(t[0])  # t[0] = Class names
        v = [float(x) for x in t[1: len(t) - 1]]
        # print('v = {}'.format(v))
        plotMat.append(v)  # rest of t = (precision, recall, f1-score, support) values
        # precisions.append(t[1])

    if debug: print(plotMat)
    precisions = [elem[0] for elem in plotMat]
    precisions = [elem * 100 for elem in precisions]
    precisions = [round(elem, 2) for elem in precisions]
    if debug: print(precisions)

    if custom_colors: # if custom_colors is True, get colors from dict
        colors = [color_dict[int(item)] for item in classes]
        colors.append('darkgray')  # append total accuracy to precision results
    classes = [label_dict[int(item)] for item in classes]

    classes.append('gesamt')  # add total accuracy as last bar
    precisions.append(acc)

    if not custom_colors:
        colors = "Blues_d"  # if no custom_colors, take built-in palette

    plt = sns.barplot(x=classes, y=precisions, order=classes, palette=colors)
    plt.set_xticklabels(plt.get_xticklabels(), rotation=45, ha="right")
    plt.figure.set_size_inches(15, 2)
    plt.set_ylabel("Akkuratheit")
    for index, row in enumerate(precisions):
        if debug: print('index = {}, class = {}, row = {}'.format(index, classes[index], row))
        plt.text(index, row, row, color='black', ha="center")

    # Finalize the plot
    sns.despine()  # remove the top and right spines from plot

    fig = plt.get_figure()
    # bbox_inches = "tight" needed to not cut off xlabels
    fig.savefig(name + '_precisions' + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(name + '_precisions' + ".png", format="png", bbox_inches="tight")
    fig.savefig(name + '_precisions' + ".pdf", format="pdf", bbox_inches="tight")

