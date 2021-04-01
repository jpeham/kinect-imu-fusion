import numpy as np
import pandas as pd
import scipy
import sklearn
import time
import datetime

from metamux.dataset.dataset import Dataset
from metamux.classification.training_data import TrainingDataLoader
from metamux.classification import defaults, training_data_filter
from metamux.classification.features import peak_detection, smoothing, util, statistics, channel_select, differential
from metamux.classification.plot_cm import plot_cm
from metamux.classification.features import feature_selection
from metamux.classification import segment_feature_union

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from metamux.classification.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, cross_validate, LeaveOneOut, LeaveOneGroupOut, GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer, StandardScaler

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

#from IPython.display import display

import Decoder # Preprocessor from JP

# to get dir_paths from final recordings
from os import listdir
from os.path import isfile, join

from plot_cm import plot_cm # Confusion Matrix from JH

from tools import make_name, save_infos

from rename_labels import label_dict # dict with names for labels (Schrauben, Stift rein,...)
from rename_labels import get_opposites_out, merge_variants_screw_lr, merge_variants_screw_abc
import recordings_paths # for filling DataLoader
from plot_data import plot_slot, make_cm_all, make_cm_tt # to plot line-plots and cm_plot from JH
from plot_data import plot_precisions, plot_precisions_old
from data_prep import prep_data, select_sensor_channel


# TODO parameter nicht einzeln übergeben sondern in irgendeiner art von Objekt?
def do_it(subjects, path_list=None, sensors_list=None,
          fuse_slots=None, slot_filling=False,
          merge_opposites=False, merge_screw_lr=False, merge_screw_abc=False,
          name=None, with_date=True,
          show_time=False, show_dots=False, debug=False, deep_debug=False):


    ###################################################
    #                                                 #
    #                   PIPELINE                      #
    #                                                 #
    ###################################################

    def np_flatten(ar):
        return ar.flatten()

    # TODO füllen von Pipeline dynamisch machen? bools übergeben und dann entsprechend dazunehmen?

    # ACC + GYRO
    pipe_all = Pipeline([
        # ('ch_imu', channel_select.SegmentChannelSelectionTransformer(range(0, 6))),
        # von JH für Selektion zwischen IMU und Myo
        ('differential', segment_feature_union.SegmentFeatureUnion([
            ('identity', util.SegmentFunctionTransformer()),
            ('diff', differential.SegmentDiffTransformer()),
            ('int1', differential.SegmentIntTransformer()),
            # ('int2', differential.SegmentInt2Transformer()),
        ])),
        ('features', FeatureUnion([
            ('min', util.SegmentFunctionTransformer(np.amin, {'axis': 0})),
            ('max', util.SegmentFunctionTransformer(np.amax, {'axis': 0})),
            ('mean', util.SegmentFunctionTransformer(np.mean, {'axis': 0})),
            # ('med', util.SegmentFunctionTransformer(np.median, {'axis': 0})),
            # ('rms', statistics.SegmentRMSTransformer()),
            # ('std', util.SegmentFunctionTransformer(np.std, {'axis': 0})),
            # ('var', util.SegmentFunctionTransformer(np.var, {'axis': 0})),
            # ('q25', statistics.SegmentQuantilesTransformer(percentiles=[25])),
            # ('q75', statistics.SegmentQuantilesTransformer(percentiles=[75])),
            # ('zcc', statistics.SegmentZCCTransformer(offset=500)),
            # ('skew', util.SegmentFunctionTransformer(scipy.stats.skew)),
            # ('kurt', util.SegmentFunctionTransformer(scipy.stats.kurtosis)),
        ]))
    ])

    # TODO Klassifikator auch ausgeben!! ------------------------------------------------------------

    diff_features = pipe_all.steps[0][1].transformer_list
    diff_features_list = [diff_feature[0] for diff_feature in diff_features]

    stat_features = pipe_all.steps[1][1].transformer_list
    stat_features_list = [stat_feature[0] for stat_feature in stat_features]

    pipe = Pipeline([
        ('features', FeatureUnion([
            ('pipe_all', pipe_all)
            # ('pipe_imu', pipe_imu),
            # ('pipe_fsr', pipe_fsr),
        ])),
        ('flattening', util.SegmentFunctionTransformer(np_flatten)),
        # ('standardscaler', StandardScaler()), # Either LDA or StandardScaler is required
        # ('normalizer', Normablizer()), # Bad idea TODO why? (NormaBBBlizer? oder Normalizer? ist doch Tippfehler!?
        # ('pca', PCA()), #n_components=15)),
        # ('te', TestEstimator()),
        ('lda', LinearDiscriminantAnalysis()),
        # ('lda', LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')),# JH special lda
        # ('svm', SVC(C=1000, gamma=0.001, probability=True, kernel='rbf')),
        # ('rf', RandomForestClassifier(n_estimators=100, max_features='auto')),
    ])

    ###################################################
    #                                                 #
    #                  PREP DATA                      #
    #                                                 #
    ###################################################

    prep_start_time = datetime.datetime.now()

    if debug: print('got name {}'.format(name))

    if name is None:
        name = make_name(subjects=subjects, sensors_list=sensors_list,
                         diff_features_list=diff_features_list, stat_features_list=stat_features_list,
                         fuse_slots=fuse_slots,
                         merge_opposites=merge_opposites,
                         merge_screw_lr=merge_screw_lr, merge_screw_abc=merge_screw_abc,
                         with_date=with_date,
                         show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)
    if debug: print('name = {}'.format(name))

    loader = recordings_paths.fill_loader(debug=debug)

    if debug:
        print('subjects = {}'.format(subjects))

    if subjects == 'all':
        subjects = 'JP10,MM,AR,MJ,CA,CD,JH10'
    elif subjects == 'all10':
        subjects = 'JP10,MM,AR,MJ,CD,JH10'
    subjects = subjects.split(',')  # making list with subjects names

    if debug:
        print('subjects = {}'.format(subjects))

    path_list = loader.get_data(subjects, debug=debug, deep_debug=deep_debug)  # get paths for subjects

    ###################################################
    #                                                 #
    #                   GET DATA                      #
    #                                                 #
    ###################################################

    # data und targets für subject und sensors_list holen
    sessions, targets, concat_data, concat_targets, worked_with = prep_data(path_list=path_list, sensors_list=None, fuse_slots=fuse_slots, slot_filling=slot_filling, show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)

    # gewünsche Sensorkanäle aus sessions holen
    sessions = select_sensor_channel(sessions, sensors_list, show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)

    # data in eine liste
    all_data = [slots.values for session in sessions for slots in session]

    # get labels out
    if not merge_opposites:
        # concat_targets_noopp = [get_opposites_out[l] for l in concat_targets]
        concat_targets = [get_opposites_out[l] for l in concat_targets]
        if debug: print('get opposites out')
    if merge_screw_lr:
        #concat_targets_abc = [merge_variants_screw_lr[l] for l in concat_targets]
        concat_targets = [merge_variants_screw_lr[l] for l in concat_targets]
        if debug: print('merge lr screwing')
    if merge_screw_abc:
        #concat_targets_lr = [merge_variants_screw_abc[l] for l in concat_targets]
        concat_targets = [merge_variants_screw_abc[l] for l in concat_targets] # TODO ------------------------------------------------------------------ auch für do_it_ud!!!!!!
        if debug: print('merge abc screwing')

    prep_end_time = datetime.datetime.now()

    prep_end_time = datetime.datetime.now()
    pred_start_time = datetime.datetime.now()

    ###################################################
    #                                                 #
    #                CLASSIFICATION                   #
    #                                                 #
    ###################################################

    # train and test
    predicted = cross_val_predict(pipe, all_data, concat_targets, cv=LeaveOneOut())

    pred_end_time = datetime.datetime.now()
    draw_start_time = datetime.datetime.now()

    # num_sessions = len(path_list) # TODO make auto toggle!!!
    num_sessions = worked_with

    ###################################################
    #                                                 #
    #                 SHOW RESULTS                    #
    #                                                 #
    ###################################################

    mode = ''
    if not merge_opposites:
        mode = mode + ' no_opp'
    if merge_screw_lr:
        mode = mode + ' abc'
    if merge_screw_abc:
        mode = mode + '  lr'
    if fuse_slots is not None:
        mode = mode + ' ' + str(fuse_slots[2])

    # schriftgroesse = 13
    # import seaborn as sns
    # sns.set_context("paper", rc={"font.size": schriftgroesse, "axes.titlesize": schriftgroesse, "axes.labelsize": schriftgroesse})

    # plot cm
    make_cm_all(sensors_list, diff_features_list, stat_features_list, worked_with, subjects, concat_targets, predicted,
                label_dict, mode, name)

    report = sklearn.metrics.classification_report(concat_targets, predicted)
    acc = accuracy_score(concat_targets, predicted)*100
    acc = round(acc, 2)
    # print('---------- acc = {}'.format(acc))

    # plot_precisions(report, schriftgroesse, name=name, acc=acc)
    plot_precisions_old(report, name=name, acc=acc)

    save_infos(sensors_list=sensors_list, diff_features_list=diff_features_list, stat_features_list=stat_features_list,
               path_list=worked_with, subjects=subjects,
               targets=concat_targets, predicted=predicted,
               rename=label_dict, mode=mode, name=name)

    draw_end_time = datetime.datetime.now()

    print('preparation duration = {}'.format(prep_end_time - prep_start_time))
    print('prediciton duration  = {}'.format(pred_end_time - pred_start_time))
    print('drawing duration     = {}'.format(draw_end_time - draw_start_time))


def do_it_sep_pipe(subjects, pipe,
                   path_list=None, sensors_list=None,
                   fuse_slots=None, slot_filling=False,
                   merge_opposites=False, merge_screw_lr=False, merge_screw_abc=False,
                   name=None, with_date=True,
                   show_time=False, show_dots=False, debug=False, deep_debug=False):

    ###################################################
    #                  PIPE

    pipe_all = pipe.steps[0][1].transformer_list[0][1]

    diff_features = pipe_all.steps[0][1].transformer_list
    diff_features_list = [diff_feature[0] for diff_feature in diff_features]

    stat_features = pipe_all.steps[1][1].transformer_list
    stat_features_list = [stat_feature[0] for stat_feature in stat_features]

    ###################################################
    #                                                 #
    #                  PREP DATA                      #
    #                                                 #
    ###################################################

    prep_start_time = datetime.datetime.now()

    if debug: print('got name {}'.format(name))

    if name is None:
        name = make_name(subjects=subjects, sensors_list=sensors_list,
                         diff_features_list=diff_features_list, stat_features_list=stat_features_list,
                         fuse_slots=fuse_slots,
                         merge_opposites=merge_opposites,
                         merge_screw_lr=merge_screw_lr, merge_screw_abc=merge_screw_abc,
                         with_date=with_date,
                         show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)
    if debug: print('name = {}'.format(name))

    loader = recordings_paths.fill_loader(debug=debug)

    if debug:
        print('subjects = {}'.format(subjects))

    if subjects == 'all':
        subjects = 'JP10,MM,AR,MJ,CA,CD,JH10'
    elif subjects == 'all10':
        subjects = 'JP10,MM,AR,MJ,CD,JH10'
    subjects = subjects.split(',')  # making list with subjects names

    if debug:
        print('subjects = {}'.format(subjects))

    path_list = loader.get_data(subjects, debug=debug, deep_debug=deep_debug)  # get paths for subjects

    ###################################################
    #                                                 #
    #                   GET DATA                      #
    #                                                 #
    ###################################################

    # data und targets für subject und sensors_list holen
    sessions, targets, concat_data, concat_targets, worked_with = prep_data(path_list=path_list, sensors_list=None, fuse_slots=fuse_slots, slot_filling=slot_filling, show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)

    # gewünsche Sensorkanäle aus sessions holen
    sessions = select_sensor_channel(sessions, sensors_list, show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)

    # data in eine liste
    all_data = [slots.values for session in sessions for slots in session]

    # get labels out
    if not merge_opposites:
        # concat_targets_noopp = [get_opposites_out[l] for l in concat_targets]
        concat_targets = [get_opposites_out[l] for l in concat_targets]
        if debug: print('get opposites out')
    if merge_screw_lr:
        #concat_targets_abc = [merge_variants_screw_lr[l] for l in concat_targets]
        concat_targets = [merge_variants_screw_lr[l] for l in concat_targets]
        if debug: print('merge lr screwing')
    if merge_screw_abc:
        #concat_targets_lr = [merge_variants_screw_abc[l] for l in concat_targets]
        concat_targets = [merge_variants_screw_abc[l] for l in concat_targets] # TODO ------------------------------------------------------------------ auch für do_it_ud!!!!!!
        if debug: print('merge abc screwing')

    prep_end_time = datetime.datetime.now()

    prep_end_time = datetime.datetime.now()
    pred_start_time = datetime.datetime.now()

    ###################################################
    #                                                 #
    #                CLASSIFICATION                   #
    #                                                 #
    ###################################################

    # train and test
    predicted = cross_val_predict(pipe, all_data, concat_targets, cv=LeaveOneOut())

    pred_end_time = datetime.datetime.now()
    draw_start_time = datetime.datetime.now()

    # num_sessions = len(path_list) # TODO make auto toggle!!!
    num_sessions = worked_with

    ###################################################
    #                                                 #
    #                 SHOW RESULTS                    #
    #                                                 #
    ###################################################

    mode = ''
    if not merge_opposites:
        mode = mode + ' no_opp'
    if merge_screw_lr:
        mode = mode + ' abc'
    if merge_screw_abc:
        mode = mode + '  lr'
    if fuse_slots is not None:
        mode = mode + ' ' + str(fuse_slots[2])

    # schriftgroesse = 13
    # import seaborn as sns
    # sns.set_context("paper", rc={"font.size": schriftgroesse, "axes.titlesize": schriftgroesse, "axes.labelsize": schriftgroesse})

    # plot cm
    make_cm_all(sensors_list, diff_features_list, stat_features_list, worked_with, subjects, concat_targets, predicted,
                label_dict, mode, name)

    report = sklearn.metrics.classification_report(concat_targets, predicted)
    acc = accuracy_score(concat_targets, predicted)*100
    acc = round(acc, 2)
    # print('---------- acc = {}'.format(acc))

    # plot_precisions(report, schriftgroesse, name=name, acc=acc)
    plot_precisions_old(report, name=name, acc=acc)

    save_infos(sensors_list=sensors_list, diff_features_list=diff_features_list, stat_features_list=stat_features_list,
               path_list=worked_with, subjects=subjects,
               targets=concat_targets, predicted=predicted,
               rename=label_dict, mode=mode, name=name)

    draw_end_time = datetime.datetime.now()

    print('preparation duration = {}'.format(prep_end_time - prep_start_time))
    print('prediciton duration  = {}'.format(pred_end_time - pred_start_time))
    print('drawing duration     = {}'.format(draw_end_time - draw_start_time))

# TODO parameter nicht einzeln übergeben sondern in irgendeiner art von Objekt? args[]
def do_it_pipe(pipe_type=0, subjects='all', path_list=None, sensors_list=None,
               fuse_slots=None, slot_filling=False,
               merge_opposites=False, merge_screw_lr=False, merge_screw_abc=False,
               name=None, with_date=True, custom_colors=False,
               show_time=False, show_dots=False, debug=False, deep_debug=False):

    ###################################################
    #                                                 #
    #                   PIPELINE                      #
    #                                                 #
    ###################################################

    pipe, diff_features_list, stat_features_list = get_pipe(pipe_type=pipe_type)

    ###################################################
    #                                                 #
    #                  PREP DATA                      #
    #                                                 #
    ###################################################

    prep_start_time = datetime.datetime.now()

    if debug: print('got name {}'.format(name))

    if name is None:
        name = make_name(subjects=subjects, sensors_list=sensors_list,
                         diff_features_list=diff_features_list, stat_features_list=stat_features_list,
                         fuse_slots=fuse_slots,
                         merge_opposites=merge_opposites,
                         merge_screw_lr=merge_screw_lr, merge_screw_abc=merge_screw_abc,
                         with_date=with_date,
                         show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)
    if debug: print('name = {}'.format(name))

    loader = recordings_paths.fill_loader(debug=debug)

    if subjects == 'all':
        subjects = 'JP10,MM,AR,MJ,CA,CD,JH10'
    elif subjects == 'all10':
        subjects = 'JP10,MM,AR,MJ,CD,JH10'
    subjects = subjects.split(',')  # making list with subjects names

    path_list = loader.get_data(subjects, debug=debug, deep_debug=deep_debug)  # get paths for subjects

    ###################################################
    #                                                 #
    #                   GET DATA                      #
    #                                                 #
    ###################################################

    # data und targets für subject und sensors_list holen
    sessions, targets, concat_data, concat_targets, worked_with = prep_data(path_list=path_list, sensors_list=None, fuse_slots=fuse_slots, slot_filling=slot_filling, show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)

    # gewünsche Sensorkanäle aus sessions holen
    sessions = select_sensor_channel(sessions, sensors_list, show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)

    # data in eine liste
    all_data = [slots.values for session in sessions for slots in session]

    # get labels out
    if not merge_opposites:
        # concat_targets_noopp = [get_opposites_out[l] for l in concat_targets]
        concat_targets = [get_opposites_out[l] for l in concat_targets]
        if debug: print('get opposites out')
    if merge_screw_lr:
        #concat_targets_abc = [merge_variants_screw_lr[l] for l in concat_targets]
        concat_targets = [merge_variants_screw_lr[l] for l in concat_targets]
        if debug: print('merge lr screwing')
    if merge_screw_abc:
        #concat_targets_lr = [merge_variants_screw_abc[l] for l in concat_targets]
        concat_targets = [merge_variants_screw_abc[l] for l in concat_targets] # TODO ------------------------------------------------------------------ auch für do_it_ud!!!!!!
        if debug: print('merge abc screwing')

    prep_end_time = datetime.datetime.now()

    prep_end_time = datetime.datetime.now()
    pred_start_time = datetime.datetime.now()

    ###################################################
    #                                                 #
    #                CLASSIFICATION                   #
    #                                                 #
    ###################################################

    # train and test
    predicted = cross_val_predict(pipe, all_data, concat_targets, cv=LeaveOneOut())

    pred_end_time = datetime.datetime.now()
    draw_start_time = datetime.datetime.now()

    # num_sessions = len(path_list) # TODO make auto toggle!!!
    num_sessions = worked_with

    ###################################################
    #                                                 #
    #                 SHOW RESULTS                    #
    #                                                 #
    ###################################################

    schriftgroesse = 13
    import seaborn as sns
    sns.set_context("paper", rc={"font.size": schriftgroesse, "axes.titlesize": schriftgroesse, "axes.labelsize": schriftgroesse})

    mode = ''
    if not merge_opposites:
        mode = mode + ' no_opp'
    if merge_screw_lr:
        mode = mode + ' abc'
    if merge_screw_abc:
        mode = mode + '  lr'
    if fuse_slots is not None:
        mode = mode + ' ' + str(fuse_slots[2])

    # plot cm
    make_cm_all(sensors_list, diff_features_list, stat_features_list, worked_with, subjects, concat_targets, predicted,
                label_dict, mode, name)

    report = sklearn.metrics.classification_report(concat_targets, predicted)
    acc = accuracy_score(concat_targets, predicted) * 100
    acc = round(acc, 2)
    # print('---------- acc = {}'.format(acc))

    plot_precisions(report, schriftgroesse, name=name, acc=acc, custom_colors=custom_colors)
    # plot_precisions_old(report, name=name, acc=acc, custom_colors=custom_colors)

    save_infos(sensors_list=sensors_list, diff_features_list=diff_features_list, stat_features_list=stat_features_list,
               path_list=worked_with, subjects=subjects,
               targets=concat_targets, predicted=predicted,
               rename=label_dict, mode=mode, name=name)

    draw_end_time = datetime.datetime.now()

    print('preparation duration = {}'.format(prep_end_time - prep_start_time))
    print('prediciton duration  = {}'.format(pred_end_time - pred_start_time))
    print('drawing duration     = {}'.format(draw_end_time - draw_start_time))


# train and test on different subjects
def do_it_ud(train_subjects, subjects, path_list=None, sensors_list=None,
          fuse_slots=None, slot_filling=False,
          merge_opposites=False, merge_screw_lr=False, merge_screw_abc=False,
          name=None, with_date=True,
          show_time=False, show_dots=False, debug=False, deep_debug=False):

    ###################################################
    #                                                 #
    #                   PIPELINE                      #
    #                                                 #
    ###################################################

    def np_flatten(ar):
        return ar.flatten()

    # ACC + GYRO
    pipe_all = Pipeline([
        # ('ch_imu', channel_select.SegmentChannelSelectionTransformer(range(0, 6))), # von JH für Selektion zwischen IMU und Myo
        ('differential', segment_feature_union.SegmentFeatureUnion([
            ('identity', util.SegmentFunctionTransformer()),
            ('diff', differential.SegmentDiffTransformer()),
            ('int1', differential.SegmentIntTransformer()),
            # ('int2', differential.SegmentInt2Transformer()),
        ])),
        ('features', FeatureUnion([
            ('min', util.SegmentFunctionTransformer(np.amin, {'axis': 0})),
            ('max', util.SegmentFunctionTransformer(np.amax, {'axis': 0})),
            ('mean', util.SegmentFunctionTransformer(np.mean, {'axis': 0})),
            #             ('med', util.SegmentFunctionTransformer(np.median, {'axis': 0})),
            #             ('rms', statistics.SegmentRMSTransformer()),
            # ('std', util.SegmentFunctionTransformer(np.std, {'axis': 0})),
            # ('var', util.SegmentFunctionTransformer(np.var, {'axis': 0})),
            # ('q25', statistics.SegmentQuantilesTransformer(percentiles=[25])),
            # ('q75', statistics.SegmentQuantilesTransformer(percentiles=[75])),
            # ('zcc', statistics.SegmentZCCTransformer(offset=500)),
            # ('skew', util.SegmentFunctionTransformer(scipy.stats.skew)),
            # ('kurt', util.SegmentFunctionTransformer(scipy.stats.kurtosis)),
        ]))
    ])

    # TODO Klassifikator auch ausgeben!! ------------------------------------------------------------

    diff_features = pipe_all.steps[0][1].transformer_list
    diff_features_list = [diff_feature[0] for diff_feature in diff_features]

    stat_features = pipe_all.steps[1][1].transformer_list
    stat_features_list = [stat_feature[0] for stat_feature in stat_features]

    pipe = Pipeline([
        ('features', FeatureUnion([
            ('pipe_all', pipe_all)
            # ('pipe_imu', pipe_imu),
            # ('pipe_fsr', pipe_fsr),
        ])),
        # ('flattening', util.SegmentFunctionTransformer(np_flatten)),
        # ('standardscaler', StandardScaler()), # Either LDA or StandardScaler is required
        # ('normalizer', Normablizer()), # Bad idea
        # ('pca', PCA()), #n_components=15)),
        # ('te', TestEstimator()),
        ('lda', LinearDiscriminantAnalysis()),
        # ('lda', LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')),# JH special lda
        # ('svm', SVC(C=1000, gamma=0.001, probability=True, kernel='rbf')),
        # ('rf', RandomForestClassifier(n_estimators=100, max_features='auto')),
    ])

    ###################################################
    #                                                 #
    #                  PREP DATA                      #
    #                                                 #
    ###################################################

    prep_start_time = datetime.datetime.now()

    loader = recordings_paths.fill_loader(debug=debug)

    if debug: print('train_subjects = {}'.format(train_subjects))
    train_path_list = loader.get_data(train_subjects, debug=debug, deep_debug=deep_debug)  # get paths for train_subjects
    if debug: print('train_path_list = {}'.format(train_path_list))

    test_subjects = subjects  # split after getting name .split(', ') # making list with test_subjects names
    # test_subjects = ['sample-20190226-150802-JP1']
    test_path_list = loader.get_data(test_subjects, debug=debug, deep_debug=deep_debug)  # get paths for test_subjects

    # weg from tools import make_name
    # get name for plot
    name = make_name(train_subjects=train_subjects, subjects=test_subjects, sensors_list=sensors_list,
                     diff_features_list=diff_features_list, stat_features_list=stat_features_list,
                     fuse_slots=fuse_slots,
                     merge_opposites=merge_opposites,
                     merge_screw_lr=merge_screw_lr, merge_screw_abc=merge_screw_abc,
                     with_date=with_date,
                     show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)

    ###################################################
    #                                                 #
    #                   GET DATA                      #
    #                                                 #
    ###################################################

    # now make subjects into list
    train_subjects = train_subjects.split('_')  # making list with train_subjects names
    test_subjects = test_subjects.split('_')  # making list with train_subjects names

    if debug: print('train_subjects = {}'.format(train_subjects))
    if debug: print('test_subjects = {}'.format(test_subjects))

    sensors_lists = ['dist pos acc gyro'.split()] # TODO OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    sensors_list = sensors_lists[0]

    # prep_start_time = datetime.datetime.now()
    # print('Start time = {}'.format(prep_start_time))

    # print('Prepare data for {}'.format(sensors_list))

    train_sessions, train_targets, train_concat_data, train_concat_targets, train_worked_with = prep_data(train_path_list,
                                                                                                          sensors_lists[0],
                                                                                                          show_time=show_time, debug=debug,
                                                                                                          deep_debug=deep_debug)
    test_sessions, test_targets, test_concat_data, test_concat_targets, test_worked_with = prep_data(test_path_list, sensors_lists[0],
                                                                                                     show_time=show_time, debug=debug,
                                                                                                     deep_debug=deep_debug)

    if debug: print('worked with {} - {}'.format(len(train_worked_with), len(test_worked_with)))

    # prep_end_time = datetime.datetime.now()
    # print('prep duration = {}'.format(prep_end_time - prep_start_time))

    l1 = [30, 40, 50]  # Drehen normal
    l2 = [31, 41, 51]  # Drehen Gegenrichtung

    train_concat_targets = [tar + 2 if tar in l1 else tar + 1 if tar in l2 else tar for tar in train_concat_targets]
    test_concat_targets = [tar + 2 if tar in l1 else tar + 1 if tar in l2 else tar for tar in test_concat_targets]

    prep_end_time = datetime.datetime.now()
    pred_start_time = datetime.datetime.now()
    if show_time: print('prep duration = {}'.format(prep_end_time - prep_start_time))

    # list with all decoders, decoder = list of all slots from all sessions # Beschreibung alt, keinen Decoder mehr TODO
    train_all_data = [slots.values for session in train_sessions for slots in session]

    # list with all decoders, decoder = list of all slots from all sessions # Beschreibung alt, keinen Decoder mehr TODO
    test_all_data = [slots.values for session in test_sessions for slots in session]

    ###################################################
    #                                                 #
    #                CLASSIFICATION                   #
    #                                                 #
    ###################################################

    # fit a model
    # lm = linear_model.LinearRegression()
    model = pipe.fit(train_all_data, train_concat_targets)
    predicted = pipe.predict(test_all_data)

    pred_end_time = datetime.datetime.now()
    draw_start_time = datetime.datetime.now()
    if show_time: print('pred duration = {}'.format(pred_end_time - pred_start_time))

    ###################################################
    #                                                 #
    #                 SHOW RESULTS                    #
    #                                                 #
    ###################################################

    mode = ''
    if not merge_opposites:
        mode = mode + ' no_opp'
    if merge_screw_lr:
        mode = mode + ' abc'
    if merge_screw_abc:
        mode = mode + '  lr'
    if fuse_slots is not None:
        mode = mode + ' ' + str(fuse_slots[2]) # TODO was macht das hier?

    schriftgroesse = 13
    import seaborn as sns
    sns.set_context("paper", rc={"font.size": schriftgroesse, "axes.titlesize": schriftgroesse,
                                 "axes.labelsize": schriftgroesse})

    i = 0
    make_cm_tt(sensors_lists[i], diff_features_list, stat_features_list, train_path_list, test_path_list,
               train_subjects, test_subjects, test_concat_targets, predicted, label_dict, name)

    report = sklearn.metrics.classification_report(test_concat_targets, predicted)
    acc = accuracy_score(test_concat_targets, predicted) * 100
    acc = round(acc, 2)
    # print('---------- acc = {}'.format(acc))
    plot_precisions(report, schriftgroesse, name=name, acc=acc)

    save_infos(sensors_list=sensors_list, diff_features_list=diff_features_list, stat_features_list=stat_features_list,
               train_path_list=train_worked_with, path_list=test_worked_with,
               train_subjects=train_subjects, subjects=subjects,
               targets=test_concat_targets, predicted=predicted,
               rename=label_dict, mode=mode, name=name)

    draw_end_time = datetime.datetime.now()

    print('preparation duration = {}'.format(prep_end_time - prep_start_time))
    print('prediciton duration  = {}'.format(pred_end_time - pred_start_time))
    print('drawing duration     = {}'.format(draw_end_time - draw_start_time))


def do_it_ud_pipe(train_subjects, subjects, pipe=None, path_list=None, sensors_list=None,
          fuse_slots=None, slot_filling=False,
          merge_opposites=False, merge_screw_lr=False, merge_screw_abc=False,
          name=None, with_date=True,
          show_time=False, show_dots=False, debug=False, deep_debug=False):

    ###################################################
    #                                                 #
    #                   PIPELINE                      #
    #                                                 #
    ###################################################

    # TODO Klassifikator auch ausgeben!! ------------------------------------------------------------

    pipe_all = pipe.steps[0][1].transformer_list[0][1]

    diff_features = pipe_all.steps[0][1].transformer_list
    diff_features_list = [diff_feature[0] for diff_feature in diff_features]

    stat_features = pipe_all.steps[1][1].transformer_list
    stat_features_list = [stat_feature[0] for stat_feature in stat_features]

    ###################################################
    #                                                 #
    #                  PREP DATA                      #
    #                                                 #
    ###################################################

    prep_start_time = datetime.datetime.now()

    loader = recordings_paths.fill_loader(debug=debug)

    if debug: print('train_subjects = {}'.format(train_subjects))
    train_path_list = loader.get_data(train_subjects, debug=debug, deep_debug=deep_debug)  # get paths for train_subjects
    if debug: print('train_path_list = {}'.format(train_path_list))

    test_subjects = subjects  # split after getting name .split(', ') # making list with test_subjects names
    # test_subjects = ['sample-20190226-150802-JP1']
    test_path_list = loader.get_data(test_subjects, debug=debug, deep_debug=deep_debug)  # get paths for test_subjects

    # weg from tools import make_name
    # get name for plot
    name = make_name(train_subjects=train_subjects, subjects=test_subjects, sensors_list=sensors_list,
                     diff_features_list=diff_features_list, stat_features_list=stat_features_list,
                     fuse_slots=fuse_slots,
                     merge_opposites=merge_opposites,
                     merge_screw_lr=merge_screw_lr, merge_screw_abc=merge_screw_abc,
                     with_date=with_date,
                     show_time=show_time, show_dots=show_dots, debug=debug, deep_debug=deep_debug)

    ###################################################
    #                                                 #
    #                   GET DATA                      #
    #                                                 #
    ###################################################

    # now make subjects into list
    train_subjects = train_subjects.split('_')  # making list with train_subjects names
    test_subjects = test_subjects.split('_')  # making list with train_subjects names

    if debug: print('train_subjects = {}'.format(train_subjects))
    if debug: print('test_subjects = {}'.format(test_subjects))

    # sensors_lists = ['dist pos acc gyro'.split()]
    # sensors_list = sensors_lists[0]

    # prep_start_time = datetime.datetime.now()
    # print('Start time = {}'.format(prep_start_time))

    # print('Prepare data for {}'.format(sensors_list))

    train_sessions, train_targets, train_concat_data, train_concat_targets, train_worked_with = prep_data(train_path_list,
                                                                                                          sensors_list,
                                                                                                          show_time=show_time, debug=debug,
                                                                                                          deep_debug=deep_debug)
    test_sessions, test_targets, test_concat_data, test_concat_targets, test_worked_with = prep_data(test_path_list, sensors_list,
                                                                                                     show_time=show_time, debug=debug,
                                                                                                     deep_debug=deep_debug)

    if debug: print('worked with {} - {}'.format(len(train_worked_with), len(test_worked_with)))

    # prep_end_time = datetime.datetime.now()
    # print('prep duration = {}'.format(prep_end_time - prep_start_time))

    l1 = [30, 40, 50]  # Drehen normal
    l2 = [31, 41, 51]  # Drehen Gegenrichtung

    train_concat_targets = [tar + 2 if tar in l1 else tar + 1 if tar in l2 else tar for tar in train_concat_targets]
    test_concat_targets = [tar + 2 if tar in l1 else tar + 1 if tar in l2 else tar for tar in test_concat_targets]

    prep_end_time = datetime.datetime.now()
    pred_start_time = datetime.datetime.now()
    if show_time: print('prep duration = {}'.format(prep_end_time - prep_start_time))

    # list with all decoders, decoder = list of all slots from all sessions # Beschreibung alt, keinen Decoder mehr TODO
    train_all_data = [slots.values for session in train_sessions for slots in session]

    # list with all decoders, decoder = list of all slots from all sessions # Beschreibung alt, keinen Decoder mehr TODO
    test_all_data = [slots.values for session in test_sessions for slots in session]

    ###################################################
    #                                                 #
    #                CLASSIFICATION                   #
    #                                                 #
    ###################################################

    # fit a model
    # lm = linear_model.LinearRegression()
    model = pipe.fit(train_all_data, train_concat_targets)
    predicted = pipe.predict(test_all_data)

    pred_end_time = datetime.datetime.now()
    draw_start_time = datetime.datetime.now()
    if show_time: print('pred duration = {}'.format(pred_end_time - pred_start_time))

    ###################################################
    #                                                 #
    #                 SHOW RESULTS                    #
    #                                                 #
    ###################################################

    mode = ''
    if not merge_opposites:
        mode = mode + ' no_opp'
    if merge_screw_lr:
        mode = mode + ' abc'
    if merge_screw_abc:
        mode = mode + '  lr'
    if fuse_slots is not None:
        mode = mode + ' ' + str(fuse_slots[2]) # TODO was macht das hier?

    schriftgroesse = 13
    import seaborn as sns
    sns.set_context("paper", rc={"font.size": schriftgroesse, "axes.titlesize": schriftgroesse,
                                 "axes.labelsize": schriftgroesse})

    i = 0
    make_cm_tt(sensors_list, diff_features_list, stat_features_list, train_path_list, test_path_list,
               train_subjects, test_subjects, test_concat_targets, predicted, label_dict, name)

    report = sklearn.metrics.classification_report(test_concat_targets, predicted)
    acc = accuracy_score(test_concat_targets, predicted) * 100
    acc = round(acc, 2)
    # print('---------- acc = {}'.format(acc))
    plot_precisions(report, schriftgroesse, name=name, acc=acc)

    save_infos(sensors_list=sensors_list, diff_features_list=diff_features_list, stat_features_list=stat_features_list,
               train_path_list=train_worked_with, path_list=test_worked_with,
               train_subjects=train_subjects, subjects=subjects,
               targets=test_concat_targets, predicted=predicted,
               rename=label_dict, mode=mode, name=name)

    draw_end_time = datetime.datetime.now()

    print('preparation duration = {}'.format(prep_end_time - prep_start_time))
    print('prediciton duration  = {}'.format(pred_end_time - pred_start_time))
    print('drawing duration     = {}'.format(draw_end_time - draw_start_time))

    return acc


def get_pipe(pipe_type=0):
    """
    return pipe according to parameter
    0 = idi pipe ( no diff or int)
    1 = i pipe ( with diff and int1)
    :param pipe_type: type of pipe we want, 0 = no diff or int, 1 = diff and int1
    :return: according pipe
    """
    ###################################################
    #                                                 #
    #                   PIPELINE                      #
    #                                                 #
    ###################################################

    def np_flatten(ar):
        return ar.flatten()

    # TODO füllen von Pipeline dynamisch machen? bools übergeben und dann entsprechend dazunehmen?

    pipe_all_i = Pipeline([
        # ('ch_imu', channel_select.SegmentChannelSelectionTransformer(range(0, 6))),
        # von JH für Selektion zwischen IMU und Myo
        ('differential', segment_feature_union.SegmentFeatureUnion([
            ('identity', util.SegmentFunctionTransformer()),
            # ('diff', differential.SegmentDiffTransformer()),
            # ('int1', differential.SegmentIntTransformer()),
            # ('int2', differential.SegmentInt2Transformer()),
        ])),
        ('features', FeatureUnion([
            ('min', util.SegmentFunctionTransformer(np.amin, {'axis': 0})),
            ('max', util.SegmentFunctionTransformer(np.amax, {'axis': 0})),
            ('mean', util.SegmentFunctionTransformer(np.mean, {'axis': 0})),
            # ('med', util.SegmentFunctionTransformer(np.median, {'axis': 0})),
            # ('rms', statistics.SegmentRMSTransformer()),
            # ('std', util.SegmentFunctionTransformer(np.std, {'axis': 0})),
            # ('var', util.SegmentFunctionTransformer(np.var, {'axis': 0})),
            # ('q25', statistics.SegmentQuantilesTransformer(percentiles=[25])),
            # ('q75', statistics.SegmentQuantilesTransformer(percentiles=[75])),
            # ('zcc', statistics.SegmentZCCTransformer(offset=500)),
            # ('skew', util.SegmentFunctionTransformer(scipy.stats.skew)),
            # ('kurt', util.SegmentFunctionTransformer(scipy.stats.kurtosis)),
        ]))
    ])

    pipe_all_idi = Pipeline([
        # ('ch_imu', channel_select.SegmentChannelSelectionTransformer(range(0, 6))),
        # von JH für Selektion zwischen IMU und Myo
        ('differential', segment_feature_union.SegmentFeatureUnion([
            ('identity', util.SegmentFunctionTransformer()),
            ('diff', differential.SegmentDiffTransformer()),
            ('int1', differential.SegmentIntTransformer()),
            # ('int2', differential.SegmentInt2Transformer()),
        ])),
        ('features', FeatureUnion([
            ('min', util.SegmentFunctionTransformer(np.amin, {'axis': 0})),
            ('max', util.SegmentFunctionTransformer(np.amax, {'axis': 0})),
            ('mean', util.SegmentFunctionTransformer(np.mean, {'axis': 0})),
            # ('med', util.SegmentFunctionTransformer(np.median, {'axis': 0})),
            # ('rms', statistics.SegmentRMSTransformer()),
            # ('std', util.SegmentFunctionTransformer(np.std, {'axis': 0})),
            # ('var', util.SegmentFunctionTransformer(np.var, {'axis': 0})),
            # ('q25', statistics.SegmentQuantilesTransformer(percentiles=[25])),
            # ('q75', statistics.SegmentQuantilesTransformer(percentiles=[75])),
            # ('zcc', statistics.SegmentZCCTransformer(offset=500)),
            # ('skew', util.SegmentFunctionTransformer(scipy.stats.skew)),
            # ('kurt', util.SegmentFunctionTransformer(scipy.stats.kurtosis)),
        ]))
    ])

    # TODO Klassifikator auch ausgeben!! ------------------------------------------------------------

    if pipe_type == 1:
        pipe_all = pipe_all_i
    else: # => pipe_type == 0:
        pipe_all = pipe_all_idi

    diff_features = pipe_all.steps[0][1].transformer_list
    diff_features_list = [diff_feature[0] for diff_feature in diff_features]

    stat_features = pipe_all.steps[1][1].transformer_list
    stat_features_list = [stat_feature[0] for stat_feature in stat_features]

    pipe = Pipeline([
        ('features', FeatureUnion([
            ('pipe_all', pipe_all)
            # ('pipe_imu', pipe_imu),
            # ('pipe_fsr', pipe_fsr),
        ])),
        ('flattening', util.SegmentFunctionTransformer(np_flatten)),
        # ('standardscaler', StandardScaler()), # Either LDA or StandardScaler is required
        # ('normalizer', Normablizer()), # Bad idea TODO why? (NormaBBBlizer? oder Normalizer? ist doch Tippfehler!?
        # ('pca', PCA()), #n_components=15)),
        # ('te', TestEstimator()),
        ('lda', LinearDiscriminantAnalysis()),
        # ('lda', LinearDiscriminantAnalysis(shrinkage='auto', solver='eigen')),# JH special lda
        # ('svm', SVC(C=1000, gamma=0.001, probability=True, kernel='rbf')),
        # ('rf', RandomForestClassifier(n_estimators=100, max_features='auto')),
    ])

    return pipe, diff_features_list, stat_features_list
