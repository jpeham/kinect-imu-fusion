
# loads data of session recordings of specific subjects
# hands it out per subject


class DataLoader:

    def __init__(self, time=True, debug=False, deep_debug=False):
        self.subject_tups = [] # TODO name? recording_data?
        self.name_list = []
        if debug: print('new DataLoader created')
        # TODO fill (siehe jh?)

    def add(self, tup, time=True, debug=False, deep_debug=False):
        self.subject_tups.append(tup)
        self.name_list.append(tup[0])
        if debug: print('-filled dataloader with paths from {}'.format(tup[0]))

    def get_data(self, subject_list=None, time=True, debug=False, deep_debug=False):
        # TODO vorerst nur mal path_lists zurückgeben, damit selbst data abholen
        # TODO --> direkt hier data abholen und gleich weitergeben TODO!!!

        if debug: print('getting data of subjects {}'.format(subject_list))

        bool_all = subject_list == None # TODO wenn keine subject_list, dann alle hergeben
        if debug: print('give all rec_paths? - {}'.format(bool_all))

        subjects_paths = []

        if bool_all:
            # TODO direkt hier Erstellen von Path_list und return, gar nicht runter gehen kompliziert in Schleife rein
            subjects_paths = [path for subject_tup in self.subject_tups for path in subject_tup[1]]
            return subjects_paths

        if debug: print('subject_list = {}'.format(subject_list))

        for subject_tup in self.subject_tups:
            if deep_debug: print('checking {}'.format(subject_tup[0]), end='')
            if deep_debug: print(' for {} - {}'.format(subject_list, type(subject_list)))

            if subject_tup[0] in subject_list:
                for path in subject_tup[1]:
                    subjects_paths.append(path)
                # subjects_paths.append(subject_tup[1])
                if debug: print('giving out subject_tup {}'.format(subject_tup[0]))

        return subjects_paths
