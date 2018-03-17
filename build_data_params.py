"""Params useful to build our datasets"""
import itertools


len_folders = [1821,2726,1471,1915]

train_addrs = [[
        "MatlabAnimal/"+str(j+1)+"/trial_"+str(i+1)+".mat" for i in range(
                len_folders[j])] for j in range(4)]
train_addrs = list(itertools.chain.from_iterable(train_addrs))

map_name_to_type_and_position = {
 'allSignalsRecorded': (38, int),
 'centerPos': (25, float),
 'cerebusOn': (37, int),
 'counter': (14, int),
 'cursorPos': (23, float),
 'decodeCommand': (20, 'sparse'),
 'decodeDiscrete': (22, int),
 'decodePos': (19, float),
 'decodeState': (21, int),
 'delayTime': (43, float),
 'endCounter': (1, int),
 'eyePos': (18, int),
 'fakeSpikeRaster': (34, 'sparse'),
 'handPos': (17, float),
 'isSuccessful': (35, int),
 'juice': (24, 'sparse'),
 'numMarkers': (16, int),
 'numTarget': (44, int),
 'numTotalSpikes': (28, int),
 'numTotalSpikes2': (29, int),
 'paramsValid': (36, int),
 'spikeRaster': (26, 'sparse'),
 'spikeRaster2': (27, 'sparse'),
 'startCounter': (0, int),
 'startDateNum': (2, float),
 'startDateStr': (3, 'str'),
 'state': (15, int),
 'subject': (13, 'str'),
 'timeCerebusEnd': (31, int),
 'timeCerebusEnd2': (33, int),
 'timeCerebusStart': (30, int),
 'timeCerebusStart2': (32, int),
 'timeCueOn': (8, float),
 'timeDelayBegins': (9, float),
 'timeDelayFailed': (10, float),
 'timeFirstTargetAcquire': (40, float),
 'timeLastTargetAcquire': (41, float),
 'timeTargetAcquire': (5, float),
 'timeTargetHeld': (6, float),
 'timeTargetOn': (4, float),
 'timeTrialEnd': (7, float),
 'trialLength': (42, int),
 'trialNum': (39, int)
 }