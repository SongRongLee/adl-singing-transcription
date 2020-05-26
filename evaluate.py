import argparse
from mir_eval import transcription, io, util
import json
import pandas as pd
import numpy as np

def eval_one_data(answer_true, answer_pred, onset_tolerance=0.05):
    ref_pitches = []
    est_pitches = []
    ref_intervals = []
    est_intervals = []
    for i in range(len(answer_true)):
        if answer_true[i] is not None and float(answer_true[i][1]) - float(answer_true[i][0]) > 0:
            ref_intervals.append([answer_true[i][0], answer_true[i][1]])
            ref_pitches.append(answer_true[i][2])

    for i in range(len(answer_pred)):
        if answer_pred[i] is not None and float(answer_pred[i][1]) - float(answer_pred[i][0]) > 0:
            est_intervals.append([answer_pred[i][0], answer_pred[i][1]])
            est_pitches.append(answer_pred[i][2])

    ref_intervals = np.array(ref_intervals)
    est_intervals = np.array(est_intervals)

    ref_pitches = np.array([float(ref_pitches[i]) for i in range(len(ref_pitches))])
    est_pitches = np.array([float(est_pitches[i]) for i in range(len(est_pitches))])

    ref_pitches = util.midi_to_hz(ref_pitches)
    est_pitches = util.midi_to_hz(est_pitches)

    if len(est_intervals) == 0:
        ret= np.zeros(14)
        ret[9]= len(ref_pitches)
        return ret

    raw_data = transcription.evaluate(ref_intervals, ref_pitches
                    , est_intervals, est_pitches, onset_tolerance=onset_tolerance, pitch_tolerance= 50)

    ret = np.zeros(14)
    ret[0] = raw_data['Precision']
    ret[1] = raw_data['Recall']
    ret[2] = raw_data['F-measure']
    ret[3] = raw_data['Precision_no_offset']
    ret[4] = raw_data['Recall_no_offset']
    ret[5] = raw_data['F-measure_no_offset']
    ret[6] = raw_data['Onset_Precision']
    ret[7] = raw_data['Onset_Recall']
    ret[8] = raw_data['Onset_F-measure']
    ret[9] = len(ref_pitches)
    ret[10] = len(est_pitches)
    ret[11] = int(round(ret[1] * ret[9]))
    ret[12] = int(round(ret[4] * ret[9]))
    ret[13] = int(round(ret[7] * ret[9]))

    return ret


def eval_all(answer_true, answer_pred, onset_tolerance=0.05):

    avg = np.zeros(14)
    for i in range(len(answer_true)):
        ret = eval_one_data(answer_true[i], answer_pred[i], onset_tolerance=onset_tolerance)

        for j in range(14):
            avg[j] = avg[j] + ret[j]

    for j in range(9):
        avg[j] = avg[j] / len(answer_true)

    final_ret = {}
    final_ret['COn'] = avg[2]
    final_ret['COnP'] = avg[5]
    final_ret['COnPOff'] = avg[8]

    return final_ret


class MirEval():
    def __init__(self):
        self.gt = None
        self.tr = None

    def _prepare_list(self, path):
    	with open(path) as json_data:
    	    sol = json.load(json_data)
    	length= len(sol)
    	data= [sol[str(i)] for i in range(1, length+ 1)]
    	return data

    def prepare_data(self, gt_path, tr_path):
        self.gt = self._prepare_list(gt_path)
        self.tr = self._prepare_list(tr_path)

    def accuracy(self):
        return eval_all(self.gt, self.tr)

def main(args):
    my_eval= MirEval()
    my_eval.prepare_data(args.gt_file, args.predicted_file)
    print (my_eval.accuracy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_file')
    parser.add_argument('predicted_file')

    args = parser.parse_args()

    main(args)
