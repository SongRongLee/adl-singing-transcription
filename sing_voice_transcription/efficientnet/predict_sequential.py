import argparse
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
from predictor import EffNetPredictor
from seq_dataset import SeqDataset
from pathlib import Path
from tqdm import tqdm
import mido

def notes2mid(notes):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    track.append(mido.MetaMessage('set_tempo', tempo=500000))
    track.append(mido.Message('program_change', program=0, time=0))

    previous_offset_time = 0
    cur_total_tick= 0

    for note in notes:
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))

        ticks_since_previous_onset = int(mido.second2tick(note[0], ticks_per_beat=480, tempo=500000))
        ticks_current_note = int(mido.second2tick(note[1]-0.0001, ticks_per_beat=480, tempo=500000))
        note_on_length= ticks_since_previous_onset - cur_total_tick
        note_off_length= ticks_current_note - note_on_length - cur_total_tick

        track.append(mido.Message('note_on', note=note[2], velocity=100, time=note_on_length))
        track.append(mido.Message('note_off', note=note[2], velocity=100, time=note_off_length))
        cur_total_tick = cur_total_tick + note_on_length + note_off_length

    return mid
    

def convert_to_midi(predicted_result, song_id, output_path):
    
    to_convert = predicted_result[song_id]
    # print (len(to_convert))
    mid = notes2mid(to_convert)
    mid.save(output_path)

def predict_one_song(predictor, wav_path, song_id, results, do_svs=False, tomidi=False, output_path= None, show_tqdm=False):

    test_dataset = SeqDataset(wav_path, song_id, do_svs=do_svs)
    results = predictor.predict(test_dataset, results=results, show_tqdm=show_tqdm)

    if tomidi == True:
        convert_to_midi(results, song_id, output_path)

    return results

def main(args):
    # Create predictor
    predictor = EffNetPredictor(model_path=args.model_path)

    results = {}
    print('Predicting {}...'.format(args.test_dir))
    for song_dir in tqdm(sorted(Path(args.test_dir).iterdir())):
        wav_path = song_dir / 'Vocal.wav'
        song_id = song_dir.stem

        output_path = song_dir / 'trans.mid'

        results = predict_one_song(predictor, wav_path, song_id, results, do_svs=False, tomidi=False, output_path= None)
        
    # Write results to target file
    with open(args.predict_file, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)

    print('Prediction done. File writed to: {}'.format(args.predict_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir')
    parser.add_argument('predict_file')
    parser.add_argument('model_path')
    parser.add_argument('-m', "--tomidi", action="store_true")

    args = parser.parse_args()

    main(args)
