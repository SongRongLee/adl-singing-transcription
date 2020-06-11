import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sing_voice_transcription/efficientnet'))
import predict_sequential, predictor

model_path = "sing_voice_transcription/efficientnet/b4_e_6600"

predictor = predictor.EffNetPredictor(model_path=model_path)

wav_path = "../yurucamp.mp3"
song_id = "1"
results = {}
do_svs = True
tomidi = True
output_path = "../yurucamp_trans.mid"

predict_sequential.predict_one_song(predictor, wav_path, song_id, results, do_svs=do_svs, tomidi=tomidi, output_path=output_path)