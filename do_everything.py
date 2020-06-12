import sys
import os
import argparse

def main(args):
	model_path = args.model_path
	model_name = args.model_name
	tomidi = args.tomidi
	wav_path = args.input
	output_path = args.output

	if model_name == "efficientnet":
		sys.path.append(os.path.join(os.path.dirname(__file__), 'sing_voice_transcription/efficientnet'))
		import predict_sequential, predictor

		predictor = predictor.EffNetPredictor(model_path=model_path)
		song_id = "1"
		results = {}
		do_svs = True

		predict_sequential.predict_one_song(predictor, wav_path, song_id
			, results, do_svs=do_svs, tomidi=tomidi, output_path=output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default="efficientnet")
    parser.add_argument('-mp', '--model_path', default="sing_voice_transcription/efficientnet/b4_e_6600")
    parser.add_argument("--tomidi", action="store_true")
    parser.add_argument('-i', "--input")
    parser.add_argument('-o', "--output")
    
    args = parser.parse_args()

    main(args)
