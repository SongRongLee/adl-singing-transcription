import sys
import os
import argparse


def main(args):
    model_path = args.model_path
    model_name = args.model_name
    wav_path = args.input
    output_path = args.output

    if model_name == 'efficientnet':
        sys.path.append(os.path.join(os.path.dirname(__file__), 'sing_voice_transcription/efficientnet'))
        import predict_sequential
        import predictor

        predictor = predictor.EffNetPredictor(model_path=model_path)
        song_id = '1'
        results = {}
        do_svs = True

        predict_sequential.predict_one_song(predictor, wav_path, song_id, results, do_svs=do_svs, tomidi=True, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('-m', '--model_name', default='efficientnet')
    parser.add_argument('-mp', '--model_path', default='sing_voice_transcription/efficientnet/b4_e_6600')

    args = parser.parse_args()

    main(args)
