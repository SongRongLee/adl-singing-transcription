# **Applied Deep Learning Spring 2020 Final Project -- Singing Transcription**

## Report
[Link](https://github.com/SongRongLee/ntu-adl-final)

## Demo Video
[Link](https://github.com/SongRongLee/ntu-adl-final)

## **Ground truth file structure**
```
{
  "song_id": [[start_time, end_time, pitch], ...]
  ...
}
```
For example:  
```
{
  "1": [[0.0, 0.5, 60], ...],
  ...
  "123": [[0.0, 0.5, 58], ...]
}
```

## **Scripts usage**

### **Convert music file to midi**
This script is used to run efficientnet (TODO: run RNN with pretrained model).

It does svs first (using spleeter), and then run efficientnet. Finally, it generates a midi file. 

`python do_everything.py $input_wav_file $output_path`
- `input_wav_file`: Path to the input file.
- `output_path`: Output midi path.

The default model is efficientnet-b3 and the model path is "sing_voice_transcription/efficientnet/b4_e_6600"

If you want to specify the model path, then you can add another argument:

`python do_everything.py $input_wav_file $output_path -mp $model_path`
- `model_path`: Path to the model file.

### **Evaluation**
This script is used to calculate the evaluation measures.  

`python evaluate.py $gt_file $predicted_file`
- `gt_file`: Ground truth file with .json format.
- `predicted_file`: Predicted file with .json format.

### **Generate dataset instance to a file:**
This script will read from a data directory and generate custom dataset class instance into a binary file.  

`python generate_dataset.py $data_dir $output_dir --for-rnn`  
- `data_dir`: Directory of the songs.
- `output_dir`: Path to the directory to save the dataset.pkl.
- `--for-rnn`: Flag for generating dataset for RNN training.
