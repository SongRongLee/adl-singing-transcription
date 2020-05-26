# **Applied Deep Learning Spring 2020 Final Project**

## **Ground truth file structure**
```
{
  "song_id": [[start_time, end_time, midi_num], ...]
  ...
}
```
For example:  
```
{
  "1": [[0.0, 0.5, 60], ...],
  ...
  "123": [[0.0, 0.5, 61], ...]
}
```

## **Scripts usage**

### **Evaluation**
This script is used to calculate the evaluation measures.  

`python evaluate.py $gt_file $predicted_file`
- `gt_file`: Ground truth file with .json format.
- `predicted_file`: Predicted file with .json format.

### **Generate dataset instance to a file:**
This script will read from a data directory and generate custom dataset class instance into a binary file.  

`python generate_dataset.py $data_dir $output_dir`  
- `data_dir`: Directory of the songs.
- `output_dir`: Path to the directory to save the dataset.pkl.