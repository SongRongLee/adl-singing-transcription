# **RNN Model**

## **Scripts Usage**

### **Start training:**
This script will train the model. If `--model-path` is given, the training will continue.  
You can use one or multiple pretrained model features in RNN by giving `resnet-model-path`, `effent-model-path` or `alexnet-model-path`. Note that at least one of them needs to be specified.

`python train.py $training_dataset $validation_dataset $model_dir --model-path $model_path  --resnet-model-path $resnet_model_path --effnet-model-path $effnet_model_path --alexnet-model-path $alexnet_model_path`  
- `training_dataset`: Path to the training dataset.pkl.
- `validation_dataset`: Path to the validation dataset.pkl.
- `model_dir`: Directory to save models for each epoch.
- `--model-path`: Path to pretrained model.
- `--resnet-model-path`; Path to pretrained ResNet model.
- `--effnet-model-path`; Path to pretrained EffNet model.
- `--alexnet-model-path`; Path to pretrained AlexNet model.

### **Predicting:**
This script will perform prediction for all songs in a given directory and generate results into a .json file.

`python predict.py $test_dir $predict_file $model_path`  
- `test_dir`: Directory containing the testing songs.
- `predict_file`: Path to output file predict.json.
- `model_path`: Model path.

### **Plot results:**
*Learning curve for testing and validation data*  
`python plot_epoch.py`  
