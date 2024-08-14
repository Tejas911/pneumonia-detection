# Implementation of Distributed Computing with Model Quantization for Pneumonia Image Detection.
This project implements distributed computing and model quantization for pneumonia image detection. A dataset is split into four subsets, each training a separate Xception model. The models are merged using weighted averaging, prioritizing those with higher accuracy. The final ensemble is quantized to optimize performance for resource-limited environments, with the AUC ROC curve used to address any accuracy loss from quantization.

## Project Structure
Project Root
```
│
├── .DS_Store                     # System file created by macOS (not project-related)
├── .gitignore                    # Specifies files/directories for Git to ignore
├── LICENSE                       # MIT License 
├── README.md                     # Overview and setup instructions for the project
│
├── class_labels.txt              # Lists class labels ("Normal" and "Pneumonia")
├── data_split.ipynb              # Splits the dataset into different subsets
├── merge.ipynb                   # Merges the trained models using weighted averaging
├── model_quantization.ipynb      # Handles the quantization of the merged model
├── quantized_model.tflite        # Final quantized model in TensorFlow Lite format
├── requirements.txt              # Python dependencies for the project
│
├── xception_build_on_entire_data.ipynb  # Trains an Xception model on the entire dataset
├── xception_build_train1.ipynb   # Trains an Xception model on the first data subset
├── xception_build_train2.ipynb   # Trains an Xception model on the second data subset
├── xception_build_train3.ipynb   # Trains an Xception model on the third data subset
└── xception_build_train4.ipynb   # Trains an Xception model on the fourth data subset
```

- ADD ACCURACIES in graphs
- HOW THE MODELS ARE MERGERD
- HOW TO USE
- BEEFITS OF DISTRIBUTED COPUTING AND QUATISATION
- ISSUES FACED
- OTHER NICE THINGS
