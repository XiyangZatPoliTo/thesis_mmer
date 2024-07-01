# Thesis: Multimodal Emotion Recognition on RAVDESS

## Audio modality part
### Model structure
* Convolutional part includes 3 ```ConvBlock``` for extracting local information from mel spectrogram input.   
   ```ConvBlock``` : ```Conv1d -> ConvLN -> GELU -> Dropout```
* The returned output of Convolutional part goes through ```Maxpool1d``` and ```LayerNormalization```.
* ```GRU``` part extracts global information bidirectionally. ```GRU``` has an advantage in handling variable-length input.
* ```BahdanauAttention``` is applied at the end of ```GRU```. This enables the model to know **where to pay attention.** 
 
### Dataset split strategy
* train/test dataset with a proportion of 8:2.
* Stratified sampling from the emotion class distribution.

### Special preprocessing of audio
* VAD(Voice Activity Detection) is applied to increase the computational efficiency.
* The start and end points are obtained from power of mel spectrogram.
* There cannot be any emotion in the silent regions(before/after speech), hence they are excluded from analysis.

### Implementation
* Loss function: ```LabelSmoothingLoss``` ([source](https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631))
* Optimizer: ```adabelief_pytorch.AdaBelief``` 
* Scheduler: ```torch.optim.lr_scheduler.ReduceLROnPlateau```

## Video modality part

### Ongoing

- `Facenet` will be used to face detection in preprocessing stage.
- `EfficientFace`(pre-trained) model will be used as visual weight initialization. 
- Same model structure as audio modality part will be used for the first attempt.

## Fusion mechanism
There are three main fusion mechanism in the related works.
This work will firstly try late fusion, which is commonly used by researchers.



## RAVDESS Dataset

* With 8 emotion classes: neutral, calm, happy, sad, angry, fearful, surprise, and disgust
    * View more dataset information [here](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

