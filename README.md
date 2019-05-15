# DCGAN-for-Character-Generation

Deep Convolutional Generative Adversarial Networks for Character Generation

#### To run

python nepali_dcgan.py -mode=TRAIN -chr='KA' -train_folder="train_folder"

python nepali_dcgan.py -mode=PREDICT -chr='KA' -n2p=100 -pepochs="3000,2450"

---

## Options
---

### mode : TRAIN or PREDICT
   - **TRAIN** (for training)
   - **PREDICT** (for predict)

### chr : Character to Train/Predict 
 
### train_folder : Training Folder

### n2p : Number of images to predict (default = 100)

### pepochs : Weights for predicting - "epoch1, epoch2, ..."

### epochs : Number of epochs to training (default = 4000)

### batch_size : Batch size for training (default = 32)

### save_interval : Weights save interval


### Demo Inputs
<img src =  https://raw.githubusercontent.com/sagunkayastha/GAN-for-Character-Generation/master/extra/input.jpg height=300 width =300> 

### Demo Outputs 
<img src = https://raw.githubusercontent.com/sagunkayastha/GAN-for-Character-Generation/master/extra/output.jpg height=300 width =300> 
