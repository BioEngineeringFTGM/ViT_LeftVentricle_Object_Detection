#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Oct  1 16:05:09 2022

@author: lisadesanti

"""


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import StratifiedKFold, train_test_split

import cv2
import pandas as pd
import random
from scipy import ndimage

from utilities_function import generate_set, translate, dice_loss_function, obj_detection_metrics, bland_altman_plot



""" Variables Definition """

# --- CURRENT FOLD, CHANGE FROM 1 to k_fold ---
n_fold = 1;                 

# Path to images and annotations
root_dir = "/home/lisadesanti/DeepLearning/MIOT/"
dicom_path = root_dir + "DICOM/";
masks_path = root_dir + "MASKS/";
label_path = root_dir + "hearthPatientList.csv";

# Iterated k-fold cross validation variables
n_trial = 1;                # set the random seed for reproducible shuffling;
random_seed = n_trial*42;   # random seed 
k_fold = 5;                 # number of folds
test_split = 0.2;
valid_split = 0.1;

# Image variables
all_TE = [0, 4, 8];
image_size = 256;
num_TE = len(all_TE); #10;
num_slices = 3;

# Patients' classes
dictionary = {0 : "NormalHeterogeneous", 
              1 : "NormalHomogeneous", 
              2 : "NormalNonMIOT", 
              3 : "PatHeterogeneous", 
              4 : "PatHomogeneous"};

dictionary2 = {"NormalHeterogeneous" : 0, 
               "NormalHomogeneous" : 1, 
               "NormalNonMIOT" : 2, 
               "PatHeterogeneous" : 3, 
               "PatHomogeneous" : 4};

da_dic = {0 : "No_Aug", 
          1 : "Little_Aug", 
          2 : "Strong_Aug"};

lr_dic = {0.0001 : "_lr104", 
          0.00001 : "_lr105"};

bs_dic = {16 : "_bs16", 
          32 : "_bs32"};

ep_dic = {300 : "_ep300", 
          500 : "_ep500"};

ps_dic = {16 : "_ps16", 
          32 : "_ps32",
          64 : "_ps64"};

# Transformer layers
n_channels = len(all_TE); #10;                      # (num_TE) 
input_shape = (image_size, image_size, n_channels); # input image shape
patch_size = 32;                                    # Size of the patches to be 
                                                    # extracted from the input 
                                                    # images
num_patches = (image_size // patch_size) ** 2;
projection_dim = 64;
num_heads = 4;
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
];
transformer_layers = 4;
mlp_head_units = [2048, 1024, 512, 64, 32]; # Size of the dense layers

aug_type =      1;       # Type of Data Augmentation applied
learning_rate = 0.0001;  # Learning Rate
batch_size =    32;      # Batch Size
num_epochs =    300;     # Epochs

# Results Directories 
trial_type = "ViT_Dice_Loss_3TE/" + da_dic[aug_type] + lr_dic[learning_rate] + bs_dic[batch_size] + ep_dic[num_epochs] + ps_dic[patch_size]+"/";
fold_path = root_dir + 'codici_python_MIOT/Results/' + trial_type + 'kfold' + str(n_fold) + '/';
fold_path_images = fold_path + 'Images/';

# Create path if not exits
if not os.path.exists(fold_path):
    os.makedirs(fold_path);
    os.makedirs(fold_path_images);

""" Load Data """

# Select only exams with the epicardial mask
subjs_dicom = os.listdir(dicom_path); # list of subjects dicom directories
subjs_dicom.sort();
subjs_masks = os.listdir(masks_path); # list of subjects mask directories
subjs_masks.sort();
subjs = np.intersect1d(subjs_dicom, subjs_masks); # subjects with masks
                                                  # NOTE: Some subjs don't have
                                                  # epicalrdial mask

# Select only exams belonging to MIOT protocol (see class lable)
# Subject classes
label_dataframe = pd.read_csv(label_path).to_numpy();
MIOT_subjs = label_dataframe[label_dataframe[:,2] != dictionary[2], 1];
label_subjs = label_dataframe[label_dataframe[:,2] != dictionary[2], 2];

# Create dictionary with:
#   - key = exams_id
#   - value = class label
dict_label = {};
for A, B in zip(MIOT_subjs, label_subjs):
    dict_label[A] = B
    
# Select only labels of subjects with epi mask 
label = [];
MIOT_subj_with_epimask = [];
for subj in subjs:
    try:
        label.append(dict_label[subj]);
        MIOT_subj_with_epimask.append(subj);
    except:
        print('Manca: ' + subj);

label = np.array(label);
MIOT_subj_with_epimask = np.array(MIOT_subj_with_epimask);

# Select only one exam per subject (the entire dataset doesn't fit memory)
subjs_id = np.array([exam[0:18] for exam in MIOT_subj_with_epimask]);
subjs_id_to_inspect, subjs_id_to_inspect_idx = np.unique(subjs_id, 
                                                         return_index=True);
exams = MIOT_subj_with_epimask[subjs_id_to_inspect_idx]; # one exams per subjects
num_exams = exams.shape[0];

# Shuffle exams 
rng = np.random.default_rng(random_seed);
shuffled_index = np.arange(num_exams);
rng.shuffle(shuffled_index);
exams = exams[shuffled_index]; # Shuffled dataset
label = label[shuffled_index]; # Shuffled labels
label_num = np.array([dictionary2[el] for el in label]);

# Dataset split into Training and Test set using kfold cross-validation
skf = StratifiedKFold(
    n_splits = k_fold, 
    random_state = random_seed, 
    shuffle = True
    );
kfold_generator = skf.split(exams, label_num);

for i in range(n_fold):
     train_index, test_index = next(kfold_generator);
     exams_train_val    = exams[train_index]; 
     exams_test         = exams[test_index];
     label_train_val    = label_num[train_index];
     label_test         = label_num[test_index];

# Split del Training set in Training set (vero) e Validation set
exams_train, exams_val, label_train, label_val = train_test_split(
    exams_train_val, 
    label_train_val, 
    test_size = valid_split, 
    shuffle = True, 
    random_state = random_seed, 
    stratify = label_train_val
    );

train_set = generate_set(exams_train, label_train, selected_TE = all_TE);
x_train =           train_set[0];
y_train =           train_set[1];
dir_train_set =     train_set[2];
pixel_dims_train =  train_set[3]; 
epi_areas_train =   train_set[4];
label_train =       train_set[5];
endo_areas_train =  train_set[6];

val_set = generate_set(exams_val, label_val, selected_TE = all_TE);
x_val =          val_set[0];
y_val =          val_set[1];
dir_val_set =    val_set[2];
pixel_dims_val = val_set[3]; 
epi_areas_val =  val_set[4];
label_val =      val_set[5];
endo_areas_val = val_set[6];

test_set = generate_set(exams_test, label_test, selected_TE = all_TE);
x_test =          test_set[0];
y_test =          test_set[1];
dir_test_set =    test_set[2];
pixel_dims_test = test_set[3]; 
epi_areas_test =  test_set[4];
label_test =      test_set[5];
endo_areas_test = test_set[6];

img_num_train   = x_train.shape[0];
img_num_val     = x_val.shape[0];
img_num_test    = x_test.shape[0];


""" On-the-Fly Data Augmentation """
def train_preprocessing(image, label):
    # Apply transformation
    image, label = translate(image, label);
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train));
train_dataset = train_dataset.map(train_preprocessing);
train_dataset = train_dataset.batch(batch_size);
train_dataset = train_dataset.prefetch(2);

# # Explore first batch of the augmented training dataset
# image, label = next(iter(train_dataset));
# image = image.numpy();
# label = label.numpy();

# # Iterate
# for i in range(batch_size):
    
#     im = image[i,:,:,0]
#     (h, w) = (im).shape[0:2]
#     coord = label[i,:]
    
#     # Real Bounding Box 
#     ymin, xmin = int(coord[0] * w), int(coord[1] * h)
#     ymax, xmax = int(coord[2] * w), int(coord[3] * h)
#     bbox = np.zeros(im.shape);
#     bbox[ymin:ymax, xmin:xmax] = 1;
    
#     # PLOT
#     fig, ax1 = plt.subplots(figsize=(15, 15));
#     # Real Bounding box
#     ax1.imshow(im, alpha=0.8, cmap='gray');
#     ax1.imshow(im, alpha=0.8, cmap='gray');
#     ax1.imshow(bbox, alpha=0.2, cmap='gray');

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val));
val_dataset = val_dataset.batch(batch_size);
val_dataset = val_dataset.prefetch(2);

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test));
test_dataset = test_dataset.batch(1);
test_dataset = test_dataset.prefetch(2);

    
"""
Implement multilayer-perceptron (MLP)

We use the code from the Keras example Image classification with Vision 
Transformer as a reference.

"""

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


""" Implement the patch creation layer """

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    #     Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # return patches
        return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
    
    
""" Display patches for an input image """

patches = Patches(patch_size)(tf.convert_to_tensor(np.array([x_train[0]]))) # class instance
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"{patches.shape[1]} patches per image \n{patches.shape[-1]} elements per patch")

n = int(np.sqrt(patches.shape[1]))
img_min = np.array([x_train[0]])[0,:,:,0].min();
img_max = np.array([x_train[0]])[0,:,:,0].max();

plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, num_TE))
    plt.imshow(patch_img[:,:,0].numpy().astype("uint8"), 
               cmap='gray', 
               vmax=img_max, 
               vmin=img_min)
    plt.axis("off")
    

"""
Implement the patch encoding layer

The PatchEncoder layer linearly transforms a patch by projecting it into a 
vector of size projection_dim. It also adds a learnable position embedding to 
the projected vector.

"""

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    

"""
Build the ViT model

The ViT model has multiple Transformer blocks. The MultiHeadAttention layer is 
used for self-attention, applied to the sequence of image patches. 
The encoded patches (skip connection) and self-attention layer outputs are 
normalized and fed into a multilayer perceptron (MLP). The model outputs four 
dimensions representing the bounding box coordinates of an object (*).

(*) (x,y) coordinates of top left and bottom right angle of bounding box.

"""

def create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
):
    inputs = layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches) 
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)

    bounding_box = layers.Dense(4)(
        features
    )  # Final four neurons that output bounding box

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=bounding_box)


""" Training setup """

vit_object_detector = create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
)

vit_object_detector.summary()

# Learning Rate
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    learning_rate, 
    decay_steps = 100000,
    decay_rate  = 0.96, 
    staircase   = True
    ); # decay learning rate every decay_steps with a base of decay_rate
 
optimizer = tf.optimizers.Adam(learning_rate = lr_schedule);

# Compile model.
vit_object_detector.compile(
    optimizer   = optimizer,
    loss        = dice_loss_function
    );

""" Run the experiment """

# Train model
checkpoint_filepath = "logs/"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor             = "val_loss",
    save_best_only      = True,
    save_weights_only   = True,
)

train_time = time.time();

history = vit_object_detector.fit(
    train_dataset, 
    epochs = num_epochs,
    validation_data = val_dataset,
    callbacks = [
        checkpoint_callback,
        # keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
    ],
    shuffle = True
)

# Plot ViT training 
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for i, metric in enumerate(["loss"]):
    ax.plot(history.history[metric])
    ax.plot(history.history["val_" + metric])
    ax.set_title("Model {}".format(metric))
    ax.set_xlabel("epochs")
    ax.set_ylabel(metric)
    ax.legend(["train", "val"])
    ax.grid(True)
    
# plt.savefig(fold_path_images + 'training.png');

elapsed_training = time.time() - train_time;

# Save the model in current path
# vit_object_detector.save(fold_path + "vit_object_detector.h5", save_format="h5")

weights_filename = root_dir + 'codici_python_MIOT/Results/' + trial_type + 'kfold' + str(n_fold) + '/vit_object_detector.h5';
vit_object_detector.load_weights(weights_filename);

pred_train = vit_object_detector.predict(train_dataset);
pred_test = vit_object_detector.predict(x_test);

i = 0;
iou_test =              [];
dice_test =             [];
abs_err_test =          [];
epi_frac_err_test =     [];
endo_frac_err_test =    [];

patients = [];
patterns = [dictionary[label_test[i]] for i in range(0, len(label_test), 3)];

iou_test_basal =              [];
dice_test_basal =             [];
abs_err_test_basal =          [];
epi_frac_err_test_basal =     [];
endo_frac_err_test_basal =    [];

iou_test_middle =              [];
dice_test_middle =             [];
abs_err_test_middle =          [];
epi_frac_err_test_middle =     [];
endo_frac_err_test_middle =    [];

iou_test_apical =              [];
dice_test_apical =             [];
abs_err_test_apical =          [];
epi_frac_err_test_apical =     [];
endo_frac_err_test_apical =    [];

# Compare results in the test set
for input_image, coord, patient in zip(x_test, y_test, dir_test_set):
    
    # get (x, y) coordinates of intersection of bounding boxes
    # label: [ymin/img_size, xmin/img_size, ymax/img_size, xmax/img_size]
    # where: 
    #    - [ymin/img_size, xmin/img_size]: TOP (left)
    #    - [ymax/img_size, xmax/img_size]: BOTTOM (right)
    
    #input_image = input_image.numpy();
    #input_image = input_image[0];
    #coord = coord.numpy();
    #coord = coord[0];

    input_image = cv2.resize(
        input_image, (image_size, image_size), interpolation=cv2.INTER_AREA
    )
    
    input_image = np.expand_dims(input_image, axis=0)
    preds = vit_object_detector.predict(input_image)[0]
    print(preds)

    im = input_image[0,:,:,0]
    (h, w) = (im).shape[0:2]

    # Predicted Bounding Box 
    top_left_y, top_left_x = int(preds[0] * w), int(preds[1] * h)
    bottom_right_y, bottom_right_x = int(preds[2] * w), int(preds[3] * h)
    box_predicted = [top_left_y, top_left_x, bottom_right_y, bottom_right_x]

    # Real Bounding Box 
    top_left_y, top_left_x = int(coord[0] * w), int(coord[1] * h)
    bottom_right_y, bottom_right_x = int(coord[2] * w), int(coord[3] * h)
    box_truth = [top_left_y, top_left_x, bottom_right_y, bottom_right_x]

    metrics = obj_detection_metrics(box_predicted, 
                                    box_truth, 
                                    pixel_dims_test[i], 
                                    epi_areas_test[i],
                                    endo_areas_test[i]);
    
    iou =               metrics[0];
    dice =              metrics[1]; 
    abs_err =           metrics[2]; 
    epi_frac_err =      metrics[3];
    endo_frac_err =     metrics[4];
    
    iou_test.append(iou);
    dice_test.append(dice);
    abs_err_test.append(abs_err);
    epi_frac_err_test.append(epi_frac_err);
    endo_frac_err_test.append(endo_frac_err);
    
    patient_name = patient[-38:-12];
    patients.append(patient_name);
    
    patient_mask = patient[-10:-4];
    
    if patient_mask == 'mask_1':
        iou_test_basal.append(iou);
        dice_test_basal.append(dice);
        abs_err_test_basal.append(abs_err);
        epi_frac_err_test_basal.append(epi_frac_err);
        endo_frac_err_test_basal.append(endo_frac_err);
        
    elif patient_mask == 'mask_2':
        iou_test_middle.append(iou);
        dice_test_middle.append(dice);
        abs_err_test_middle.append(abs_err);
        epi_frac_err_test_middle.append(epi_frac_err);
        endo_frac_err_test_middle.append(endo_frac_err);
        
    else:
        iou_test_apical.append(iou);
        dice_test_apical.append(dice);
        abs_err_test_apical.append(abs_err);
        epi_frac_err_test_apical.append(epi_frac_err);
        endo_frac_err_test_apical.append(endo_frac_err);
    
    i += 1;
    
patients = list(dict.fromkeys(patients));

excel_table = {
    
    "Patient" : patients,
    "Pattern" : patterns,
    "iou_B" : iou_test_basal,
    "iou_M" : iou_test_middle,
    "iou_A" : iou_test_apical,
    "dice_B" : dice_test_basal,
    "dice_M" : dice_test_middle,
    "dice_A" : dice_test_apical,
    "abs_err_B" : abs_err_test_basal,
    "abs_err_M" : abs_err_test_middle,
    "abs_err_A" : abs_err_test_apical,
    "epi_frac_err_B" : epi_frac_err_test_basal,
    "epi_frac_err_M" : epi_frac_err_test_middle,
    "epi_frac_err_A" : epi_frac_err_test_apical,
    "endo_frac_err_B" : endo_frac_err_test_basal,
    "endo_frac_err_M" : endo_frac_err_test_middle,
    "endo_frac_err_A" : endo_frac_err_test_apical
    
};

excel_table = pd.DataFrame(excel_table);

endo_frac_err_test_basal    = np.array(endo_frac_err_test_basal);
endo_frac_err_test_middle   = np.array(endo_frac_err_test_middle);
endo_frac_err_test_apical   = np.array(endo_frac_err_test_apical);

corr_id_rate_basal  = (len(endo_frac_err_test_basal[endo_frac_err_test_basal < 1])/len(endo_frac_err_test_basal));
corr_id_rate_middle = (len(endo_frac_err_test_middle[endo_frac_err_test_middle < 1])/len(endo_frac_err_test_middle));
corr_id_rate_apical = (len(endo_frac_err_test_apical[endo_frac_err_test_apical < 1])/len(endo_frac_err_test_apical));


# IoU
iou_test    = np.array(iou_test);
mean_iou    = np.mean(iou_test);
std_iou     = np.std(iou_test);

# DICE
dice_test   = np.array(dice_test);
mean_dice   = np.mean(dice_test);
std_dice    = np.std(dice_test);

# Center Point Absolute Error
abs_err_test = np.array(abs_err_test);
mean_abs_err = np.mean(abs_err_test);
std_abs_err  = np.std(abs_err_test);

# Center Point Epicardial Fractional Error
epi_frac_err_test   = np.array(epi_frac_err_test);
mean_epi_frac_err   = np.mean(epi_frac_err_test);
std_epi_frac_err    = np.std(epi_frac_err_test);

# Center Point Endocardial Fractional Error
endo_frac_err_test  = np.array(endo_frac_err_test);
mean_endo_frac_err  = np.mean(endo_frac_err_test);
std_endo_frac_err   = np.std(endo_frac_err_test);

# Classification Results: Mean +- Std
iou = str(round(mean_iou,2)) + ' ± ' + str(round(std_iou,2));
iou_test_NHet = iou_test[label_test == 0];
iou_NHet = str(round(iou_test_NHet.mean(),2)) + ' ± ' + str(round(iou_test_NHet.std(),2));
iou_test_NHomo = iou_test[label_test == 1];
iou_NHomo = str(round(iou_test_NHomo.mean(),2)) + ' ± ' + str(round(iou_test_NHomo.std(),2));
iou_test_PHet = iou_test[label_test == 3];
iou_PHet = str(round(iou_test_PHet.mean(),2)) + ' ± ' + str(round(iou_test_PHet.std(),2));
iou_test_PHomo = iou_test[label_test == 4];
iou_PHomo = str(round(iou_test_PHomo.mean(),2)) + ' ± ' + str(round(iou_test_PHomo.std(),2));
all_iou = [iou, iou_NHet, iou_NHomo, iou_PHet, iou_PHomo];

dice = str(round(mean_dice,2)) + ' ± ' + str(round(std_dice,2));
dice_test_NHet = dice_test[label_test == 0];
dice_NHet = str(round(dice_test_NHet.mean(),2)) + ' ± ' + str(round(dice_test_NHet.std(),2));
dice_test_NHomo = dice_test[label_test == 1];
dice_NHomo = str(round(dice_test_NHomo.mean(),2)) + ' ± ' + str(round(dice_test_NHomo.std(),2));
dice_test_PHet = dice_test[label_test == 3];
dice_PHet = str(round(dice_test_PHet.mean(),2)) + ' ± ' + str(round(dice_test_PHet.std(),2));
dice_test_PHomo = dice_test[label_test == 4];
dice_PHomo = str(round(dice_test_PHomo.mean(),2)) + ' ± ' + str(round(dice_test_PHomo.std(),2));
all_dice = [dice, dice_NHet, dice_NHomo, dice_PHet, dice_PHomo];

abs_err = str(round(mean_abs_err,2)) + ' ± ' + str(round(std_abs_err,2));
abs_err_test_NHet = abs_err_test[label_test == 0];
abs_err_NHet = str(round(abs_err_test_NHet.mean(),2)) + ' ± ' + str(round(abs_err_test_NHet.std(),2));
abs_err_test_NHomo = abs_err_test[label_test == 1];
abs_err_NHomo = str(round(abs_err_test_NHomo.mean(),2)) + ' ± ' + str(round(abs_err_test_NHomo.std(),2));
abs_err_test_PHet = abs_err_test[label_test == 3];
abs_err_PHet = str(round(abs_err_test_PHet.mean(),2)) + ' ± ' + str(round(abs_err_test_PHet.std(),2));
abs_err_test_PHomo = abs_err_test[label_test == 4];
abs_err_PHomo = str(round(abs_err_test_PHomo.mean(),2)) + ' ± ' + str(round(abs_err_test_PHomo.std(),2));
all_abs_err = [abs_err, abs_err_NHet, abs_err_NHomo, abs_err_PHet, abs_err_PHomo];

epi_frac_err = str(round(mean_epi_frac_err,2)) + ' ± ' + str(round(std_epi_frac_err,2));
epi_frac_err_NHet = str(round(np.mean([epi_frac_err for epi_frac_err in epi_frac_err_test[label_test == 0]]),2)) + ' ± ' + str(round(np.std([epi_frac_err for epi_frac_err in epi_frac_err_test[label_test == 0]]),2));
epi_frac_err_NHomo = str(round(np.mean([epi_frac_err for epi_frac_err in epi_frac_err_test[label_test == 1]]),2)) + ' ± ' + str(round(np.std([epi_frac_err for epi_frac_err in epi_frac_err_test[label_test == 1]]),2));
epi_frac_err_PHet = str(round(np.mean([epi_frac_err for epi_frac_err in epi_frac_err_test[label_test == 3]]),2)) + ' ± ' + str(round(np.std([epi_frac_err for epi_frac_err in epi_frac_err_test[label_test == 3]]),2));
epi_frac_err_PHomo = str(round(np.mean([epi_frac_err for epi_frac_err in epi_frac_err_test[label_test == 4]]),2)) + ' ± ' + str(round(np.std([epi_frac_err for epi_frac_err in epi_frac_err_test[label_test == 4]]),2));
all_epi_frac_err = [epi_frac_err, epi_frac_err_NHet, epi_frac_err_NHomo, epi_frac_err_PHet, epi_frac_err_PHomo];

endo_frac_err = str(round(mean_endo_frac_err,2)) + ' ± ' + str(round(std_endo_frac_err,2));
endo_frac_err_test_NHet = endo_frac_err_test[label_test == 0];
endo_frac_err_NHet = str(round(endo_frac_err_test_NHet.mean(),2)) + ' ± ' + str(round(endo_frac_err_test_NHet.std(),2));
endo_frac_err_test_NHomo = endo_frac_err_test[label_test == 1];
endo_frac_err_NHomo = str(round(endo_frac_err_test_NHomo.mean(),2)) + ' ± ' + str(round(endo_frac_err_test_NHomo.std(),2));
endo_frac_err_test_PHet = endo_frac_err_test[label_test == 3];
endo_frac_err_PHet = str(round(endo_frac_err_test_PHet.mean(),2)) + ' ± ' + str(round(endo_frac_err_test_PHet.std(),2));
endo_frac_err_test_PHomo = endo_frac_err_test[label_test == 4];
endo_frac_err_PHomo = str(round(endo_frac_err_test_PHomo.mean(),2)) + ' ± ' + str(round(endo_frac_err_test_PHomo.std(),2));
all_endo_frac_err = [endo_frac_err, endo_frac_err_NHet, endo_frac_err_NHomo, endo_frac_err_PHet, endo_frac_err_PHomo];

# Correct Identification Rate
corr_id_rate = (len(endo_frac_err_test[endo_frac_err_test < 1])/len(endo_frac_err_test));
corr_id_rate_NHet = (len(endo_frac_err_test_NHet[endo_frac_err_test_NHet < 1])/len(endo_frac_err_test_NHet));
corr_id_rate_NHomo = (len(endo_frac_err_test_NHomo[endo_frac_err_test_NHomo < 1])/len(endo_frac_err_test_NHomo));
corr_id_rate_PHet = (len(endo_frac_err_test_PHet[endo_frac_err_test_PHet < 1])/len(endo_frac_err_test_PHet));
corr_id_rate_PHomo = (len(endo_frac_err_test_PHomo[endo_frac_err_test_PHomo < 1])/len(endo_frac_err_test_PHomo));
all_corr_id_rate = [corr_id_rate, corr_id_rate_NHet, corr_id_rate_NHomo, corr_id_rate_PHet, corr_id_rate_PHomo];

# Build table
all_class = [iou, dice, abs_err, epi_frac_err, endo_frac_err, corr_id_rate];
NHet      = [iou_NHet, dice_NHet, abs_err_NHet, epi_frac_err_NHet, endo_frac_err_NHet, corr_id_rate_NHet];
NHomo     = [iou_NHomo, dice_NHomo, abs_err_NHomo, epi_frac_err_NHomo, endo_frac_err_NHomo, corr_id_rate_NHomo];
PHet      = [iou_PHet, dice_PHet, abs_err_PHet, epi_frac_err_PHet, endo_frac_err_PHet, corr_id_rate_PHet];
PHomo     = [iou_PHomo, dice_PHomo, abs_err_PHomo, epi_frac_err_PHomo, endo_frac_err_PHomo, corr_id_rate_PHomo];

all_performances = pd.DataFrame(
    data = [all_class, NHet, NHomo, PHet, PHomo], 
    columns=['IoU', 
             'DICE', 
             'CP Absolute Error', 
             'CP Epicardial Fractional Error', 
             'CP Endocardial Fractional Error', 
             'Correct Identification Rate']
);

# np.save(fold_path + 'exams_test', exams_test);
# np.save(fold_path + 'exams_train', exams_train);
# np.save(fold_path + 'exams_val', exams_val);
# np.save(fold_path + 'label_test', label_test);
# np.save(fold_path + 'label_train', label_train);
# np.save(fold_path + 'label_val', label_val);
# np.save(fold_path + 'pred_test', pred_test);
# np.save(fold_path + 'pred_train', pred_train);
# np.save(fold_path + 'iou_test', iou_test);
# np.save(fold_path + 'dice_test', dice_test);
# np.save(fold_path + 'abs_err_test', abs_err_test);
# np.save(fold_path + 'epi_frac_err_test', epi_frac_err_test);
# np.save(fold_path + 'endo_frac_err_test', endo_frac_err_test);
# np.save(fold_path + 'all_performances', all_performances);


# Explore output

x_sub = x_test[14:20];
y_sub = y_test[14:20];
count = 1;

for input_image, coord in zip(x_sub, y_sub):
    
    im = input_image[:,:,0]
    
    input_image = cv2.resize(
        input_image, (image_size, image_size), interpolation=cv2.INTER_AREA
    )
    
    input_image = np.expand_dims(input_image, axis=0)
    preds = vit_object_detector.predict(input_image)[0]
    print(preds)

    (h, w) = (im).shape[0:2]

    # Real Bounding Box 
    ymin, xmin = int(coord[0] * w), int(coord[1] * h)
    ymax, xmax = int(coord[2] * w), int(coord[3] * h)
    box_truth = [ymin, xmin, ymax, xmax]
    
    # PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15));
    # Real Bounding box
    bbox = np.zeros(im.shape);
    bbox[ymin:ymax, xmin:xmax] = 1;
    ax1.imshow(im, alpha=0.8, cmap='gray');
    ax1.imshow(bbox, alpha=0.2, cmap='gray');
    ax1.set_xlabel('Real: ' + str(box_truth));
    
    # Predicted Bounding Box
    ymin, xmin = int(preds[0] * w), int(preds[1] * h)
    ymax, xmax = int(preds[2] * w), int(preds[3] * h)
    box_pred = [ymin, xmin, ymax, xmax]
    ax2.set_xlabel('Predicted: ' + str(box_pred));
    
    bbox_pred = np.zeros(im.shape);
    bbox_pred[ymin:ymax, xmin:xmax] = 1;
    ax2.imshow(im, alpha=0.8, cmap='gray');
    ax2.imshow(bbox_pred, alpha=0.2, cmap='gray');
    
    #plt.savefig(fold_path_images + 'example_' + str(count) + '.png');
    
    count += 1;
    
test_time = time.time();
preds = vit_object_detector.predict(input_image)[0]
print(preds)
elapsed_test= time.time() - test_time;
    
    
""" Output charcteristics """

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))

ax1.plot(y_train[:,0])  
ax1.plot(y_train[:,1])
ax1.plot(y_train[:,2])
ax1.plot(y_train[:,3])
ax1.set_title('Training set - Label');
ax1.legend(['top y (left)', 'top x (left)', 'bottom y (right)', 'bottom x (right)'])
ax1.grid(True)
ax1.set_ylim([0.33, 0.68])

ax2.plot(pred_train[:,0])  
ax2.plot(pred_train[:,1])
ax2.plot(pred_train[:,2])
ax2.plot(pred_train[:,3])
ax2.set_title('Training set - Prediction');
ax2.legend(['top y (left)', 'top x (left)', 'bottom y (right)', 'bottom x (right)'])
ax2.grid(True)
ax2.set_ylim([0.33, 0.68])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))

ax1.plot(y_test[:,0])  
ax1.plot(y_test[:,1])
ax1.plot(y_test[:,2])
ax1.plot(y_test[:,3])
ax1.set_title('Test set - Label');
ax1.legend(['top y (left)', 'top x (left)', 'bottom y (right)', 'bottom x (right)'])
ax1.grid(True)
ax1.set_ylim([0.33, 0.68])

ax2.plot(pred_test[:,0])  
ax2.plot(pred_test[:,1])
ax2.plot(pred_test[:,2])
ax2.plot(pred_test[:,3])
ax2.set_title('Test set - Prediction');
ax2.legend(['top y (left)', 'top x (left)', 'bottom y (right)', 'bottom x (right)'])
ax2.grid(True)
ax2.set_ylim([0.33, 0.68])

loss_train = vit_object_detector.evaluate(train_dataset);
loss_val = vit_object_detector.evaluate(val_dataset);
loss_test = vit_object_detector.evaluate(test_dataset);

print("Loss Train: " + str(loss_train));
print("Loss Validation: " + str(loss_val));
print("Loss Test: " + str(loss_test));
print("Mean IoU (Jaccard): " + str(mean_iou) + ' ± ' + str(std_iou));
print("Mean Dice: " + str(mean_dice) + ' ± ' + str(std_dice));
print("Mean Centroid Absolute Error: " + str(mean_abs_err) + ' ± ' + str(std_abs_err));
print("Mean Centroid Epicardial Fractional Error: " + str(mean_epi_frac_err) + ' ± ' + str(std_epi_frac_err));
print("Mean Centroid Endocardial Fractional Error: " + str(mean_endo_frac_err) + ' ± ' + str(std_endo_frac_err));
print("Correct Identification Rate: " + str(corr_id_rate));

results = {
    
    "Learning Rate: " : str(learning_rate),
    "Batch Size: " : str(batch_size),
    "Number of Epochs: " : str(num_epochs),
    "Patch Size: " : str(patch_size),
    "Loss Train: " : str(loss_train),
    "Loss Validation: " : str(loss_val),
    "Loss Test: " : str(loss_test),
    "Mean IoU (Jaccard): " : str(mean_iou) + ' ± ' + str(std_iou),
    "Mean Dice: " : str(mean_dice) + ' ± ' + str(std_dice),
    "Mean Centroid Absolute Error: " : str(mean_abs_err) + ' ± ' + str(std_abs_err),
    "Mean Centroid Epicardial Fractional Error: " : str(mean_epi_frac_err) + ' ± ' + str(std_epi_frac_err), 
    "Mean Centroid Endocardial Fractional Error: " : str(mean_endo_frac_err) + ' ± ' + str(std_endo_frac_err),
    "Correct Identification Rate: " : str(corr_id_rate)
    
    }

# open file for writing
f = open(fold_path + "results.txt","w")
# write file
# f.write( str(results) )
# close file
f.close()


fig, ax = plt.subplots(2, 2, figsize=(12, 12))
title_dic = {"0":"y left corner", "1":"x left corner", "2":"y right corner", "3":"x right corner"};
count = 0;
for i in range(2):
    for j in range(2):
        ax[i,j].scatter(y_test[:, count], pred_test[:, count], marker='*', linewidths=0.7);
        xmin = min([y_test[:, count].min(), pred_test[:, count].min()]);
        xmax = max([y_test[:, count].max(), pred_test[:, count].max()]);
        delta = 0.01;
        bisector = np.arange(xmin-delta, xmax+delta, delta);
        ax[i,j].plot(bisector, bisector, 'r-.', linewidth=0.7)
        ax[i,j].set_title(title_dic[str(count)]);
        ax[i,j].set_xlabel("Ground Truth");
        ax[i,j].set_ylabel("Predicted");
        ax[i,j].grid(True, which='both');
        count += 1;
fig.suptitle(r"$\mathbf{k fold}$ " + str(n_fold), fontsize='large')
# plt.savefig(fold_path_images + 'regression_coordinates.png');

fig, ax = plt.subplots(2, 2, figsize=(12, 8));
count = 0;
for i in range(2):
    for j in range(2):
        
        data1     = np.asarray(y_test[:, count])
        data2     = np.asarray(pred_test[:, count])
        mean      = np.mean([data1, data2], axis=0)
        diff      = data1 - data2                   # Difference between data1 and data2
        md        = np.mean(diff)                   # Mean of the difference
        sd        = np.std(diff, axis=0)            # Standard deviation of the difference
        CI_low    = md - 1.96*sd
        CI_high   = md + 1.96*sd

        ax[i,j].scatter(mean, diff, marker='o', linewidths=0.7);
        ax[i,j].axhline(md,           color='black', linestyle='-');
        ax[i,j].axhline(md + 1.96*sd, color='gray', linestyle='--');
        ax[i,j].axhline(md - 1.96*sd, color='gray', linestyle='--');
        ax[i,j].grid(True, which='both');
        
        fig.suptitle(r"$\mathbf{Bland-Altman}$" + " " + r"$\mathbf{Plot}$" + "\n k fold " + str(n_fold));
        ax[i,j].set_title(title_dic[str(count)]);
        
        ax[i,j].set_xlabel("Means");
        ax[i,j].set_ylabel("Difference");
        ax[i,j].set_ylim(md - 3.5*sd, md + 3.5*sd);

        xOutPlot = np.min(mean) + (np.max(mean)-np.min(mean))*1.14;

        ax[i,j].text(xOutPlot, md - 1.96*sd, r'-1.96SD:' + "\n" + "%.2f" % CI_low, ha = "center", va = "center",);
        ax[i,j].text(xOutPlot, md + 1.96*sd, r'+1.96SD:' + "\n" + "%.2f" % CI_high, ha = "center", va = "center",);
        ax[i,j].text(xOutPlot, md, r'Mean:' + "\n" + "%.2f" % md, ha = "center", va = "center",);
        plt.subplots_adjust(wspace=0.4, hspace=0.4);
        
        count += 1;
#plt.savefig(fold_path_images + 'ba_coordinates.png');




















        
        
        
        
        
