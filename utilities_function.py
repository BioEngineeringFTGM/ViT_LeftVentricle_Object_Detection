#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:55:01 2022

@author: lisadesanti
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pydicom
import SimpleITK as sitk
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
from sklearn.model_selection import StratifiedKFold

import cv2
import scipy.io
import shutil
import random
from scipy import ndimage



def generate_set(exams, labels, image_size = 256, num_TE = 10, num_slices = 3, 
                 root_dir = "/home/lisadesanti/DeepLearning/MIOT/",
                 plot_epimask = False,
                 selected_TE = False):
    """
    Function which returns:
        
        - X:    Array of input images, 
                type: numpy.ndarray, 
                shape: (n_img, image_size, image_size, num_TE);
        - y:    Ground truth, 
                type: numpy.ndarray, 
                shape: (n_img, 4);
        - all_directories:  list of masks directories
        - pixel_dims:       list of images resolution
        - epi_areas:        list of epicardial mask's areas
        - endo_areas:       list of endocardial mask's areas
        - real_labels:      list of classes of selected images
    
    """
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    
    dicom_path = root_dir + "DICOM/";
    masks_path = root_dir + "MASKS/";
    
    X =                 [];
    boxes_truth =       [];
    all_directories =   [];
    pixel_dims =        [];
    epi_areas =         [];
    real_labels =       [];
    endo_areas =        [];
    
    if selected_TE:
        num_TE = len(selected_TE);
    else:
        selected_TE = list(range(num_TE));
    
    for exam, label in zip(exams, labels):
        
        exam_path = os.path.join(dicom_path, exam);
        reader = sitk.ImageSeriesReader();
        dicom_names_image = reader.GetGDCMSeriesFileNames(exam_path);
        reader.SetFileNames(dicom_names_image);
        mri_seq = reader.Execute();
        pixel_dim = mri_seq.GetSpacing();
        dx = pixel_dim[0];
        dy = pixel_dim[1];
        dz = pixel_dim[2];
        mri_seq = sitk.GetArrayFromImage(mri_seq);  # dim: (30, 256, 256); 
                                                    # (3 slices, 10 TE):
                                                    #   - [0:9] = Basal
                                                    #   - [10:19] = Middle
                                                    #   - [20:29] = Apical
        
        epi_masks  = ['mask_1.tif', 'mask_2.tif', 'mask_3.tif'];
        endo_masks = ['mask_1_endo.tif', 'mask_2_endo.tif', 'mask_3_endo.tif'];
    
        # 'mask_1.tif':         Epicardial BASAL
        # 'mask_1_endo.tif':    Endocardial BASAL
        # 'mask_2.tif':         Epicardial MIDDLE
        # 'mask_2_endo.tif':    Endocardial MIDDLE
        # 'mask_3.tif':         Epicardial APICAL
        # 'mask_3_endo.tif':    Endocardial APICAL

        for i, mask in enumerate(zip(epi_masks, endo_masks)):
            
            epi_mask    = mask[0];
            endo_mask   = mask[1];
            
            epi_mask_path = os.path.join(masks_path, exam, epi_mask);
            epi_mask = np.flip(np.array(Image.open(epi_mask_path)), axis=0);
            # epi_mask = np.array(Image.open(epi_mask_path));
            
            endo_mask_path = os.path.join(masks_path, exam, endo_mask);
            endo_mask = np.flip(np.array(Image.open(endo_mask_path)), axis=0);
            
            # Generate ground truth (bounding box coordinates)
            epi = np.argwhere(epi_mask>0);  # index of non-zero elements,
                                            # [id_row, id_col]
            epi_area = np.sum(epi_mask>0)*dx*dy; 
            
            endo = np.argwhere(endo_mask>0);    # index of non-zero elements,
                                                # [id_row, id_col]
            endo_area = np.sum(endo_mask>0)*dx*dy; 
            
            ymin = epi[:,0].min();      # top y (left)
            ymax = epi[:,0].max()+1;    # bottom y (right)
            xmin = epi[:,1].min();      # top x (left)
            xmax = epi[:,1].max()+1;    # bottom x (right)
            
            box_coord = np.array([ymin/image_size, 
                                  xmin/image_size, 
                                  ymax/image_size, 
                                  xmax/image_size]);       
            
            # Bounding box
            bbox = np.zeros(epi_mask.shape);
            bbox[ymin:ymax, xmin:xmax] = 1;
    
            # Save inputs images
            flag_image = False;
            image = np.zeros((image_size, image_size, num_TE));
            
            # Save inputs images
            for idx_im, j in enumerate(selected_TE):
                
                # Dimensionality Check
                if mri_seq[i*num_TE+j,:,:].shape == (image_size, image_size):
                    a = int(mri_seq.shape[0]/len(epi_masks));
                    image[:, :, idx_im] = mri_seq[i*a+j, :, :];
                    # image[:,:,j] = epi_mask; # toy task 
                    flag_image = True;
                    
                else:
                    break;
            
            # Check that image has the expected dimensions
            if flag_image:
                X.append(image);
                boxes_truth.append(box_coord); # Append to list box coordinate
                all_directories.append(epi_mask_path);
                epi_areas.append(epi_area);
                endo_areas.append(endo_area);
                pixel_dims.append(pixel_dim);
                real_labels.append(label);
                
            if plot_epimask:
                fig, ax= plt.subplots(figsize=(15, 15));
                im = image[:,:,0];
                plt.imshow(im, alpha=0.8, cmap='gray');
                plt.imshow(epi_mask, alpha=0.2, cmap='gray');
                plt.imshow(bbox, alpha=0.2, cmap='gray');
    
    X = np.array(X);
    y = np.array(boxes_truth);  # shape: (num_slices*num_exams, 4) 
                                # = (3*num_exams, 4)
    real_num_img = y.shape[0];
    real_labels = np.array(real_labels);
    
    return X, y, all_directories, pixel_dims, epi_areas, real_labels, endo_areas


""" Data Augmentation Function """

@tf.function
def translate(image, label): 
    def scipy_translate(image, label):
        shift = np.arange(-80,80,5); # translation factor
        x_shift = random.choice(shift); # pick x shift at random
        y_shift = random.choice(shift); # pick y shift at random
        z_shift = 0;
        image = ndimage.shift(image, 
                              [y_shift, x_shift, z_shift], 
                              order=1,
                              mode='nearest');
        label = [label[0] + y_shift/256,
                 label[1] + x_shift/256,
                 label[2] + y_shift/256,
                 label[3] + x_shift/256]
        image = tf.cast(image, tf.double);
        label = tf.cast(label, tf.double);
        #print(str(x_shift) + ' ' + str(y_shift))
        return (image, label)
    aug_image, aug_label = tf.numpy_function(scipy_translate, [image, label], [tf.double, tf.double]);
    return aug_image, aug_label


""" Custom Loss Function """

def dice_loss_function(y_true, y_pred):
    
    # get (x, y) coordinates of intersection of bounding boxes
    # label: [ymin/img_size, xmin/img_size, ymax/img_size, xmax/img_size]
    # where: 
    #    - [ymin/img_size, xmin/img_size]: TOP (left)
    #    - [ymax/img_size, xmax/img_size]: BOTTOM (right)
    
    top_y_intersect = tf.math.maximum(y_pred[:, 0], y_true[:, 0]);
    top_x_intersect = tf.math.maximum(y_pred[:, 1], y_true[:, 1]);
    bottom_y_intersect = tf.math.minimum(y_pred[:, 2], y_true[:, 2]);
    bottom_x_intersect = tf.math.minimum(y_pred[:, 3], y_true[:, 3]);

    # calculate area of the intersection bb (bounding box)
    intersection_area = tf.math.maximum(0., bottom_x_intersect - top_x_intersect + 1) * tf.math.maximum(
        0., bottom_y_intersect - top_y_intersect + 1
    );

    # calculate area of the prediction bb and ground-truth bb
    box_predicted_area = (y_pred[:, 2] - y_pred[:, 0] + 1) * (y_pred[:, 3] - y_pred[:, 1] + 1);
    box_truth_area = (y_true[:, 2] - y_true[:, 0] + 1) * (y_true[:, 3] - y_true[:, 1] + 1);
    
    # calculate dice by taking intersection area and dividing it by the mean of 
    # predicted bb and ground truth bb areas 
    num = 2*intersection_area + 0.00001;
    den = box_predicted_area + box_truth_area + 0.00001;
    dice = num/den;
    dice = 1-K.mean(dice);
    # tf.cast(dice, tf.float32)
   
    return dice


""" Evaluation Metrics """

def obj_detection_metrics(box_predicted, box_truth, pixel_dim, 
                          epi_mask_area, endo_mask_area):
    
    # get (x, y) coordinates of intersection of bounding boxes
    # label: [ymin/img_size, xmin/img_size, ymax/img_size, xmax/img_size]
    # where: 
    #    - [ymin/img_size, xmin/img_size]: TOP (left)
    #    - [ymax/img_size, xmax/img_size]: BOTTOM (right)
    
    top_y_intersect = max(box_predicted[0], box_truth[0]);
    top_x_intersect = max(box_predicted[1], box_truth[1]);
    bottom_y_intersect = min(box_predicted[2], box_truth[2]);
    bottom_x_intersect = min(box_predicted[3], box_truth[3]);

    # Area of the intersection bb (bounding box)
    intersection_area = max(0, bottom_x_intersect - top_x_intersect + 1) * max(
        0, bottom_y_intersect - top_y_intersect + 1
    );

    # Area of the prediction bb and ground-truth bb
    box_predicted_area = (box_predicted[2] - box_predicted[0] + 1) * (
        box_predicted[3] - box_predicted[1] + 1
    );
    box_truth_area = (box_truth[2] - box_truth[0] + 1) * (
        box_truth[3] - box_truth[1] + 1
    );
    
    # Centre of predicted and ground-truth bb
    center_y_predicted = (box_predicted[0] + box_predicted[2] + 1) / 2;
    center_x_predicted = (box_predicted[1] + box_predicted[3] + 1) / 2;
    center_y_truth = (box_truth[0] + box_truth[2] + 1) / 2;
    center_x_truth = (box_truth[1] + box_truth[3] + 1) / 2;

    # calculate intersection over union by taking intersection
    # area and dividing it by the sum of predicted bb and ground truth
    # bb areas subtracted by  the interesection area
    iou = intersection_area / float(
        box_predicted_area + box_truth_area - intersection_area
    );
    
    # calculate DICE by taking intersection area and dividing it by the mean of 
    # predicted bb and ground truth bb areas 
    dice = 2 * intersection_area / float(box_predicted_area + box_truth_area);
    
    # Absolute Error between real and predicted BB centre
    dx = pixel_dim[1];
    dy = pixel_dim[0];
    center_predicted = [center_y_predicted*dy, center_x_predicted*dx];
    center_truth = [center_y_truth*dy, center_x_truth*dx];
    abs_err = math.dist(center_predicted, center_truth);
    
    # Fractional Error between real and predicted BB centre of epicardial mask
    epi_mask_rad = math.sqrt(epi_mask_area/math.pi);
    epi_frac_err = abs_err/epi_mask_rad;
    
    # Fractional Error between real and predicted BB centre of endocardial mask
    endo_mask_rad = math.sqrt(endo_mask_area/math.pi);
    endo_frac_err = abs_err/endo_mask_rad;
    
    # return metrics
    return (iou, dice, abs_err, epi_frac_err, endo_frac_err)


def bland_altman_plot(data1, data2, title=False):
    
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2           # Difference between data1 and data2
    md        = np.mean(diff)           # Mean of the difference
    sd        = np.std(diff, axis=0)    # Standard deviation of the difference
    CI_low    = md - 1.96*sd
    CI_high   = md + 1.96*sd

    fig, ax = plt.subplots(figsize=(12, 8));
    ax.scatter(mean, diff, marker='o', linewidths=0.7);
    plt.axhline(md,           color='black', linestyle='-');
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--');
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--');
    ax.grid(True, which='both');
    
    ax.set_title(r"$\mathbf{Bland-Altman}$" + " " + r"$\mathbf{Plot}$");
    plt.xlabel("Means");
    plt.ylabel("Difference");
    plt.ylim(md - 3.5*sd, md + 3.5*sd);

    xOutPlot = np.min(mean) + (np.max(mean)-np.min(mean))*1.14;

    plt.text(xOutPlot, md - 1.96*sd, r'-1.96SD:' + "\n" + "%.2f" % CI_low, ha = "center", va = "center",);
    plt.text(xOutPlot, md + 1.96*sd, r'+1.96SD:' + "\n" + "%.2f" % CI_high, ha = "center", va = "center",);
    plt.text(xOutPlot, md, r'Mean:' + "\n" + "%.2f" % md, ha = "center", va = "center",);
    plt.subplots_adjust(right=0.85);
    plt.show()
    
    if title:
        fig.suptitle(title);

    return md, sd, mean, CI_low, CI_high


def plot_slices(data, num_columns=10, cmap="gray", title=False, data_min=False, data_max=False):
    
    width = data.shape[0];
    height = data.shape[1];
    depth = data.shape[2];
    
    if not(data_min) or not(data_max):
        data_min = data.min();
        data_max = data.max();
    
    r, num_rows = math.modf(depth/num_columns);
    num_rows = int(num_rows);
    if num_rows == 0:
        num_columns = int(r*num_columns);
        num_rows +=1;
        r = 0;
    elif r > 0:
        new_im = int(num_columns-(depth-num_columns*num_rows));
        add = np.zeros((width,height,new_im),dtype=type(data[0,0,0]));
        data = np.dstack((data,add));
        num_rows +=1;
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    );
    
    if title:
        f.suptitle(title, fontsize=30);
        
    for i in range(rows_data):
        for j in range(columns_data):
            if rows_data > 1:
                img = axarr[i, j].imshow(data[i][j], cmap=cmap,
                                         vmin=data_min, vmax=data_max);
                axarr[i, j].axis("off");
            else:
                img = axarr[j].imshow(data[i][j], cmap=cmap,
                                      vmin=data_min, vmax=data_max);
                axarr[j].axis("off");

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right = 0.9, bottom=0, top=0.9);
    cbar_ax = f.add_axes([0.92, 0, 0.015, 0.9])         
    f.colorbar(img, cax=cbar_ax);
    plt.show()









