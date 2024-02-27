# Adaptive Fusion of Deep Learning with Statistical Anatomical Knowledge for Robust Patella Segmentation from CT Images

## Description

The codes for the work "Adaptive Fusion of Deep Learning with Statistical Shape Model for Robust Patella Segmentation from CT Images". Our work is especially designed for patella segmentation in low-data regimes.  


## Environment  
Tensorflow>=2.1.0, Python>=3.6.10  

## Instructions

1. Preprocess data to ensure they fit into input size of CNN model.
2. Run NNseg to obtain segmentation results of a CNN model (defined in models.py).  
3. Run SSMseg to obtain segmentation results of SSM.  
4. Run fusion to obtain ultimate segmentation.  
