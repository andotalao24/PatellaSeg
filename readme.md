# Adaptive Fusion of Deep Learning with Statistical Anatomical Knowledge for Robust Patella Segmentation from CT Images

## Description

The codes for the work "Adaptive Fusion of Deep Learning with Statistical Shape Model for Robust Patella Segmentation from CT Images".
Our work is especially designed for patella segmentation in low-data regimes. Our paper is still under review.  


## Environment  
Tensorflow>=2.1.0, Python>=3.6.10  

## Instruction

- Preprocess data to ensure they fit into input size of CNN model.
- Run NNseg to obtain segmentation results of a CNN model (defined in models.py).  
- Run SSMseg to obtain segmentation results of SSM.  
- Run fusion to obtain ultimate segmentation.  
