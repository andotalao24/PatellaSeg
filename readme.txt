
You may need to run conver2img.py and dataProcess.py to preprocess your data and split them into different sets first.

The assumed file structure is as below
./data
   /Patient case
       / images

1 run NNseg.py to generate segmentation from neural networks
        - add your neural segmentation model to models.py and update NNseg.py correspondingly

2 run SSMseg.py to generate SSM
	- the results are stored in pickle files

3 run fusion.py to conduct the proposed fusion framework.
	- for details of fusion algorithms, please view our paper

If you have relatively ample data, there may be no need to further run step2 and step3. 
Using a proper neural network is already capable enough. The improvement brought by fusion with SSM is subtle, which depends on your data and setting of hyperparameters. 
