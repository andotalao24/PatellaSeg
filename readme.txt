
You may need to run conver2img.py and dataProcess.py to preprocess your data first.

The assumed file structure is as below
./data
   /Patient case
       / images

1 run NNseg.py to generate segmentation from neural networks
                - add any model you want to models.py and update NNseg.py correspondingly

2 run SSMseg.py to generate SSM
	- the results are stored in pickle file

3 run fusion.py to conduct the proposed fusion framework.
	- for details, please view our paper

If you have relatively ample data (e.g., >30 patient cases), there may be no need to further run step2 and step3. 
Using a proper neural network is already capable enough. The improvement brought by fusion with SSM is subtle, which depends on your data. 
