#!/bin/bash
(nohup python CAE_Depth_2.py &&
nohup python CAE_SEM_2.py &&
nohup python CAE_RGB_2.py &&
nohup python CAE_full_2.py &&
nohup python CAE_full_2_test.py) &


