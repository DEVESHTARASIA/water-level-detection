
#...............................Imports..................................................................

import argparse
import os
import torch
import numpy as np
import FCN_NetModel as FCN # The net Class
import CategoryDictionary as CatDic
import cv2

############################################Input parameters###################################################################################
#-------------------------------------Input parameters-----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--inputdir', type=str, default='InputImages/')
parser.add_argument('--outdir', type=str, default='Out/')
parser.add_argument('--gpu', type=str, default=False)
parser.add_argument('--freeze', type=str, default=False)
parser.add_argument('--trainedmodel', type=str, default='logs//TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch')

opt_parser = parser.parse_args()

InputDir=opt_parser.inputdir 
OutDir=opt_parser.outdir 

UseGPU=bool(opt_parser.gpu) 
FreezeBatchNormStatistics=bool(opt_parser.freeze) 
OutEnding="" 
if not os.path.exists(OutDir): os.makedirs(OutDir) 

Trained_model_path = opt_parser.trainedmodel

Net=FCN.Net(CatDic.CatNum) 
if UseGPU==True:
    print("USING GPU")
    Net.load_state_dict(torch.load(Trained_model_path))
else:
    print("USING CPU")
    Net.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))


for name in os.listdir(InputDir): 
    print(name)
    InPath=InputDir+"/"+name
    Im=cv2.imread(InPath)
    h,w,d=Im.shape
    r=np.max([h,w])
    if r>840: 
        fr=840/r
        Im=cv2.resize(Im,(int(w*fr),int(h*fr)))
    Imgs=np.expand_dims(Im,axis=0)
    if not (type(Im) is np.ndarray):
        continue
    with torch.autograd.no_grad():
          OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNormStatistics) # Run net inference and get prediction
    
    for nm in OutLbDict:
        Lb=OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8)
        if Lb.mean()<0.001: continue
        if nm=='Ignore': continue
        ImOverlay1 = Im.copy()
        ImOverlay1[:, :, 0][Lb==1] = 255
        ImOverlay1[:, :, 1][Lb==1] = 0
        ImOverlay1[:, :, 2][Lb==1] = 255
        FinIm=np.concatenate([Im,ImOverlay1],axis=1)
        OutPath = OutDir + "//" + nm+"/"

        if not os.path.exists(OutPath): os.makedirs(OutPath)
        OutName=OutPath+name[:-4]+OutEnding+".png"
        cv2.imwrite(OutName,FinIm)







