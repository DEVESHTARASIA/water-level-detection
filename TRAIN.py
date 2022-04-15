#

import os
import torch
import numpy as np
import ChemReader
import FCN_NetModel as FCN # The net Class
import CategoryDictionary as CatDic
import Evaluator
import scipy.misc as misc
TrainFolderPath=r"/scratch/gobi2/seppel/Chemscape/LabPicsV1/"

ChemTrainDir=TrainFolderPath+r"/Complex/Train//" 
ChemTestDir=TrainFolderPath+r"/Complex/Test//" 


TrainedModelWeightDir="logs/" 
if not os.path.exists(TrainedModelWeightDir): os.mkdir(TrainedModelWeightDir)
Trained_model_path="" 

Learning_Rate_Init=1e-5 
Learning_Rate=1e-5 
Learning_Rate_Decay=Learning_Rate/20
StartLRDecayAfterSteps=100000
MaxBatchSize=7 
MinSize=250 
MaxSize=1000
MaxPixels=340000*3
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" 
Weight_Decay=1e-5
MAX_ITERATION = int(10000000010) 
InitStep=0

Eval=Evaluator.Evaluator(ChemTestDir,TrainedModelWeightDir+"/Evaluat.xls")

ChemReader=ChemReader.Reader(MainDir=ChemTrainDir,MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)

if os.path.exists(TrainedModelWeightDir + "/Defult.torch"): Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"): Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate_Init.npy"): Learning_Rate_Init=np.load(TrainedModelWeightDir+"/Learning_Rate_Init.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))

Net=FCN.Net(CatDic.CatNum) 
if Trained_model_path!="": 
    Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) 

AVGLoss={}
for nm in CatDic.CatLossWeight:
    AVGLoss[nm]=-1

AVGtotalLoss=-1

if not os.path.exists(TrainedModelWeightDir):
    os.makedirs(TrainedModelWeightDir) 
f = open(TrainLossTxtFile, "w+")
txt="Iteration\t Learning Rate\t Learning rate\t"
for nm in AVGLoss: txt+="\t"+nm+" loss"
f.write(txt+"\n")
f.close()

print("Start Training")
for itr in range(InitStep,MAX_ITERATION): 
    Imgs, Ignore, AnnMaps, AnnMapsBG = ChemReader.LoadBatch()

    OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=True) 
    Net.zero_grad()

    Loss = 0
    LossByCat={}
    ROI = torch.autograd.Variable(torch.from_numpy((1-Ignore).astype(np.float32)).cuda(), requires_grad=False)
    for nm in OutProbDict:
        if CatDic.CatLossWeight[nm]<=0: continue
        if nm in AnnMaps:
            GT=torch.autograd.Variable( torch.from_numpy(AnnMaps[nm].astype(np.float32)).cuda(), requires_grad=False)
            LossByCat[nm]=-torch.mean(ROI*(GT * torch.log(OutProbDict[nm][:,1,:,:] + 0.0000001)+(1-GT) * torch.log(OutProbDict[nm][:,0,:,:] + 0.0000001)))
            Loss=LossByCat[nm]*CatDic.CatLossWeight[nm]+Loss



    Loss.backward() 
    optimizer.step() 

    if AVGtotalLoss == -1:
        AVGtotalLoss = float(Loss.data.cpu().numpy())  
    else:
        AVGtotalLoss = AVGtotalLoss * 0.999 + 0.001 * float(Loss.data.cpu().numpy())

    for nm in LossByCat:
        if AVGLoss[nm]==-1:  AVGLoss[nm]=float(LossByCat[nm].data.cpu().numpy()) 
        else: AVGLoss[nm]= AVGLoss[nm]*0.999+0.001*float(LossByCat[nm].data.cpu().numpy()) 
    if itr % 2000 == 0 and itr>0: 
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/Learning_Rate_Init.npy",Learning_Rate_Init)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 10000 == 0 and itr>1: 
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
    if itr % 10000 == 0:
        Eval.Eval(Net,itr)

    if itr % 50==0: 
        txt="\nIteration\t="+str(itr)+"\tLearning Rate\t"+str(Learning_Rate)+"\tInit_LR=\t"+str(Learning_Rate_Init)+"\tLoss=\t"+str(AVGtotalLoss)+"\t"
        for nm in AVGLoss:
            txt+="\t"+nm+"=\t"+str(AVGLoss[nm])
        print(txt)
        
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()

    if itr%10000==0 and itr>=StartLRDecayAfterSteps:
        Learning_Rate-= Learning_Rate_Decay
        if Learning_Rate<=1e-7:
            Learning_Rate_Init-=2e-6
            if Learning_Rate_Init<1e-6: Learning_Rate_Init=1e-6
            Learning_Rate=Learning_Rate_Init*1.00001
            Learning_Rate_Decay=Learning_Rate/20
        print("Learning Rate="+str(Learning_Rate)+"   Learning_Rate_Init="+str(Learning_Rate_Init))
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  
        torch.cuda.empty_cache()  
