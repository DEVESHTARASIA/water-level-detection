
import cv2
import numpy as np
import random
import CategoryDictionary as CatDic


def show(Im):
    cv2.imshow("show",Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()

CatName={}
CatName[1]='Vessel'
CatName[2]='V Label'
CatName[3]='V Cork'
CatName[4]='V Parts GENERAL'
CatName[5]='Ignore'
CatName[6]='Liquid GENERAL'
CatName[7]='Liquid Suspension'
CatName[8]='Foam'
CatName[9]='Gel'
CatName[10]='Solid GENERAL'
CatName[11]='Granular'
CatName[12]='Powder'
CatName[13]='Solid Bulk'
CatName[14]='Vapor'
CatName[15]='Other Material'
CatName[16]='Filled'

import os
import scipy.misc as misc
import random
import cv2
import json
import threading
import random

class Reader: 

    def __init__(self, MainDir=r"\ChemLabScapeDataset_Finished\Annotations\\", MaxBatchSize=100,MinSize=250,MaxSize=1000,MaxPixels=800*800*5,TrainingMode=True):# Initiate reader and define the main parameters for the data reader

        self.MaxBatchSize=MaxBatchSize 
        self.MinSize=MinSize 
        self.MaxSize=MaxSize 
        self.MaxPixels=MaxPixels 
        self.epoch = 0 
        self.itr = 0 
        self.ClassBalance=False

        self.AnnList = [] 
        self.AnnByCat = {} 

        for i in CatName:
           self.AnnByCat[CatName[i]]=[]
        uu=0

        for AnnDir in os.listdir(MainDir):
            SemDir=MainDir+"/"+AnnDir+r"//Semantic//"
            if not os.path.isdir(SemDir): continue
            self.AnnList.append(MainDir+"/"+AnnDir)
            for Name in os.listdir(SemDir):
                i=int(Name[:Name.find("_")])
                self.AnnByCat[CatName[i]].append(MainDir+"/"+AnnDir)
            uu+=1
        if TrainingMode:
            for i in self.AnnByCat: # suffle
                    np.random.shuffle(self.AnnByCat[i])
            np.random.shuffle(self.AnnList)

        self.CatNum={}
        for i in self.AnnByCat:
              print(str(i)+") Num Examples="+str(len(self.AnnByCat[i])))
              self.CatNum[i]=len(self.AnnByCat[i])
        print("Total=" + str(len(self.AnnList)))
        print("done making file list")
        iii=0
        if TrainingMode: self.StartLoadBatch() 
        self.AnnData=False

    def CropResize(self,Img, AnnMap,Hb,Wb):

        h,w,d=Img.shape
        Bs = np.min((h/Hb,w/Wb))
        if Bs<1 or Bs>1.5:  
            h = int(h / Bs)+1
            w = int(w / Bs)+1
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            for i in CatName:
                AnnMap[CatName[i]] = cv2.resize(AnnMap[CatName[i]], dsize=(w, h), interpolation=cv2.INTER_NEAREST)


        if np.random.rand()<0.6:
            if w>Wb:
                X0 = np.random.randint(w-Wb)
            else:
                X0 = 0
            if h>Hb:
                Y0 = np.random.randint(h-Hb)
            else:
                Y0 = 0

            Img=Img[Y0:Y0+Hb,X0:X0+Wb,:]
            for i in CatName:
                AnnMap[CatName[i]] = AnnMap[CatName[i]][Y0:Y0+Hb,X0:X0+Wb,:]

        if not (Img.shape[0]==Hb and Img.shape[1]==Wb):
            Img = cv2.resize(Img, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)
            for i in CatName:
                AnnMap[CatName[i]] = cv2.resize(AnnMap[CatName[i]], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)


        return Img,AnnMap
        # misc.imshow(Img)

    def Augment(self,Img,AnnMap,prob): 
        Img=Img.astype(np.float)
        if np.random.rand()<0.5: 
            Img=np.fliplr(Img)
            for i in CatName:
               AnnMap[CatName[i]] = np.fliplr(AnnMap[CatName[i]])
        if np.random.rand()<0.5:
            Img = Img[..., :: -1]


        if np.random.rand() < prob: 
            r=r2=(0.3 + np.random.rand() * 1.7)
            if np.random.rand() < prob*2:
                r2=(0.5 + np.random.rand())
            h = int(Img.shape[0] * r)
            w = int(Img.shape[1] * r2)
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            for i in CatName:
                AnnMap[CatName[i]] =  cv2.resize(AnnMap[CatName[i]], dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        # if np.random.rand() < prob/3: 
        #     noise = np.random.rand(Img.shape[0],Img.shape[1],Img.shape[2])*0.2+np.ones(Img.shape)*0.9
        #     Img *=noise
        #     Img[Img>255]=255
        #
        # if np.random.rand() < prob/3: # Gaussian blur
        #     Img = cv2.GaussianBlur(Img, (5, 5), 0)

        if np.random.rand() < prob*2:  
            Img = Img * (0.5 + np.random.rand() * 0.65)
            Img[Img>255]=255

        if np.random.rand() < prob:  
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img,AnnMap

    def LoadNext(self, pos, Hb=-1, Wb=-1): #
            if self.ClassBalance: 
                while (True):
                     CL=random.choice(self.AnnByCat.keys())
                     CatSize=len(self.AnnByCat[CL])
                     if CatSize>0: break

                Nim = np.random.randint(CatSize)
                AnnDir=self.AnnByCat[CL][Nim]
            else: 
                Nim = np.random.randint(len(self.AnnList))
                AnnDir=self.AnnList[Nim]
                CatSize=len(self.AnnList)

            Img = cv2.imread(AnnDir+"/"+"Image.png")  
            if (Img.ndim == 2):  
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  

            AnnDir+="/Semantic/"
            AnnMasks={}
            for i in CatName:
                path=AnnDir+"/"+str(i)+"_"+CatName[i]+".png"
                if os.path.exists(path):
                    AnnMasks[CatName[i]]=cv2.imread(path)  
                else:
                    AnnMasks[CatName[i]] = np.zeros(Img.shape)  

            Img,AnnMap=self.Augment(Img,AnnMasks,np.min([float(1000/CatSize)*0.5+0.06+1,1]))

            if not Hb==-1:
               Img, AnnMap = self.CropResize(Img, AnnMap, Hb, Wb)


            self.BImg[pos] = Img
            for i in CatName:

                CN=CatName[i]
                if CN == 'Ignore':
                   self.BIgnore[pos] = AnnMap[CN][:, :, 0]
                else:
                    self.BAnnMapsFR[CN][pos] = AnnMap[CN][:, :, 0]
                    self.BAnnMapsBG[CN][pos] = AnnMap[CN][:, :, 1]

    def StartLoadBatch(self):
        
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))


        
        
        self.BAnnMapsFR={}
        self.BAnnMapsBG = {}
        self.thread_list = []
        self.BIgnore = np.zeros([BatchSize, Hb, Wb], dtype=float)
        self.BImg = np.zeros([BatchSize, Hb, Wb,3], dtype=float)
        for i in CatName:
              CN=CatName[i]
              if CN=='Ignore': continue
              self.BAnnMapsFR[CN] = np.zeros([BatchSize, Hb, Wb], dtype=float)
              self.BAnnMapsBG[CN] = np.zeros([BatchSize, Hb, Wb], dtype=float)
        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNext,name="thread"+str(pos),args=(pos,Hb,Wb))
            self.thread_list.append(th)
            th.start()
        self.itr+=BatchSize

    def WaitLoadBatch(self):
            for th in self.thread_list:
                 th.join()


    def LoadBatch(self):


            self.WaitLoadBatch()
            Imgs=self.BImg
            Ignore=self.BIgnore
            AnnMapsFR = self.BAnnMapsFR
            AnnMapsBG = self.BAnnMapsBG
            self.StartLoadBatch()
            return Imgs, Ignore, AnnMapsFR, AnnMapsBG


    def LoadSingle(self):
        if self.itr>=len(self.AnnList):
            self.epoch+=1
            self.itr=0
        AnnDir = self.AnnList[self.itr]
        self.itr+=1
        Img = cv2.imread(AnnDir + "/" + "Image.png")  
        if (Img.ndim == 2):  
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  
        
        AnnDir += "/Semantic/"
        AnnMasks = {}
        for i in CatName:
            path = AnnDir + "/" + str(i) + "_" + CatName[i] + ".png"
            if os.path.exists(path):
                AnnMasks[CatName[i]] = cv2.imread(path)  
            else:
                AnnMasks[CatName[i]] = np.zeros(Img.shape)  

        Ignore=AnnMasks['Ignore'][:,:,0]
        del AnnMasks['Ignore']
        return Img, AnnMasks,Ignore, self.itr>=len(self.AnnList)
