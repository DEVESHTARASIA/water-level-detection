import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
import scipy.misc as misc

def rgb2id(color): 
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color[0] + 256 * color[1] + 256 * 256 * color[2]

class Generator:

    def __init__(self, ImageDir,AnnotationDir,OutDir, DataFile, AnnotationFileType="png", ImageFileType="jpg",UnlabeledTag=0,VesselCats=[44,46,47,51,86],IgnoreCats=[70,81,81,50]):

        self.ImageDir=ImageDir 
        self.AnnotationDir=AnnotationDir 
        self.AnnotationFileType=AnnotationFileType 
        self.ImageFileType=ImageFileType 
        self.DataFile=DataFile 
        self.UnlabeledTag=UnlabeledTag 

        self.Outdir=OutDir
        self.SemanticDir = OutDir + "/SemanticMaps/"
        self.OutImageDir = OutDir + "/Image/"
        if not os.path.exists(OutDir): os.mkdir(OutDir)
        if not os.path.exists(self.SemanticDir):
            os.mkdir(self.SemanticDir)
        
        if not os.path.exists(self.OutImageDir):
            os.mkdir(self.OutImageDir)


        self.VesselCats=VesselCats 
        self.IgnoreCats=IgnoreCats

        with open(DataFile) as json_file:
            self.AnnData=json.load(json_file)

        self.FileList=[]
        for FileName in os.listdir(AnnotationDir):
            if AnnotationFileType in FileName:
                self.FileList.append(FileName)


    def GetAnnnotationData(self, AnnFileName):
            for item in self.AnnData['annotations']:  
                if (item["file_name"] == AnnFileName):
                    return(item['segments_info'])

    def GetCategoryData(self,ID):
                for item in self.AnnData['categories']:
                    if item["id"]==ID:
                        return item["name"],item["isthing"]
                return "", 0

    def GenerateSemanticMap(self,Ann,Ann_name):
        AnnList = self.GetAnnnotationData(Ann_name)
        h,w=Ann.shape
        SemanticMap = np.zeros([h,w],dtype=np.uint8) 
        ROIMap = np.zeros([h, w], dtype=np.uint8) 
        for an in AnnList:
            ct=an["category_id"]
            if ct in  self.VesselCats:
                            SemanticMap[Ann == an['id']] = 1
            if not (ct in  self.IgnoreCats):
                            ROIMap[Ann == an['id']] = 1
        return SemanticMap, ROIMap


    def Generate(self):


           for f,Ann_name in enumerate(self.FileList): 
                print(str(f) + ")" + Ann_name)
                Ann = cv2.imread(self.AnnotationDir + "/" + Ann_name)  
                Ann = Ann[..., :: -1]
                self.AnnColor = Ann
                Ann = rgb2id(Ann)
                H, W = Ann.shape
                SemanticMap, ROIMask = self.GenerateSemanticMap(Ann, Ann_name)  

                Ann = cv2.imread(self.AnnotationDir + "/" + Ann_name)  

                if not (1 in SemanticMap):continue
                from shutil import copyfile

                copyfile(self.ImageDir + "/" + Ann_name.replace(".png",".jpg"), self.OutImageDir+ "/" + Ann_name.replace(".png",".jpg"))

                SemanticMap=np.expand_dims(SemanticMap,axis=2)
                ROIMask=np.expand_dims(ROIMask,axis=2)

                SemanticMap= np.concatenate([SemanticMap, ROIMask,SemanticMap],axis=2)
                cv2.imwrite(self.SemanticDir + "/" + Ann_name, SemanticMap.astype(np.uint8))
