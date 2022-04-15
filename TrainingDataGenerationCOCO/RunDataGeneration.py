
import CocoPanopticToSemanticMap as Generator

ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/train2017/" 
AnnotationDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/panoptic_train2017/" 
DataFile="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/panoptic_train2017.json" 


OutDir="/scratch/gobi2/seppel/Chemscape/FromOtherDataBase/COCO_Vessel_Related_Class/" 

VesselCats = [44,46,47,51,86] 
IgnoreCats = [70,50,64,196] 

x=Generator.Generator(ImageDir,AnnotationDir,OutDir, DataFile, VesselCats=VesselCats,IgnoreCats=IgnoreCats) 
x.Generate() 