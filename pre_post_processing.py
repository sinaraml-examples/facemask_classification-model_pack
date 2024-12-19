from typing import List, BinaryIO
import io
import json
import numpy as np
import cv2
from PIL import Image

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

class PrePostProcessing:
    
    def prep_processing(self, file_stream, input_size=(32,32), mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375]):     
        pil_img=Image.open(file_stream)
        image_array = np.asarray(pil_img)
        # image_array = image_array[...,::-1] # to bgr        
        resized = cv2.resize(image_array, input_size).astype(np.float32)
        processing_img = (resized - np.array(mean)) / np.array(std)        
        processing_img = processing_img.transpose(2,0,1)[np.newaxis, ...].astype(np.float32)        
        return processing_img
    
    def post_processing(self, predicts, categories):   
        
        predict_class_ids = np.argmax(predicts[0], axis=1).tolist()
        predict_scores = np.round(softmax(predicts[0]).astype(np.float64), decimals=3).tolist()      
        predict_class_names = [categories.get(class_id) for class_id in predict_class_ids]
        predict_out = {"class_ids": predict_class_ids,
                       "class_names": predict_class_names,
                       "class_scores": predict_scores}
        return predict_out
    