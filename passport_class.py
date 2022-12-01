import cv2
import numpy as np
import math
import torch
import pandas as pd
# import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class Sts:
    def __init__(self):
        self.model_round = torch.hub.load('yolov5_master', 'custom', path='yolo5/sts_povorot.pt', source='local')
        self.model_detect = torch.hub.load('yolov5_master', 'custom', path='yolo5/sts_detect.pt', source='local')
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

    def rotate_image(self, mat, angle, point):
        height, width = mat.shape[:2] 
        image_center = (width/2, height/2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat

    def get_angle_rotation(self, centre, point, target_angle):
        new_point =(point[0] - centre[0], point[1] - centre[1])
        a,b = new_point[0], new_point[1]
        res = math.atan2(b,a)
        if (res < 0) :
            res += 2 * math.pi
        return (math.degrees(res)+target_angle) % 360

    def get_image_after_rotation(self, img):
        results = self.model_round(img)
        ('get_image_after_rotation',results)
        pd = results.pandas().xyxy[0]
        pd = pd.assign(centre_x = pd.xmin + (pd.xmax-pd.xmin)/2)
        pd = pd.assign(centre_y = pd.ymin + (pd.ymax-pd.ymin)/2)
        
        tmp = pd.loc[pd['name']=='svidetelstvo']
        
        N, V = None, None
        for index, row in tmp.iterrows(): 
            N = (row['centre_x'], row['centre_y'])
            break

        tmp = pd.loc[pd['name']=='ts']
        for index, row in tmp.iterrows(): 
            V = (row['centre_x'], row['centre_y'])
            break
        if N == None or V == None:
            return img
        
        angle = self.get_angle_rotation(N, V, 0)
        img = self.rotate_image(img, angle, N) 
        
        return img

    def get_crop(self, file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.get_image_after_rotation(image)
        image = self.get_image_after_rotation(image) # второй подряд поворот еще лучше выравнивает

        return image

    def yolo_5_snils(self, img):
        results = self.model_detect(img)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.6)[0])
        ob = pd.DataFrame()
        ob['class'] = df['name']
        ob['x'] = df['xmin']
        ob['y'] = df['ymin']
        ob['w'] = df['xmax']-df['xmin']
        ob['h'] = df['ymax']-df['ymin']
        oblasty = ob.values.tolist()
        return img, oblasty

    def oblasty_yolo_5_snils(self, image, box):
        oblasty = {}
        spissok = sorted(box, reverse=False, key=lambda x: x[2])
        for l in spissok:
            cat = l[0]
            y = int(l[2])
            x = int(l[1])
            h = int(l[4])
            w = int(l[3])
            ob = cat
            oblasty[ob] = image[self.zero(y - math.ceil(h * 0.1)):y + math.ceil(h * 1.1),
                      self.zero(x - math.ceil(w * 0.03)):x + math.ceil(w * 1.03)]
        return oblasty

    def zero(self,n):
        return n * (n > 0)

    def predict(self, file):
        result = {}

        image = self.get_crop(file)
        img, res = self.yolo_5_snils(image)
        oblasty = self.oblasty_yolo_5_snils(img, res)

        img = oblasty['sign']
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result['sign'] = generated_text

        img = oblasty['vin']
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result['vin'] = generated_text

        return result



    