import cv2
import numpy as np
import math
import torch
import pandas as pd
# import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import warnings
warnings.filterwarnings('ignore')


class Snils:
    def __init__(self):
        self.reader = easyocr.Reader(['ru'],
                        model_storage_directory='EasyOCR/model',
                        user_network_directory='EasyOCR/user_network',
                        recog_network='custom_example',
                        gpu=False)

        self.model_round = torch.hub.load('yolov5_master', 'custom', path='yolo5/sts_povorot.pt', source='local')
        self.model_detect = torch.hub.load('yolov5_master', 'custom', path='yolo5/snils_detect.pt', source='local')
        self.model_numbers = torch.hub.load('yolov5_master', 'custom', path='yolo5/yolov5m.pt', source='local')

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

        tmp = pd.loc[pd['name']=='Svidetelstvo_test']

        N, V = None, None
        for index, row in tmp.iterrows():
            N = (row['centre_x'], row['centre_y'])
            break
        #получим координаты верха, там где печать
        tmp = pd.loc[pd['name']=='Svidetelstvo_train']
        for index, row in tmp.iterrows():
            V = (row['centre_x'], row['centre_y'])
            break
        if N == None or V == None: #похоже там нет нужных нам строк
            return img

        angle = self.get_angle_rotation(N, V, 180)

        img = self.rotate_image(img, angle, N)  #вращаем той процедурой, что выше
        return img

    def crop_img(self, img):
        results = self.model_round(img)
        pd = results.pandas().xyxy[0]
        try:
        #определяем координаты вырезки
            x1 =int(pd.xmin.min())
            x2 = int(pd.xmax.max())

            y1 = int(pd.ymin.min())
            y2 = int(pd.ymax.max())

            img = img[y1:y2,x1:x2]
        except Exception:
            print(pd)
        return img

    def get_crop(self, file):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = self.get_image_after_rotation(image)
        image = self.get_image_after_rotation(image) #второй подряд поворот еще лучше выравнивает

        return image

    def zero(self,n):
        return n * (n > 0)

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

    def Intersection(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return interArea

    def IoU(self, boxA, boxB):
        # compute the area of intersection rectangle
        interArea = self.Intersection(boxA, boxB)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # calculation iou
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def numbers_detect(self, img):

        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = self.model_numbers(image)

        final_res = result.pandas().xyxy[0]
        result = result.pandas().xyxy[0]

        # delete intersection objects
        for obj1 in range(len(result)):
            for obj2 in range(len(result)):

                boxA = result.iloc[obj1, :4].values
                boxB = result.iloc[obj2, :4].values

                if boxA.all() != boxB.all():
                    if self.IoU(boxA, boxB) > 0.2:
                        if result.iloc[obj1, 4] > result.iloc[obj2, 4]:
                            final_res = final_res[final_res.xmin != result.iloc[obj2, 0]]

                        else:
                            final_res = final_res[final_res.xmin != result.iloc[obj1, 0]]
        return final_res


    def recognition_slovar_snils(self, oblasty):
        data = {}
        data['snils'] = []
        d = {}
        fio=''
        for i, v in oblasty.items():
            image = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)

            # plt.imshow(image)
            # plt.show()

            if 'number_strah' in i:
                # result = self.reader.readtext(image, allowlist='0123456789- ')

                numbs = self.numbers_detect(image).sort_values(by=['xmin']).iloc[:, 5].values
                res_numbs = ''.join(str(e) for e in numbs)
                res_numbs = f'{res_numbs[:3]}-{res_numbs[3:6]}-{res_numbs[6:9]}_{res_numbs[-2:]}'
                result = [([], f'{res_numbs}', 0.0)]

            elif 'fio' in i:
                result = self.reader.readtext(image,
                                         allowlist='АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ- ')
            pole = ''
            for k in range(len(result)):
                pole = pole + ' ' + str(result[k][1]).replace(' ', '').replace('_', ' ')
            if pole:
                pole = pole.strip()
                if 'fio' in i:
                    fio = fio + ' ' + pole.upper().strip()
                else:
                    d[i.split('.', 1)[0]] = pole.upper().strip()
        d['fio'] = fio.strip()
        data['snils'].append(d)
        return data['snils']

    def detect_snils(self,photo):
        pole = ['number_strah','fio1','fio2','fio3']

        croped = self.get_crop(photo)
        if croped != '':
            img, detect = self.yolo_5_snils(croped)
            obl = self.oblasty_yolo_5_snils(img, detect)
            rec = self.recognition_slovar_snils(obl)
            key = list(rec[0].keys())
            value = list(rec[0].values())
            if set(key) == set(pole):
                if '' in value:
                    flag = 1
                else:
                    flag = 0
            else:
                flag = 1
            return rec[0], flag
        else:
            rec = {}
            return rec, 1


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

        # получим координаты верха, там где печать
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



    