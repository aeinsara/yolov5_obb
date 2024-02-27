import os
import cv2
import json
import torch
import logging
import detectron2
import numpy as np
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from sklearn.metrics.pairwise import cosine_similarity
from utils.datasets import create_dataloader
from utils.general import check_dataset, colorstr
from utils.rboxs_utils import poly2hbb, rbox2poly


from resnet import resnet101 #........................................................
from boxes import Boxes #............
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from tqdm import tqdm

from PIL import Image 
from numpy import asarray

from sklearn.cluster import KMeans

class PrototypicalCalibrationBlock:

    def __init__(self, data):
        super().__init__()
        # self.cfg = cfg
        self.device = torch.device("cpu")
        self.imagenet_model = self.build_model()
        # self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        
        # self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
        
        #.................... sara added ..........................
        self.alpha = 0.5 #self.alpha = 0.5
        print("self.alpha in pcb ============ ", self.alpha)
        # dataset = build_dataset(cfg.data.train) # ......... train_fewshot
        # self.dataloader = build_dataloader(
        #     dataset,
        #     samples_per_gpu=1,
        #     workers_per_gpu=cfg.data.workers_per_gpu,
        #     dist=False,
        #     shuffle=False)
        
        task = 'train'   
        data = check_dataset(data)  # check
        imgsz = 1024
        batch_size = 16
        stride = 2
        names = {1: 'plane',
        2: 'baseball-diamond', 
        3: 'bridge', 
        4: 'ground-track-field',
        5: 'small-vehicle',
        6: 'large-vehicle',
        7: 'ship', 
        8: 'tennis-court',
        9: 'basketball-court',
        10: 'storage-tank', 
        11: 'soccer-ball-field',
        12: 'roundabout',
        13: 'harbor', 
        14: 'swimming-pool',
        15: 'helicopter',
        16: 'container-crane'} 
        
        single_cls = False
        pad = 0.5
        pt = True
        workers = 8
        self.dataloader_train = create_dataloader(data[task], imgsz, batch_size, stride, names, single_cls, pad=pad, rect=pt, workers=workers, prefix=colorstr(f'{task}: '))[0] 
        task = 'val'   
        # self.dataloader_val = create_dataloader(data[task], imgsz, batch_size, stride, names, single_cls, pad=pad, rect=pt, workers=workers, prefix=colorstr(f'{task}: '))[0] 
        
    
        print("calibration_layer init ... ")
        
        self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignRotated")
        #.................... end ..........................
        self.prototypes = self.build_prototypes()

        self.exclude_cls = list(range(0, 12))

        print('self.exclude_cls == ', self.exclude_cls)

    def build_model(self):
        #.................... sara added ..........................
        imagenet_model = resnet101()
        state_dict = torch.load("/mnt/1tra/hajizadeh/yolov5_obb/weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth")
        #.................... end ..........................    
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model
    
    def build_prototypes(self):
        # pkl_file = 'prototypes_10shot_aug.pkl'  
        # prototypes_dict = pickle.load(open(pkl_file, 'rb'))
        # print(f'load {pkl_file} ... ')
        pkl_file_kmeans = 'prototypes_10shot_aug_kmeans.pkl'
        prototypes_dict = pickle.load(open(pkl_file_kmeans, 'rb'))
        print(f'load {pkl_file_kmeans} ... ')
        return prototypes_dict
        all_features, all_labels = [], []
        for index in tqdm(range(len(self.dataloader_train.dataset))):
            inputs = [self.dataloader_train.dataset[index]]
            assert len(inputs) == 1
            # inputs[0] # 0: , 1:image , 2:path, 3:size ((534, 1024), ((1.0, 1.0), (1.0, 246.0))
            im, target, path, shape = inputs[0]
            # target (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) θ ∈ [-pi/2, pi/2)
            # shapes (tensor): (b, [(h_raw, w_raw), (hw_ratios, wh_paddings)])
            # img = cv2.imread(path)  # BGR
            # cv2.imwrite('out/_'+path.split('/')[-1], img)
            img = im.permute(1,2,0).cpu().numpy()
            # cv2.imwrite('out/'+path.split('/')[-1], img)
            img_h, img_w = img.shape[0], img.shape[1]

            boxes = []
            gt_classes = []
            for tar in target:
                theta = tar[6]
                
                x1, y1, x2, y2 = self.xywh2xyxy(tar[2:6])
                cx, cy, w, h = tar[2:6]

                gt_classes.append(tar[1].item())
                
                # boxes.append([x1,y1,x2,y2, theta])
                boxes.append([cx, cy, w, h, theta])
            boxes = [Boxes(boxes)]
            
            # extract roi features
            features = self.extract_roi_features(img, boxes)
            all_features.append(features.cpu().data)

            # gt_classes = [inputs[0]['gt_labels'].data]
            # all_labels.append(gt_classes[0].cpu().data)

            all_labels.extend(gt_classes)

        # print('all_labels ==== ', all_labels)
        # print('\ngt_classes === ', gt_classes)
        # concat
        all_features = torch.cat(all_features, dim=0)
        # print('all_features.shape ====== ', all_features.shape)
        # all_labels = [tensor.item() for sublist in all_labels for tensor in sublist]
        # all_labels = torch.tensor(all_labels)
        
        # all_labels = torch.cat(all_labels, dim=0)
        assert all_features.shape[0] == len(all_labels)

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features[i].unsqueeze(0))
            # print('all_features[i].shape , all_features[i].unsqueeze(0).shape ==== ', all_features[i].shape, all_features[i].unsqueeze(0).shape)
        print('label in calibration_layer in build_prototypes ...')
        prototypes_dict = {}
              
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)
            # print('label , features.shape = ', label, features.shape)
        pickle.dump(prototypes_dict, open(pkl_file, 'wb'))
        pickle.dump(features_dict, open(pkl_file_features, 'wb'))
        return prototypes_dict

    def xywh2xyxy(self, tar):
        cx = tar[0]
        cy = tar[1]
        l = tar[2]
        s = tar[3]

        x1 = cx - s/2 # top left x
        y1 = cy - l/2 # top left y
        x2 = cx + s/2 # bottom right x
        y2 = cy + l/2 # bottom right y
        
        return x1,y1,x2,y2
        
   
    def extract_roi_features(self, img, boxes):
        """
        :param img:
        :param boxes:
        :return:
        """
        mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
        std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255. - mean) / std]
        images = ImageList.from_tensors(images, 0)
        
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW
        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)
        activation_vectors = self.imagenet_model.fc(box_features) 
        
        # print('box_features.shape == ', box_features.shape)
        # print('conv_feature.shape == ', conv_feature.shape)
        # print('activation_vectors.shape == ', activation_vectors.shape)

        return box_features

    def execute_calibration(self, path, inputs, dts, img):
        ID2CLASS = {
        1: 'plane',
        2: 'baseball-diamond', 
        3: 'bridge', 
        4: 'ground-track-field',
        5: 'small-vehicle',
        6: 'large-vehicle',
        7: 'ship', 
        8: 'tennis-court',
        9: 'basketball-court',
        10: 'storage-tank', 
        11: 'soccer-ball-field',
        12: 'roundabout',
        13: 'harbor', 
        14: 'swimming-pool',
        15: 'helicopter',
        16: 'container-crane'}
        img = img.permute(1,2,0).cpu().numpy().astype(np.uint8)
        img_copy = np.copy(img)
        img = np.ascontiguousarray(img_copy)
        # img = cv2.imread(path)
        polyInp = rbox2poly(inputs[:,1:])
#         for i in range(len(inputs)):
#              # clsid cx cy l s theta
#             x1, y1, x2, y2 = self.xywh2xyxy(inputs[i][1:5].tolist())
#             points = []
#             points.append((int(polyInp[i].tolist()[0]), int(polyInp[i].tolist()[1])))
#             points.append((int(polyInp[i].tolist()[2]), int(polyInp[i].tolist()[3])))
#             points.append((int(polyInp[i].tolist()[4]), int(polyInp[i].tolist()[5])))
#             points.append((int(polyInp[i].tolist()[6]), int(polyInp[i].tolist()[7])))

#             points = np.array(points)
#             category = ID2CLASS[int(inputs[i][0].tolist())+1]
            
#             color = (255, 0, 0)  # Green color for bounding boxes
#             thickness = 2
#             img = cv2.circle(img, (int(inputs[i][0]), int(inputs[i][1])), 2, color, thickness)
#             img = cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)

#             cv2.putText(img, category, (int(x1), int(y1)-5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        pred_boxes = []
        scores = []
        pred_classes = []
        poly = rbox2poly(dts[:,:5])
        
        for i in range(len(dts)):
            # print(f'dts[i] === ',i,  dts[i])
            # x1, y1, x2, y2 = self.xywh2xyxy(dts[i][:4].tolist())
            cx, cy, w, h = dts[i][:4].tolist()
            # print('poly == ', poly[i].tolist())
#             if (dts[i][5]).tolist() > 0.3:
#                 points = []
#                 points.append((int(poly[i].tolist()[0]), int(poly[i].tolist()[1])))
#                 points.append((int(poly[i].tolist()[2]), int(poly[i].tolist()[3])))
#                 points.append((int(poly[i].tolist()[4]), int(poly[i].tolist()[5])))
#                 points.append((int(poly[i].tolist()[6]), int(poly[i].tolist()[7])))

#                 points = np.array(points)
#                 category = ID2CLASS[int(dts[i][6].tolist())+1]
#                 color = (0, 255, 0)  # Green color for bounding boxes
#                 thickness = 2
#                 img = cv2.circle(img, (int(dts[i][0]), int(dts[i][1])), 2, color, thickness)
#                 img = cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)

#                 cv2.putText(img, category, (int(x1), int(y1)-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
#             cv2.imwrite('out/'+path.split('/')[-1], img)
            
            theta = dts[i][4].tolist()
            pred_boxes.append([cx, cy, w, h, theta])
            scores.append((dts[i][5]).tolist())
            pred_classes.append(int(dts[i][6].tolist()))
        assert len(pred_boxes) == len(scores)  

        boxes = [Boxes(pred_boxes)]
        #.................... end ..........................
        if len(boxes[0]) == 0:
            return dts         
        features = self.extract_roi_features(img, boxes)

        for i in range(len(boxes[0])):
            tmp_class = int(dts[i][6].tolist())
            if tmp_class in self.exclude_cls:
                continue
            # print('tmp_class == ', tmp_class)
            tmps_cos = []
            for prototype in self.prototypes[tmp_class]:
                tmps_cos.append(cosine_similarity(features[i].cpu().data.numpy().reshape((1, -1)),
                                        prototype.cpu().data.numpy().reshape((1, -1)))[0][0])
            tmp_cos = max(tmps_cos)
            # print(tmps_cos, tmp_cos)
            # print(features[i].cpu().data.numpy().reshape((1, -1)).shape,
            #                             self.prototypes[tmp_class].cpu().data.numpy())
            # tmp_cos = cosine_similarity(features[i].cpu().data.numpy().reshape((1, -1)),
            #                             self.prototypes[tmp_class].cpu().data.numpy())[0][0]
            # print('tmp_cos === ', tmp_cos)
            score = (dts[i][5]).tolist() * self.alpha + tmp_cos * (1 - self.alpha)
            # print('old , new score, tmp_cos ==== ', (dts[i][5]).tolist(), score, tmp_cos)
            dts[i][5] = score

        return dts

    def clsid_filter(self):
        exclude_ids = list(range(0, 12))
        


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
