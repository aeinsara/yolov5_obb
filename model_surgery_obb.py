# import os
# import torch
# import argparse


# def surgery_loop(args, surgery):

#     save_name = args.tar_name + '_' + ('remove' if args.method == 'remove' else 'surgery') + '.pt'
#     save_path = os.path.join(args.save_dir, save_name)
#     os.makedirs(args.save_dir, exist_ok=True)

#     # ckpt = torch.hub.load('ultralytics/yolov5', 'custom', path=args.src_path, force_reload=True) 
    
#     # ckpt = torch.load(args.src_path)   
#     print(args.src_path)
#     ckpt = torch.load(args.src_path)
#     print("model === ",ckpt.keys(), '\n')
#     # print("ckpt['model'].float().state_dict().keys() === ", ckpt['model'].float().state_dict().keys(),'\n')
#     state_dict = ckpt['model'].float().state_dict() 

#     if 'scheduler' in ckpt:
#         print('del ckpt[scheduler]')
#         del ckpt['scheduler']
#     # if 'optimizer' in ckpt:
#     #     print('del ckpt[optimizer]')
#     #     del ckpt['optimizer']
#     # if 'epoch' in ckpt:
#     #     print('ckpt[epoch]:: ',ckpt['epoch'])
#     #     ckpt['epoch'] = 0
        
#     # model.24.m.0.bias  :  torch.rand([603])
#     # model.24.m.1.weight  :  torch.rand([603, 128, 1, 1])
#     # model.24.m.1.bias  :  torch.rand([603])
#     # model.24.m.2.weight  :  torch.rand([603, 256, 1, 1])
#     # model.24.m.2.bias  :  torch.rand([603])
#     for i in state_dict.keys():
#         print(i, " : ", state_dict.get(i).shape)
            
            

#     if args.method == 'remove':
#         for param_name in args.param_name:
#             del state_dict[param_name + '.weight']
#             print(param_name, '.weight deleted!')
#             if param_name+'.bias' in state_dict:
#                 del state_dict[param_name+'.bias']
#                 print(param_name, '.bias deleted!')
#     elif args.method == 'randinit':
#         for param_name in args.param_name:
#             tar_size = 603
#             surgery(param_name, True, tar_size, ckpt)
#             surgery(param_name, False, tar_size, ckpt)
#     else:
#         raise NotImplementedError
        
#     state_dict = ckpt['model'].float().state_dict() 
#     for i in state_dict.keys():
#         print(i, " : ", state_dict.get(i).shape)

#     torch.save(ckpt, save_path)
#     print('save changed ckpt to {}'.format(save_path))


# def main(args):
#     """
#     Either remove the final layer weights for fine-tuning on novel dataset or
#     append randomly initialized weights for the novel classes.
#     """
#     def surgery(param_name, is_weight, tar_size, ckpt):
#         print('in surgery', tar_size)
#         state_dict = ckpt['model'].float().state_dict() 
#         weight_name = param_name + ('.weight' if is_weight else '.bias')
#         pretrained_weight = state_dict[weight_name]
#         print('pretrained_weight weight_name:: ', weight_name, pretrained_weight.shape)
#         prev_cls = pretrained_weight.size(0)
#         # if 'fc_cls' in param_name:
#         #     prev_cls -= 1
#         if is_weight:
#             feat_size = pretrained_weight.size(1)
#             new_weight = torch.rand((tar_size, feat_size, 1, 1))
#             torch.nn.init.normal_(new_weight, 0, 0.01)
#         else:
#             new_weight = torch.zeros(tar_size)
#         print("new_weight", new_weight.shape)

#         new_weight[:prev_cls] = pretrained_weight[:prev_cls]
#         # if 'fc_cls' in param_name:
#         #     new_weight[-1] = pretrained_weight[-1]  # bg class
#         state_dict[weight_name] = new_weight
#         # for i in state_dict.keys():
#         #     print(i, " : ", state_dict.get(i).shape)
#         ckpt['model'].float().load_state_dict(state_dict, strict=False)
#     surgery_loop(args, surgery)

# '''
# # ------------------------------ Model Preparation -------------------------------- #
# python3 tools/model_surgery.py --dataset dota --method remove                         \
#     --src-path ${SAVEDIR}/defrcn_det_r101_base/model_final.pth                        \
#     --save-dir ${SAVEDIR}/defrcn_det_r101_base
# '''
    
# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='dota', choices=['voc', 'coco', 'dota'])
#     parser.add_argument('--src-path', type=str, default='/mnt/1tra/hajizadeh/yolov5_obb_1/runs/train/base3/weights/best.pt', help='Path to the main checkpoint')
#     parser.add_argument('--save-dir', type=str, default='/mnt/1tra/hajizadeh/yolov5_obb_1/runs/train/base3/weights/',help='Save directory')
#     parser.add_argument('--method', choices=['remove', 'randinit'], required=True,
#                         help='remove = remove the final layer of the base detector. '
#                              'randinit = randomly initialize novel weights.')
# #     parser.add_argument('--param-name', type=str, nargs='+', help='Target parameter names',
# #                         default=['roi_heads.box_predictor.cls_score', 'roi_heads.box_predictor.bbox_pred']) 
#     parser.add_argument('--param-name', type=str, nargs='+', help='Target parameter names',
#                         default=['model.24.m.0', 'model.24.m.1', 'model.24.m.2'])
# #     , 'roi_head.bbox_head.fc_reg'
# #roi_head.bbox_head.shared_fcs.0
# #roi_head.bbox_head.shared_fcs.1
# #............ imp
#     parser.add_argument('--tar-name', type=str, default='model_reset', help='Name of the new ckpt')
#     args = parser.parse_args()

# #     if args.dataset == 'coco':
# #         NOVEL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
# #         BASE_CLASSES = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
# #                         39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
# #                         61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
# #         ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES)
# #         IDMAP = {v: i for i, v in enumerate(ALL_CLASSES)}
# #         TAR_SIZE = 80
# #     elif args.dataset == 'voc':
# #         TAR_SIZE = 20
# #     elif args.dataset == 'dota':
# #         NOVEL_CLASSES = [13,14,15,16]
# #         BASE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12]  #............... add 13,14,15,16

# #         ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES) #+ NOVEL_CLASSES
# #         IDMAP = {v: i for i, v in enumerate(ALL_CLASSES)}
# #         print('IDMAP === ', IDMAP)
# #         TAR_SIZE = 16
# #     else:
# #         raise NotImplementedError

#     main(args)

import torch
import torch.nn as nn

def adjust_weights(state_dict, target_size, IDMAP, BASE_CLASSES):
    # for i in ckpt['model'].float().state_dict().keys():
        # if '24' in i:
        #      print(i, " : ", ckpt['model'].float().state_dict().get(i).shape)  
    for key in state_dict.keys():
        if '24' in key:
            if key.endswith('.weight') or key.endswith('.bias'):
                weight = state_dict[key]
                original_size = 591 #weight.size(0)
                print(original_size , target_size)
                # if original_size < target_size:
                # Create a new tensor with the target size
                print('(target_size,), weight.shape[1:] ===========> ',(target_size,), weight.shape[1:])
                new_weight = torch.rand((target_size,) + weight.shape[1:])
                print('new_weight.shape, weight.shape===========> ',new_weight.shape, weight.shape)
                # Copy the original weights to the new tensor
                # print(original_size/3 , original_size/3 + 4, 2*(original_size/3)+4,2*(original_size/3) + 8, 3*(original_size/3)+8)
                new_weight[:int(original_size/3)] = weight[:int(original_size/3)]
                new_weight[int(original_size/3) + 4 : 2*int(original_size/3)+4] = weight[int(original_size/3) : 2*int(original_size/3)]
                new_weight[2*int(original_size/3) + 8 : 3*int(original_size/3)+8] = weight[2*int(original_size/3) : original_size]
                
                      
                if torch.equal(new_weight[2*int(original_size/3) + 8 : 3*int(original_size/3)+8], weight[2*int(original_size/3) : original_size]):
                    print('okey')
                
                # new_weight[:original_size] = weight
                # Update the state dictionary with the new tensor
                state_dict[key] = new_weight

                # for idx, c in enumerate(BASE_CLASSES):
                #     new_weight = torch.rand((target_size,) + weight.shape[1:])
                #     new_weight[(185 + IDMAP[c]) * 3: (185 + IDMAP[c]) * 3] = weight[idx*27: idx*28]
                #     state_dict[key] = new_weight
    for i in ckpt['model'].float().state_dict().keys():
        if '24' in i:
             print(i, " : ", ckpt['model'].float().state_dict().get(i).shape)           
    return ckpt
    

src_path = '/mnt/1tra/hajizadeh/yolov5_obb/runs/train/base3/weights/best.pt'
save_path = '/mnt/1tra/hajizadeh/yolov5_obb/runs/train/base3/weights/model_surgery.pt'

NOVEL_CLASSES = [13, 14, 15, 16]
BASE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES)
IDMAP = {v: i for i, v in enumerate(ALL_CLASSES)}
# Example usage
ckpt = torch.load(src_path)
target_size = 603
# print(ckpt)
# for k in ckpt.keys():
#     print('keys :::: ', k)
ckpt['model'].model[24].m[0] = nn.Conv2d(64, 603, kernel_size=(1, 1), stride=(1, 1))
ckpt['model'].model[24].m[1] = nn.Conv2d(128, 603, kernel_size=(1, 1), stride=(1, 1))
ckpt['model'].model[24].m[2] = nn.Conv2d(256, 603, kernel_size=(1, 1), stride=(1, 1))

if not(ckpt['ema'] is None):
    ckpt['ema'] = None
    ckpt['epoch'] = -1
    # ckpt['ema'].model[24].m[0] = nn.Conv2d(64, 603, kernel_size=(1, 1), stride=(1, 1))
    # ckpt['ema'].model[24].m[1] = nn.Conv2d(128, 603, kernel_size=(1, 1), stride=(1, 1))
    # ckpt['ema'].model[24].m[2] = nn.Conv2d(256, 603, kernel_size=(1, 1), stride=(1, 1))
# Adjust the weights for the specified layers
ckpt = adjust_weights(ckpt['model'].float().state_dict(), target_size, IDMAP, BASE_CLASSES)
    

# print(ckpt)

# Save the modified checkpoint
torch.save(ckpt, save_path)