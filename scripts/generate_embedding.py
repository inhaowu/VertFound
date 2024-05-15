import cv2
import json
import numpy as np
import torch
import os
from vertfound.modeling.segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
from argparse import ArgumentParser

def main(args):
    # 载入模型
    sam = sam_model_registry["vit_b"](checkpoint="../sam_checkpoints/sam_vit_b_01ec64.pth")
    sam = sam.cuda()
    predictor = SamPredictor(sam)

    # 读取COCO JSON文件
    def load_coco_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    train_json = load_coco_json(args.train_file)
    test_json = load_coco_json(args.test_file)

    # 给定image_id，从两个JSON文件中查找annotations
    def find_annotations_and_classes(image_id):
        annotations = []
        classes = []
        for item in train_json['annotations']:
            if str(item['image_id']) == image_id.split('.')[0]:
                annotations.append(item)
                classes.append(item['category_id'])
        for item in test_json['annotations']:
            if str(item['image_id']) == image_id.split('.')[0]:
                annotations.append(item)
                classes.append(item['category_id'])
        return annotations, classes

    # 读取文件夹中所有图片
    folder_path = args.image_path

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    image_files = sorted(image_files, key=lambda x : int(x.split('.')[0]))

    file_counter = 0
    cls=[]
    save_path = args.out_dir
    for image_file in tqdm(image_files, desc="Processing images"):
        # print(image_file)
        if image_file != '15.png': continue

        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path) # (512, 512, 3)
        # 设置图片
        predictor.set_image(image)

        all_embeddings = []
        all_classes = []
        all_boxes = []
        # 尝试从两个JSON文件中获取annotations
        annotations, classes = find_annotations_and_classes(image_file)
        
        if annotations:
            for annotation, cls in zip(annotations, classes):
                bbox = annotation['bbox']
                input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

                all_boxes.append(bbox)
                # 获取embeddings
                _, _, _, image_embeddings = predictor.predict(box=input_box, multimask_output=False)
                
                image_embeddings = image_embeddings[0,:,:] #(1, 1, 256)
                # img=image_embeddings.cpu().numpy().mean(0)
                # img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
                # cv2.imwrite('gray_image.png', img)
                # gray_image = cv2.imread('gray_image.png', cv2.IMREAD_GRAYSCALE)
                # color_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
                # resized_image = cv2.resize(color_image, (256, 256))
                # cv2.imwrite('mean.png',resized_image)
                # for i in range(image_embeddings.shape[0]):
                #     name='plot/' + str(i) + '.png'
                #     img = cv2.normalize(image_embeddings[i].cpu().numpy(),None,0,255,cv2.NORM_MINMAX)
                #     cv2.imwrite('gray_image.png', img)
                #     gray_image = cv2.imread('gray_image.png', cv2.IMREAD_GRAYSCALE)
                #     color_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
                #     # rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                #     resized_image = cv2.resize(color_image, (256, 256))
                #     cv2.imwrite(name,resized_image)
                # 添加到列表中
                all_embeddings.append(image_embeddings)
                all_classes.append(cls)
        
        # 将所有目标的embeddings堆叠成一个tensor
        if all_embeddings:
            stacked_embeddings = torch.stack(all_embeddings)
            img1 = np.sum(stacked_embeddings.cpu().numpy(), axis=0)
            # img=img.mean(0)
            # img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
            # cv2.imwrite('gray_image.png', img)
            # gray_image = cv2.imread('gray_image.png', cv2.IMREAD_GRAYSCALE)
            # color_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
            # resized_image = cv2.resize(color_image, (256, 256))
            # cv2.imwrite('mean.png',resized_image)
            for i in range(img1.shape[0]):
                name='plot/' + str(i) + '.png'
                img = cv2.normalize(img1[i],None,0,255,cv2.NORM_MINMAX)
                cv2.imwrite('gray_image.png', img)
                gray_image = cv2.imread('gray_image.png', cv2.IMREAD_GRAYSCALE)
                color_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
                # rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                resized_image = cv2.resize(color_image, (256, 256))
                cv2.imwrite(name,resized_image)
            data_dict = {
                'mask_tokens': stacked_embeddings.cpu().numpy(),
                'classes': all_classes,
                'instances': all_boxes ,
            }
            # 保存为.pth文件，例如0.pth, 1.pth, 2.pth 等
            torch.save(data_dict, os.path.join(save_path, f'{file_counter}.pth'))
            file_counter += 1
        else:
            print(f"No embeddings to save for {image_file}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_file', default='datasets/vert/annotations/train.json', help='train file path.')
    parser.add_argument('--test_file', default='datasets/vert/annotations/test.json', help='test file path.')
    parser.add_argument('--image_path', default='datasets/vert/all_images', help='all images\' path.')
    parser.add_argument('--out_dir', default='datasets/datasets_mask_tokens_vit_b/vert', help='generate sam embeddings of particular dataset.')
    
    args = parser.parse_args()
    
    main(args)