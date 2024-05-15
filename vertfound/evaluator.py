import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator
import itertools
import numpy as np
import torch
from vertfound.util.metrics import ConfusionMatrix

class VertEvaluator(COCOEvaluator):
    def evaluate(self, img_ids=None):
        result = super().evaluate(img_ids)

        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        result['IRA'] = VertEvaluator.calc_IRA(predictions, self._coco_api)
        result['IDR'] = VertEvaluator.calc_IDR(predictions, self._coco_api)
        VertEvaluator.calc_confusion(predictions, self._coco_api)
        return result
    
    @classmethod
    def calc_IRA(cls, predictions, coco_api):
        corrects = 0
        tot_img = len(coco_api.getImgIds())

        id2pred = {}
        for pred in predictions:
            image_id, instances = pred['image_id'], pred['instances']
            if image_id not in id2pred:
                id2pred[image_id] = {}
            for ins in instances:
                bbox = ins['bbox']
                if str(bbox) not in id2pred[image_id]:
                    id2pred[image_id][str(bbox)] = []
                id2pred[image_id][str(bbox)].append(ins)

        def is_correct(preds, image_id):
            ann_ids = coco_api.getAnnIds(imgIds=[image_id])
            annos = coco_api.loadAnns(ann_ids)

            cls_anno = {anno['category_id'] - 1 : np.array(anno['bbox'], dtype=np.int64) for anno in annos}

            for pred in preds:
                id = pred['category_id']
                if cls_anno.get(id) is None:
                    return False
                if np.mean(np.array(pred['bbox'], dtype=np.int64) - cls_anno.get(id)) > 2:
                    return False
            return True

        for id in id2pred.keys():
            for box in id2pred[id].keys():
                ind = np.argmax([id2pred[id][box][i]['score'] for i in range(len(id2pred[id][box]))])
                id2pred[id][box] = id2pred[id][box][ind]

        for id in id2pred.keys():
            preds = list(id2pred[id].values())
            if is_correct(preds, id):
                corrects += 1
        
        return (corrects / tot_img) * 100 if tot_img > 0 else 0
    
    @classmethod
    def calc_IDR(cls, predictions, coco_api):
        corrects = 0
        tot_box = 0

        id2pred = {}
        for pred in predictions:
            image_id, instances = pred['image_id'], pred['instances']
            if image_id not in id2pred:
                id2pred[image_id] = {}
            for ins in instances:
                bbox = ins['bbox']
                if str(bbox) not in id2pred[image_id]:
                    id2pred[image_id][str(bbox)] = []
                id2pred[image_id][str(bbox)].append(ins)

        def correct_fn(preds, image_id):
            ann_ids = coco_api.getAnnIds(imgIds=[image_id])
            annos = coco_api.loadAnns(ann_ids)
            cnt, tot = 0, len(annos)

            cls_anno = {anno['category_id'] - 1 : np.array(anno['bbox'], dtype=np.int64) for anno in annos}

            for pred in preds:
                id = pred['category_id']
                if cls_anno.get(id) is None:
                    continue
                if np.mean(np.array(pred['bbox'], dtype=np.int64) - cls_anno.get(id)) < 2:
                    cnt += 1
            return cnt, tot

        for id in id2pred.keys():
            for box in id2pred[id].keys():
                ind = np.argmax([id2pred[id][box][i]['score'] for i in range(len(id2pred[id][box]))])
                id2pred[id][box] = id2pred[id][box][ind]

        for id in id2pred.keys():
            preds = list(id2pred[id].values())
            cnt, tot = correct_fn(preds, id)
            corrects += cnt
            tot_box += tot
        
        return corrects / tot_box * 100 if tot_box > 0 else 0
    
    @classmethod
    def calc_confusion(cls, predictions, coco_api):
        cls_num = len(coco_api.loadCats(coco_api.getCatIds()))
        conf = ConfusionMatrix(cls_num, conf=0.04)

        def cxcywh_to_xyxy(x : np.ndarray):
            x_c, y_c, w, h = x[0], x[1], x[2], x[3]
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                 (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return b
        
        for pred in predictions:
            
            image_id, instances = pred['image_id'], pred['instances']
            
            ann_ids = coco_api.getAnnIds(imgIds=[image_id])
            annos = coco_api.loadAnns(ann_ids)

            detections = np.array([
                np.hstack([cxcywh_to_xyxy(ins['bbox']), ins['score'], ins['category_id']])
                for ins in instances
            ])

            labels = np.array([
                np.hstack([anno['category_id']-1, cxcywh_to_xyxy(anno['bbox'])])
                for anno in annos
            ])

            conf.process_batch(torch.tensor(detections), torch.tensor(labels))
            
            return conf