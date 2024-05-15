from detectron2.data.datasets.register_coco import register_coco_instances
import os


categories = [
  {'id': 1, 'name': 'C1'},
 {'id': 2, 'name': 'C2'},
 {'id': 3, 'name': 'C3'},
 {'id': 4, 'name': 'C4'},
 {'id': 5, 'name': 'C5'},
 {'id': 6, 'name': 'C6'},
 {'id': 7, 'name': 'C7'},
 {'id': 8, 'name': 'T1'},
 {'id': 9, 'name': 'T2'},
 {'id': 10, 'name': 'T3'},
 {'id': 11, 'name': 'T4'},
 {'id': 12, 'name': 'T5'},
 {'id': 13, 'name': 'T6'},
 {'id': 14, 'name': 'T7'},
 {'id': 15, 'name': 'T8'},
 {'id': 16, 'name': 'T9'},
 {'id': 17, 'name': 'T10'},
 {'id': 18, 'name': 'T11'},
 {'id': 19, 'name': 'T12'},
 {'id': 20, 'name': 'L1'},
 {'id': 21, 'name': 'L2'},
 {'id': 22, 'name': 'L3'},
 {'id': 23, 'name': 'L4'},
 {'id': 24, 'name': 'L5'},
 {'id': 25, 'name': 'L6'},
]

# categories = [
#     {'id': 0, 'name': 'Vertebrae_S'},
#     {'id': 1, 'name': 'Vertebrae_L5'},
#     {'id': 2, 'name': 'Vertebrae_L4'},
#     {'id': 3, 'name': 'Vertebrae_L3'},
#     {'id': 4, 'name': 'Vertebrae_L2'},
#     {'id': 5, 'name': 'Vertebrae_L1'},
#     {'id': 6, 'name': 'Vertebrae_T12'},
#     {'id': 7, 'name': 'Vertebrae_T11'},
#     {'id': 8, 'name': 'Vertebrae_T10'},
#     {'id': 9, 'name': 'Vertebrae_T9'},
#     {'id': 10, 'name': 'Vertebrae_T8'},
#     {'id': 11, 'name': 'Vertebrae_T7'},
#     {'id': 12, 'name': 'Vertebrae_T6'},
#     {'id': 13, 'name': 'Vertebrae_T5'},
#     {'id': 14, 'name': 'Vertebrae_T4'},
#     {'id': 15, 'name': 'Vertebrae_T3'},
#     {'id': 16, 'name': 'Vertebrae_T2'},
#     {'id': 17, 'name': 'Vertebrae_T1'},
#     {'id': 18, 'name': 'Vertebrae_C7'},
#     {'id': 19, 'name': 'Vertebrae_C6'},
#     {'id': 20, 'name': 'Vertebrae_C5'},
#     {'id': 21, 'name': 'Vertebrae_C4'},
#     {'id': 22, 'name': 'Vertebrae_C3'},
#     {'id': 23, 'name': 'Vertebrae_C2'},
#     {'id': 24, 'name': 'Vertebrae_C1'},
# ]



def _get_builtin_metadata(categories):
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]

    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


_PREDEFINED_SPLITS_VertDet = {
    "vertdet_train": ("vert/split1/train_images", "vert/annotations/dataset_2_train_new_prompt_4_split1.json"),
    "vertdet_val": ("vert/split1/test_images", "vert/annotations/dataset_2_test_new_prompt_4_split1.json"),
    "vertdet_train2": ("vert/split2/train_images", "vert/annotations/dataset_2_train_new_prompt_4_split2.json"),
    "vertdet_val2": ("vert/split2/test_images", "vert/annotations/dataset_2_test_new_prompt_4_split2.json"),
    "vertdet_train3": ("vert/split3/train_images", "vert/annotations/dataset_2_train_new_prompt_4_split3.json"),
    "vertdet_val3": ("vert/split3/test_images", "vert/annotations/dataset_2_test_new_prompt_4_split3.json"),
    "vertdet_train4": ("vert/split4/train_images", "vert/annotations/dataset_2_train_new_prompt_4_split4.json"),
    "vertdet_val4": ("vert/split4/test_images", "vert/annotations/dataset_2_test_new_prompt_4_split4.json"),
    "vertdet_train5": ("vert/split5/train_images", "vert/annotations/dataset_2_train_new_prompt_4_split5.json"),
    "vertdet_val5": ("vert/split5/test_images", "vert/annotations/dataset_2_test_new_prompt_4_split5.json"),
    "verse19_train":("verse2019/train/images", "verse2019/verse19_train.json"),
    "verse19_val":("verse2019/val/images", "verse2019/verse19_val.json"),
    "verse19_test":("verse2019/test/images", "verse2019/verse19_test.json"),
    "verse20_train":("verse2020/train/images", "verse2020/verse20_train.json"),
    "verse20_val":("verse2020/val/images", "verse2020/verse20_val.json"),
    "verse20_test":("verse2020/test/images", "verse2020/verse20_test.json")
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_VertDet.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )