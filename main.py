from utils import read_json, xywh2xyxy, read_xml_to_dict
from visualize import imshow_det_bboxes, imshow_gt_bboxes, imshow_gt_det_bboxes
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os


def visualize_coco(json_file, image_dir, save_dir):
    """
    visualize the coco format annotations
    :param json_file: coco format annotations, json file
    :param image_dir: the directory to put images
    :param save_dir: the directory to save images
    :return: None
    """
    assert Path(json_file).exists(), 'result_file does not exist!'
    assert Path(image_dir).exists(), 'image_dir does not exist!'
    assert Path(save_dir).exists(), 'save_dir does not exist!'
    assert Path(result_file).suffix == '.json', \
        f' result_file should be json file, but {Path(result_file).suffix}.'
    # read the josn annotations
    images_info, anns_info, categories_info = read_json(result_file)
    # get the class_name
    class_names = [x['name'] for x in categories_info]

    for image_info in tqdm(images_info):
        image_name = image_info['file_name']

        gt_bboxes, gt_labels = [], []
        for ann_info in anns_info:
            if ann_info['image_id'] == image_info['id']:
                gt_bboxes.append(ann_info['bbox'])
                gt_labels.append(ann_info['category_id'])

        # convert the xywh2xyxy
        gt_bboxes = xywh2xyxy(np.array(gt_bboxes))
        gt_labels = np.array(gt_labels) - 1 # start from 0

        # visualize
        image_dir = Path(image_dir)
        save_dir = Path(save_dir)
        _ = imshow_gt_bboxes(
            img=str(image_dir.joinpath(image_name)),
            annotation={'gt_bboxes': gt_bboxes, 'gt_labels': gt_labels},
            class_names=class_names,
            show=True,
            out_file=str(save_dir.joinpath(image_name))
        )

    return None


def visualize_xml(xml_file, image_dir, save_dir, class_dict):
    """
    visualize the xml file
    :param xml_file: xml file or the directory saving xml file
    :param image_dir: the directory to put the images
    :param save_dir: the directory to save images
    :param class_dict: {'name': id} to obtain labels
    :return: None
    """
    assert Path(xml_file).exists(), 'result_file does not exist!'
    assert Path(image_dir).exists(), 'image_dir does not exist!'
    assert Path(save_dir).exists(), 'save_dir does not exist!'
    if Path(xml_file).is_file():
        xml_file = Path(xml_file)
        xml_dir = xml_file.parents
        xml_file_list = [xml_file.stem + xml_file.suffix]
    else:
        xml_dir = Path(xml_file)
        xml_file_list = os.listdir(xml_file)

    image_dir = Path(image_dir)
    save_dir = Path(save_dir)
    class_names = [x for x, _ in class_dict.items()]

    for xml_file in tqdm(xml_file_list):
        # parse xml file to dict
        xml_dict = read_xml_to_dict(str(xml_dir.joinpath(xml_file)))
        gt_bboxes = []
        gt_labels = []
        image_name = xml_dict['annotation']['filename']

        anns = xml_dict['annotation'].get('object', None)
        if anns is None:
            print(f'image {image_name} is empty!')
            continue
        for ann in anns:
            bbox = [ann['bndbox']['xmin'], ann['bndbox']['ymin'], ann['bndbox']['xmax'], ann['bndbox']['ymax']]
            label = class_dict[ann['name']]
            gt_bboxes.append(bbox)
            gt_labels.append(label)
        gt_bboxes = np.array(gt_bboxes)
        gt_labels = np.array(gt_labels)
        if np.min(list(class_dict.values())) != 0:
            gt_labels = gt_labels - np.min(list(class_dict.values()))

        _ = imshow_gt_bboxes(
            img=str(image_dir.joinpath(image_name)),
            annotation={'gt_bboxes': gt_bboxes, 'gt_labels': gt_labels},
            class_names=class_names,
            show=True,
            out_file=str(save_dir.joinpath(image_name))
        )
        break
    return None


def main(name, file, image_dir, save_dir, class_dict=None):
    if name == 'COCO':
        visualize_coco(json_file=file, image_dir=image_dir, save_dir=save_dir)
    elif name == 'xml':
        print(class_dict)
        assert class_dict is not None, 'class_dict should be not None in xml format'
        visualize_xml(xml_file=file, image_dir=image_dir, save_dir=save_dir, class_dict=class_dict)


if __name__ == '__main__':
    # result_file = 'annotations/val1.json'
    # image_dir = 'image'
    # save_dir = 'visualize_data'
    # name = 'COCO'
    # main(name=name, file=result_file, image_dir=image_dir, save_dir=save_dir)

    result_file = 'bbox'
    image_dir = 'image'
    save_dir = 'visualize_data'
    name = 'xml'
    label_ids = { "scratch": 1, "bubble": 2, "pinhole": 3, "tin_ash": 4}
    main(name=name, file=result_file, image_dir=image_dir, save_dir=save_dir, class_dict=label_ids)
