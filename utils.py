import json
import xml.etree.ElementTree as ET


def read_json(json_file):
    with open(json_file, 'r') as r:
        data = json.load(r)
    return data['images'], data['annotations'], data['categories']
    # return image_info, ann_info, category_info


def xywh2xyxy(bboxes):
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes


def parse_xml_to_dict(xml):
    """
    将 xml 文件解析为字典形式
    :param xml:  xml tree
    :return: dict
    """
    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        # 递归遍历标签信息
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def read_xml_to_dict(xml):
    """
    xml : xml file path
    xml_data: dict
    """
    tree = ET.parse(xml)
    root = tree.getroot()
    xml_data = parse_xml_to_dict(root)
    return xml_data
