import os
import json
import random
import base64
from lxml import etree
import re
from PIL import Image
import io

from utils import save_data


def process_data(root_dir, output_dir, mode, part):
    DATASET_NAME = f"mind2web_agenttuning_{part}"
    save_dir = os.path.join(output_dir, f"processed/mind2web_agenttuning_llama/{mode}")
    os.makedirs(save_dir, exist_ok=True)
    raw_root = os.path.join(root_dir, "raw_dump")
    data_root = os.path.join(root_dir, "data")
    all_data = []
    drop_img, drop_15, drop1, drop_empty, drop_filt_pos, drop_filt_neg, item_num = 0, 0, 0, 0, 0, 0, 0
    # ratio = []
    # bad_height = []
    for fn in os.listdir(os.path.join(data_root, mode)):
        if not fn.endswith(f"_{part}.json"):
            continue
        # if not fn.endswith(f".json"):
        #     continue
        with open(os.path.join(data_root, mode, fn), "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            annotation_id = item['annotation_id']
            with open(os.path.join(raw_root, f"task/{annotation_id}/processed/screenshot.json"), "r") as f:
                screenshots = json.load(f)
            question = item['confirmed_task']
            action_reprs = []
            for action, repr, screen in zip(item['actions'], item['action_reprs'], screenshots):
                operation = action['operation']
                action_uid = action['action_uid']
                pos_can = [x for x in action['pos_candidates'] if json.loads(x["attributes"])['bounding_box_rect'] != '-1,-1,-1,-1']
                neg_can = [x for x in action['neg_candidates'] if json.loads(x["attributes"])['bounding_box_rect'] != '-1,-1,-1,-1']
                if not pos_can or not neg_can:
                    drop1 += 1
                    continue
                try:
                    jpg = base64.b64decode(screen["before"]["screenshot"])
                    img = Image.open(io.BytesIO(jpg)).convert('RGB')
                except KeyboardInterrupt:
                    exit()
                except Exception as e:
                    print("wrong image")
                    drop_img += 1
                    continue
                # aspect_ratio = 2560 / 1440

                # # 获取原始图片的宽度和高度
                # original_width, original_height = img.size

                # new_height = int(original_width / aspect_ratio)

                width, height = img.size
                max_down = height * 1.5
                if len(pos_can) == 0:
                    drop_empty += 1
                    continue
                good_pos = []
                for pos in pos_can:
                    label = json.loads(pos["attributes"])
                    bounding = label['bounding_box_rect'].split(',')
                    # ratio.append((float(bounding[1])+float(bounding[3])) / height)
                    # bad_height.append((float(bounding[1])+float(bounding[3])) / new_height)
                    if float(bounding[1])+float(bounding[3]) < max_down:
                        good_pos.append(pos)
                if not good_pos:
                    drop_15 += 1
                    continue
                
                # raw_action = screen['action']
                from dom_utils import get_tree_repr

                cleaned_tree = etree.fromstring(action['cleaned_html'])

                better_pos = []
                for pos in good_pos:
                    candidate_id = pos['backend_node_id']
                    res = cleaned_tree.xpath(f'//*[@backend_node_id="{candidate_id}"]')
                    if len(res) == 0:
                        continue
                    # pos['cleaned_html'] = etree.tostring(res[0]).decode('utf-8')
                    # pos['html'] = get_tree_repr(res[0])
                    better_pos.append(pos)
                better_neg = []
                for neg in neg_can:
                    candidate_id = pos['backend_node_id']
                    res = cleaned_tree.xpath(f'//*[@backend_node_id="{candidate_id}"]')
                    if len(res) == 0:
                        continue
                    # neg['html'] = get_tree_repr(res[0])
                    better_neg.append(neg)
                if not better_pos:
                    drop_filt_pos += 1
                    continue
                if not better_neg:
                    drop_filt_neg += 1
                    continue

                # dom_tree = etree.fromstring(action['raw_html'])
                
                # id2box = {}
                # select_elements = dom_tree.xpath('//select')
                # # 遍历所有找到的select元素并将它们转换为字符串
                # for select in select_elements:
                #     if select.attrib['bounding_box_rect'] != '-1,-1,-1,-1':
                #         id2box[select.attrib['backend_node_id']] = select.attrib['bounding_box_rect']
                
                # select_elements = cleaned_tree.xpath('//select')
                # selects = []
                # bad_select = []
                # # 遍历所有找到的select元素并将它们转换为字符串
                # for select in select_elements:
                #     if select.attrib['backend_node_id'] not in id2box:
                #         bad_select.append(select.attrib['backend_node_id'])
                #         continue
                #     select.attrib['bounding_box_rect'] = id2box[select.attrib['backend_node_id']]
                #     for elem in select.iterdescendants():
                #         del elem.attrib['backend_node_id']
                #     # etree.tostring方法将元素转换为字节串
                #     select_string = etree.tostring(select)
                #     # 将字节串解码为utf-8格式的字符串
                #     select_string_decoded = select_string.decode('utf-8')
                #     selects.append(re.sub(r">\s+<", "><", select_string_decoded).strip())
                
                # good_pos = [x for x in good_pos if x['backend_node_id'] not in bad_select]
                # neg_can = [x for x in neg_can if x['backend_node_id'] not in bad_select]
                # if not good_pos:
                #     drop_filt_pos += 1
                #     continue
                # if not neg_can:
                #     drop_filt_neg += 1
                #     continue

                c_data = {
                    "datatype": "html_agent_qa_text",
                    "question_id": annotation_id + "_" + action_uid,
                    "metadata": {
                        "cleaned_html": action['cleaned_html'],
                        "question": question,
                        "operation": operation,
                        "pos_candidates": better_pos,
                        "neg_candidates": better_neg,
                        # "raw_action": raw_action,
                        "repr": repr,
                        "pre_actions": action_reprs.copy(),
                        # "cleaned_html": action['cleaned_html']
                        # "selects": selects
                    }
                }
                action_reprs.append(repr)
                all_data.append({"image_bytes": jpg, "json": [c_data]})
                item_num += 1
    # import numpy as np

    # # 使用numpy计算直方图的数值
    # hist, bins = np.histogram(ratio, bins=30, range=(0, 1))

    # # 打印直方图
    # print("ratio Value\tHist")
    # for i in range(len(hist)):
    #     # 打印直方图的每个桶，使用ASCII字符
    #     print(f"{bins[i]:.1f}-{bins[i+1]:.1f}\t{hist[i]}")

    # # 使用numpy计算直方图的数值
    # data_min = min(bad_height)
    # data_max = max(bad_height)
    # hist, bins = np.histogram(bad_height, bins=30, range=(data_min, data_max))

    # # 打印直方图
    # print("height Value\tHist")
    # for i in range(len(hist)):
    #     # 打印直方图的每个桶，使用ASCII字符
    #     print(f"{bins[i]:.1f}-{bins[i+1]:.1f}\t{hist[i]}")
    random.shuffle(all_data)
    image_num = save_data(all_data, save_dir, DATASET_NAME, mode)
    print(f"Save: {image_num} images, {item_num} samples. Drop: {drop_img} wrong img, {drop_15} >1.5, {drop1} drop-1, {drop_empty} empty pos, {drop_filt_pos} filt pos empty, {drop_filt_neg} filt neg empty.")


if __name__ == '__main__':
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    root_dir = "/share/home/lqs/benchmark/Mind2Web"
    output_dir = "/share/home/lqs/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    for mode in ['train', 'test_domain', 'test_task', 'test_website']:
        print(f"process {mode}.")
        process_data(root_dir, output_dir, mode, rank)