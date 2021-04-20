import os
import pickle
import pandas as pd
import h5py
import numpy as np
import json
import argparse
import spacy


def reformat_text_dict(data):
    dic = {}
    for i in range(len(data['video name'].keys())):
        dic[data['video name'][i]] = data['caption'][i]
    return dic


def get_tvsum_data(path):
    dict = {}
    f = h5py.File(path, 'r')
    for name in f.keys():
        vname = str(np.array(f.get(name + '/video_name')))
        vname = vname[1:-1]
        dict.update({vname: get_dict(name, f)})
    f.close()
    return dict


def get_dict(name, f):
    feature = np.array(f.get(name + '/features'))
    picks = np.array(f.get(name + '/picks'))
    change_points = np.array(f.get(name + '/change_points'))
    gt_score = np.array(f.get(name + '/gtscore'))
    fps = np.array(f.get(name + '/n_frame_per_seg'))
    user_summary = np.array(f.get(name + '/user_summary'))
    dict = {'features': feature, 'user_summary': user_summary,
            'picks': picks, 'change_points': change_points,
            'gt_score': gt_score, 'fps': fps}
    return dict


def process_h5_data(path):
    data = h5py.File(path, 'r')
    data_dict = {}
    for vid_index in data.keys():
        name = data[vid_index]['video_name'][()].decode('utf-8')
        change_points = data[vid_index]['change_points'][()]
        features = data[vid_index]['features'][()]
        gtscore = data[vid_index]['gtsummary'][()]  # equal to .value
        picks = data[vid_index]['picks'][()]
        user_summary = data[vid_index]['user_summary'][()]
        fps = data[vid_index]['n_frame_per_seg'][()]
        data_dict[name] = {}
        data_dict[name]['change_points'] = change_points
        data_dict[name]['features'] = features
        data_dict[name]['gt_score'] = gtscore
        data_dict[name]['picks'] = picks
        data_dict[name]['fps'] = fps
        data_dict[name]['user_summary'] = user_summary
    return data_dict


def load(root_dir):
    text_path = os.path.join(root_dir, 'text')
    vid_path = os.path.join(root_dir, 'video')
    summe_vid_path = os.path.join(vid_path, 'eccv16_dataset_summe_google_pool5.h5')
    summe_text_path = os.path.join(text_path, 'summe_caption.csv')
    tvsum_vid_path = os.path.join(vid_path, 'eccv16_dataset_tvsum_google_pool5.h5')
    tvsum_text_path = os.path.join(text_path, 'tvsum_caption.csv')
    summe_text = reformat_text_dict(pd.read_csv(summe_text_path).to_dict())
    tvsum_text = reformat_text_dict(pd.read_csv(tvsum_text_path).to_dict())
    summe_vid = process_h5_data(summe_vid_path)
    tvsum_vid = get_tvsum_data(tvsum_vid_path)
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # print(sys.maxsize)
    # print(tvsum_vid['AwmHb44_ouw']['user_summary'].mean(axis=0))
    # exit()
    # return {**summe_vid, **tvsum_vid}, {**summe_text, **tvsum_text}
    return summe_vid, summe_text, tvsum_vid, tvsum_text


def write_format_data(data_path, out_path):
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass
    new_data = {}
    spc = spacy.load('en')
    summe_vid, summe_text, tvsum_vid, tvsum_text = load(data_path)
    idx = 0
    split = {'train': [], 'test': []}
    for i in summe_text.keys():
        new_data[i] = dict()
        new_data[i]['path'] = out_path + '/' + str(i) + '.pkl'
        new_data[i]['category'] = 'summe'
        new_data[i]['caption'] = summe_text[i]
        new_data[i]['final_caption'] = ["<sos>"] + [tok.text.lower() for tok in spc.tokenizer(summe_text[i])] + [
            "<eos>"]
        if idx < 20:
            split['train'].append(str(i))
        else:
            split['test'].append(str(i))
        idx += 1
    idx = 0
    for i in tvsum_text.keys():
        new_data[i] = dict()
        new_data[i]['path'] = out_path + '/' + str(i) + '.pkl'
        new_data[i]['category'] = 'tvsum'
        new_data[i]['caption'] = tvsum_text[i]
        new_data[i]['final_caption'] = ["<sos>"] + [tok.text.lower() for tok in spc.tokenizer(tvsum_text[i])] + [
            "<eos>"]
        if idx < 40:
            split['train'].append(str(i))
        else:
            split['test'].append(str(i))
        idx += 1
    new_data = {
        'videos': new_data,
        'splits': split
    }
    for i in new_data.keys():
        with open(os.path.join(out_path, str(i) + '.pkl'), 'wb') as f:
            if i in summe_vid.keys():
                pickle.dump(summe_vid[i], f)
            elif i in tvsum_vid.keys():
                pickle.dump(tvsum_vid[i], f)
    with open(os.path.join(out_path, 'info.json'), 'w') as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--data_path', default='G:/dataset',
                        help='path of tvsum and summe dataset.')
    parser.add_argument('--out_path', default='summary_data',
                        help='path of output path.')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    write_format_data(params['data_path'], params['out_path'])
