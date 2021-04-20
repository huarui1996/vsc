import os
import h5py
import pandas as pd
import numpy as np
import torch
import json
import random
from torch.utils.data import Dataset


def slice_dict(dic):
    len_ = len(dic)
    l1 = int(len_*0.8)
    dict1 = {}
    dict2 = {}
    for name in dic:
        if l1 != 0:
            dict1.update({name: dic[name]})
            l1 -= 1
        else:
            dict2.update({name: dic[name]})
    return dict1, dict2


def reformat_text_dict(data):
    dic = {}
    for i in range(len(data['video name'].keys())):
        dic[data['video name'][i]] = data['caption'][i]
    return dic


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


def get_tvsum_data(path):
    dict = {}
    f = h5py.File(path, 'r')
    for name in f.keys():
        vname = str(np.array(f.get(name + '/video_name')))
        vname = vname[1:-1]
        dict.update({vname: get_dict(name, f)})
    f.close()
    return dict


def convert_to_fine_grained(video_dict, text_dict, max_vid_len) -> [np.ndarray, list]:
    names = text_dict.keys()
    vid, txt, g, vid_idx, fps = [], [], [], [], []
    cps = []
    pick_lis = []
    idx = 0
    for name in names:
        v = video_dict[name]['features']
        cp = video_dict[name]['change_points']
        gt = video_dict[name]['gt_score']
        fps_ = video_dict[name]['fps']
        if gt.shape == ():
            print(name)
        cp_idx = 0
        picks = video_dict[name]['picks']
        cps.append(cp)
        pick_lis.append(picks)
        fps.append(fps_)
        seg, seg_gt = [], []
        for i in range(v.shape[0]):
            if picks[i] <= cp[cp_idx][1]:
                seg.append(v[i])
                seg_gt.append(gt[i])
            else:
                cp_idx += 1
                if seg:
                    vid.append(seg)
                    txt.append(text_dict[name])
                    g.append(seg_gt)
                    vid_idx.append(idx)
                seg, seg_gt = [], []
        idx += 1
    vid = pad_to_max(vid, max_vid_len)
    g = pad_to_max(g, max_vid_len)
    return np.array(vid), txt, g, vid_idx, cps, pick_lis, fps


def convert_to_coarse_grained(video_dict, text_dict, max_vid_len):
    names = text_dict.keys()
    idx = 0
    vid_idx, vid, txt, g, cps, picks, fps, user_summary = [], [], [], [], [], [], [], []
    for name in names:
        vid.append(video_dict[name]['features'])
        txt.append(text_dict[name])
        g.append(video_dict[name]['gt_score'])
        cps.append(video_dict[name]['change_points'])
        picks.append(video_dict[name]['picks'])
        fps.append(video_dict[name]['fps'])
        user_summary.append(video_dict[name]['user_summary'])
        vid_idx.append(idx)
        idx += 1
    vid = pad_to_max(vid, max_vid_len)
    g = pad_to_max(g, max_vid_len)
    return np.array(vid), txt, g, vid_idx, cps, picks, fps, user_summary


def pad_to_max(vid, max_len=-1):
    data = []
    if max_len == -1:
        max_len = 0
        for v in vid:
            if len(v) > max_len:
                max_len = len(v)
        print("Max Len of Seg/Vid (fine-grained/coarse-grained) %d" % max_len)
    else:
        print("Set max length to %d" % max_len)
    for v in vid:
        if type(v) == list:
            while len(v) < max_len:
                v.append(np.zeros(v[0].shape))
            data.append(v)
        elif isinstance(v, np.ndarray):
            try:
                data.append(np.pad(v, ((0, max_len - v.shape[0]), (0, 0)), mode='constant'))
            except ValueError:
                data.append(np.pad(v, (0, max_len - v.shape[0]), mode='constant'))
        else:
            raise TypeError("Need to be list or ndarray")
    return data


def load(root_dir, split):
    text_path = os.path.join(root_dir, split, 'text')
    vid_path = os.path.join(root_dir, split, 'video')
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


def load_dataset(root_dir, batch_size=16, max_vid_len=1300):
    summe_vid, summe_text, tvsum_vid, tvsum_text = load(root_dir, 'train')
    summe_train = list(summe_vid.keys())[:int(0.8 * len(summe_vid))]
    summe_test = list(summe_vid.keys())[int(0.8 * len(summe_vid)):]
    tvsum_train = list(tvsum_vid.keys())[:int(0.8 * len(tvsum_vid))]
    tvsum_test = list(tvsum_vid.keys())[int(0.8 * len(tvsum_vid)):]
    train_vid = {**{i: summe_vid[i] for i in summe_train}, **{i: tvsum_vid[i] for i in tvsum_train}}
    test_vid = {**{i: summe_vid[i] for i in summe_test}, **{i: tvsum_vid[i] for i in tvsum_test}}
    train_text = {**{i: summe_text[i] for i in summe_train}, **{i: tvsum_text[i] for i in tvsum_train}}
    test_text = {**{i: summe_text[i] for i in summe_test}, **{i: tvsum_text[i] for i in tvsum_test}}
    vid, txt, gt_score, vid_idx, cps, picks, fps, usr = convert_to_coarse_grained(train_vid, train_text, max_vid_len)
    train = VSCDataset(batch_size, video_features=vid, vid_idx=vid_idx,
                       gt_score=gt_score, text=txt, cps=cps, picks=picks, fps=fps, usr=usr)
    vid, txt, gt_score, vid_idx, cps, picks, fps, usr = convert_to_coarse_grained(test_vid, test_text, max_vid_len)
    test = VSCDataset(1, video_features=vid, vid_idx=vid_idx,
                      gt_score=gt_score, text=txt, cps=cps, picks=picks, fps=fps, usr=usr)
    return train, test


class VSCDataset(Dataset):
    def __init__(self,
                 batch_size,
                 split: str = 'train',
                 max_vid_length: int = -1,
                 fine_grained: bool = False,
                 root_dir: str = None,
                 video_features: np.ndarray = None,
                 gt_score: np.ndarray = None,
                 vid_idx: np.ndarray = None,
                 text=None,
                 cps=None,
                 picks=None,
                 fps=None,
                 usr=None):  # list of str
        """
        root_dir is root dir of dataset, leave it None if video and text were read.
        Structure as follow:
            root/
                <split_name>/ [train|test|dev]
                    video/
                        eccv16_dataset_summe_google_pool5.h5
                        eccv16_dataset_tvsum_google_pool5.h5
                    text/
                        summe_caption.csv
                        tvsum_caption.csv
        """
        self.batch_size = batch_size
        self.fine_grained = fine_grained
        if root_dir is not None:
            video_dict, text_dict = load(root_dir, split)
            if fine_grained:
                vid, txt, gt_score, vid_idx, cps, picks, fps, usr = convert_to_fine_grained(video_dict, text_dict, max_vid_length)
            else:
                vid, txt, gt_score, vid_idx, cps, picks, fps, usr = convert_to_coarse_grained(video_dict, text_dict, max_vid_length)
            video_features = vid
            text = txt
        self.VideoFeatures = video_features
        self.Text = text
        if gt_score:
            self.Gt_score = np.array(gt_score)
        if vid_idx:
            self.vid_idx = np.array(vid_idx)
        if cps:
            self.cps = np.array(cps)
        if picks:
            self.picks = np.array(picks)
        if fps:
            self.fps = np.array(fps)
        if usr:
            self.usr = np.array(usr)
            #lis = []
            #for i in self.usr:
            #    lis.append(i.mean(0))
            #self.usr = np.array(lis)
        if True:
            self.VideoFeatures = np.pad(self.VideoFeatures, ((0, batch_size - self.VideoFeatures.shape[0] % batch_size),
                                                             (0, 0), (0, 0)),
                                        mode='constant')
            if gt_score:
                self.Gt_score = np.pad(self.Gt_score, ((0, batch_size - self.Gt_score.shape[0] % batch_size), (0, 0)),
                                       mode='constant')
            if vid_idx:
                self.vid_idx = np.pad(self.vid_idx, (0, batch_size - self.vid_idx.shape[0] % batch_size),
                                      mode='constant')

        for i in range(batch_size - len(self.Text) % batch_size):
            self.Text.append("")
        self.batch_num = len(self.Text)
        self.max_len = 0
        for i in self.Text:
            if len(i) > self.max_len:
                self.max_len = len(i)
        if batch_size != 1:
            self.batch_num = self.VideoFeatures.shape[0] // batch_size
            self.VideoFeatures = self.VideoFeatures.reshape([self.batch_num, batch_size,
                                                             self.VideoFeatures.shape[1], self.VideoFeatures.shape[2]])
            self.Text = np.array(self.Text).reshape(self.batch_num, batch_size)
            if gt_score:
                self.Gt_score = self.Gt_score.reshape([self.batch_num, batch_size, self.Gt_score.shape[1]])
            if vid_idx:
                self.vid_idx = self.vid_idx.reshape([self.batch_num, batch_size])
        else:
            self.VideoFeatures = self.VideoFeatures.reshape([-1, 1,
                                                             self.VideoFeatures.shape[1], self.VideoFeatures.shape[2]])
            self.Text = np.array(self.Text).reshape([-1, 1])
            if gt_score:
                self.Gt_score = self.Gt_score.reshape([-1, 1, self.Gt_score.shape[1]])
            if vid_idx:
                self.vid_idx = self.vid_idx.reshape([-1, 1])
        self.is_training = True

    def __len__(self):
        if self.batch_size != 1:
            return self.batch_num
        else:
            return self.batch_num - 1

    @property
    def max_length(self):
        return self.max_len

    def resize(self, batch_size):
        if batch_size != 1:
            self.batch_num = self.VideoFeatures.shape[0] // batch_size
            self.VideoFeatures = self.VideoFeatures.reshape([self.batch_num, batch_size,
                                                             self.VideoFeatures.shape[1], self.VideoFeatures.shape[2]])
            self.Text = np.array(self.Text).reshape(self.batch_num, batch_size)
            self.Gt_score = self.Gt_score.reshape([self.batch_num, batch_size, self.Gt_score.shape[1]])
            self.vid_idx = self.vid_idx.reshape([self.batch_num, batch_size])
        else:
            self.VideoFeatures = self.VideoFeatures.reshape([-1, 1,
                                                             self.VideoFeatures.shape[1], self.VideoFeatures.shape[2]])
            self.Text = np.array(self.Text).reshape([-1, 1])
            self.Gt_score = self.Gt_score.reshape([-1, 1, self.Gt_score.shape[1]])
            self.vid_idx = self.vid_idx.reshape([-1, 1])

    def __getitem__(self, idx):
        return_dict = {}
        if idx == self.batch_num - 1 and self.batch_size != 1:
            return_dict['data'] = self.VideoFeatures[idx][:len(self.Text) % self.batch_size]
            return_dict['text'] = self.Text[idx][:len(self.Text) % self.batch_size]
            return_dict['gt_score'] = self.Gt_score[idx][:len(self.Text) % self.batch_size]
            return_dict['vid_idx'] = self.vid_idx[idx][:len(self.Text) % self.batch_size]
        else:
            return_dict['data'] = self.VideoFeatures[idx]
            return_dict['text'] = self.Text[idx]
            if hasattr(self, 'Gt_score'):
                return_dict['gt_score'] = self.Gt_score[idx]
            if hasattr(self, 'vid_idx'):
                return_dict['vid_idx'] = self.vid_idx[idx]
            if hasattr(self, 'usr') and self.batch_size == 1:
                return_dict['usr'] = self.usr[idx]
        return return_dict


def get_pretrain_data(path):
    data = h5py.File(os.path.join(path, 'MSVD_GoogleNet.hdf5'))
    text_dict = {}
    with open(os.path.join(path, 'text'), 'r') as f:
        text = f.readlines()
    for line in text:
        name, cap = line.strip('\n').strip('.').split('\t')
        if name not in text_dict.keys():
            text_dict[name] = cap
    files = list(data.keys())
    vid = []
    text = []
    for name in files:
        vid.append(data[name][()])
        text.append(text_dict[name])
    vid = pad_to_max(vid)
    return np.array(vid), text


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['validate']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = opt["feats_dir"]
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])
        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % (ix))))
        fc_feat = np.concatenate(fc_feat, axis=1)
        if self.with_c3d == 1:
            c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy'%(ix)))
            c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
            fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        captions = self.captions['video%i'%(ix)]['final_captions']
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]

        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1

        data = {'fc_feats': torch.from_numpy(fc_feat).float(), 'labels': torch.from_numpy(label).long(),
                'masks': torch.from_numpy(mask).float(), 'gts': torch.from_numpy(gts).long(),
                'video_ids': 'video%i' % ix}
        return data

    def __len__(self):
        return len(self.splits[self.mode])


if __name__ == '__main__':
    vid, text = get_pretrain_data('pretrain')
    print(vid.shape)
    print(vid.nbytes)

    data = VSCDataset(batch_size=4, video_features=vid[:300], text=text[:300])
    print(data[0]['data'].shape)