import os
import numpy as np
import torch
import json
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle


class VideoDataset(Dataset):
    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

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
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        # which part of data to load
        if self.mode == 'validate':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['validate'])
        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % (ix))))
        fc_feat = np.concatenate(fc_feat, axis=1)
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


class SummaryDataset(Dataset):
    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def __init__(self, args, mode, param, part="tvsum"):
        super(SummaryDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        info = json.load(open(args["info_json"]))
        vid = json.load(open(args["vid_json"]))
        final_captions = {}
        for i in vid['videos'].keys():
            final_captions[str(i)] = vid['videos'][i]['final_caption']
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.videos = vid['videos']
        self.splits = {'train': [], 'test': []}
        for i in vid['splits']['train']:
            if vid['videos'][i]['category'] == part:
                self.splits['train'].append(i)
        for i in vid['splits']['test']:
            if vid['videos'][i]['category'] == part:
                self.splits['test'].append(i)
        print('number of train videos: ', len(self.splits['train']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = args["feats_dir"]
        print('load feats from %s' % self.feats_dir)
        # load in the sequence data
        self.max_len = param["vid_len"]
        self.max_sen = param["max_sen"]
        self.max_user = 20
        self.max_frame = 20000
        self.max_cps = 300
        print('max sequence length in data is', self.max_len)
        self.captions = final_captions

    def __getitem__(self, ix):
        if self.mode == 'test':
            name = self.splits['test'][ix]
        else:
            name = self.splits['train'][ix]
        vid = pickle.load(open(self.videos[name]['path'], 'rb'))
        mask = np.zeros(self.max_sen)
        cap = self.captions[name]
        gts = np.zeros(self.max_sen)
        if len(cap) > self.max_sen:
            cap = cap[:self.max_sen]
            cap[-1] = '<eos>'
        for j, w in enumerate(cap):
            gts[j] = self.word_to_ix[w]
        label = gts
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1

        true_len = vid['features'].shape[0]
        true_seg = vid['change_points'].shape[0]
        true_user = vid['user_summary'].shape[0]
        true_frame = vid['user_summary'].shape[1]
        features = np.zeros((self.max_len, vid['features'].shape[1]))
        features[:vid['features'].shape[0], :] = vid['features']
        gt_score = np.zeros((self.max_len,))
        gt_score[:vid['gt_score'].shape[0]] = vid['gt_score']
        cps = np.zeros((self.max_cps, vid['change_points'].shape[1]))
        cps[:vid['change_points'].shape[0], :] = vid['change_points']
        fps = np.zeros((self.max_cps,))
        fps[:vid['fps'].shape[0]] = vid['fps']
        picks = np.zeros((self.max_len,))
        picks[:vid['picks'].shape[0]] = vid['picks']
        usr = np.zeros((self.max_user, self.max_frame))
        usr[:true_user, :true_frame] = vid['user_summary']

        # vid['features'] = pad_sequence(vid['features'], batch_first=True, padding_value=0.0)
        return_dict = {
            'features': torch.from_numpy(features).float(),
            'gt_score': torch.from_numpy(gt_score).float(),
            'labels': torch.from_numpy(label).long(),
            'masks': torch.from_numpy(mask).long(),
            'change_points': cps,
            'picks': picks,
            'fps': fps,
            'user_summary': usr,
            'true_len': true_len,
            'true_seg': true_seg,
            'true_user': true_user,
            'true_frame': true_frame
        }
        return return_dict

    def __len__(self):
        return len(self.splits[self.mode])


if __name__ == '__main__':
    param = {}
    import yaml
    with open('vsc_exp_0.yaml', 'r') as f:
        params = yaml.load(f)
    param["info_json"] = 'out_path/info.json'
    param["vid_json"] = 'summary_data/info.json'
    param["max_len"] = 15
    param["feats_dir"] = "summary_data"
    data = SummaryDataset(param, "train", params)
    print(data[16])