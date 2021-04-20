import numpy as np
import torch
import torch.nn as nn
from models.Attention import TemporalAttentionModule
from models.SentenceEncoder import CustomSentenceEncoder
from models.VisualEmbedding import VisualSemanticEmbedding
from models.CaptionGen import CaptionGenerationModule


class VSCModel(nn.Module):
    def __init__(self,
                 device,
                 model_type,
                 vocab_size,
                 word_dim,
                 max_vid,
                 vid_dim,
                 max_sen,
                 params,
                 ):
        super(VSCModel, self).__init__()

        if model_type not in ['bert', 'glove', 'custom']:
            raise NameError('Model type should be one of ' + str(['bert', 'glove', 'custom']))
        else:
            self.model_type = model_type

        if model_type == 'custom':
            emb = torch.nn.Embedding(vocab_size, word_dim).to(device)
            self.SentenceEncoder = CustomSentenceEncoder(device, emb, word_dim,
                                                         params['se']['num_layer'],
                                                         params['se']['hidden']).to(device)

            self.Att = TemporalAttentionModule(vid_dim, max_vid,
                                               vid_dim - params['ve']['v_hidden'][-1]).to(device)

            self.VisualEmb = VisualSemanticEmbedding(vid_dim,
                                                     params['se']['hidden'],
                                                     params['ve']['v_hidden'],
                                                     params['ve']['t_hidden'],
                                                     params['ve']['kl_loss']).to(device)

            self.CapGen = CaptionGenerationModule(vocab_size=vocab_size,
                                                  max_len=max_sen,
                                                  dim_word=word_dim,
                                                  dim_vid=vid_dim,
                                                  dim_hidden=params['cap']['hidden'],
                                                  embedding=emb,
                                                  n_layers=params['cap']['num_layer'],
                                                  rnn_cell=params['cap']['rnn_type'],
                                                  rnn_dropout_p=params['cap']['rnn_dropout'],
                                                  device=device)

    def forward(self, inp_vid, inp_sen, mode='train'):
        if not isinstance(inp_vid, torch.Tensor):
            inp_vid = torch.tensor(inp_vid)
        if not isinstance(inp_sen, torch.Tensor):
            inp_sen = torch.tensor(inp_sen)
        return_dict = {}
        f_t = self.SentenceEncoder(inp_sen)
        f_v, att, f_s = self.Att(inp_vid)
        return_dict['att'] = att
        if self.VisualEmb.kl:
            f_v_, f_t_, p_v, p_t = self.VisualEmb(f_v, f_t)
            return_dict['f_v_'], return_dict['f_t_'], return_dict['p_v'], return_dict['p_t'] = f_v_, f_t_, p_v, p_t
        else:
            f_v_, f_t_ = self.VisualEmb(f_v, f_t)
            return_dict['f_v_'], return_dict['f_t_'] = f_v_, f_t_
        f_vs = torch.cat((f_s, f_v_), -1)
        seq_prob, seq_pred = self.CapGen(f_vs.view(-1, 1, f_vs.shape[-1]), inp_sen, mode=mode)
        return_dict['out_cap'] = seq_pred
        return_dict['prob'] = seq_prob
        return return_dict

