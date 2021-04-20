import torch.nn as nn
import torch.nn.functional as F

class VisualSemanticEmbedding(nn.Module):
    def __init__(self,
                 input_fv_dim,
                 input_sen_dim,
                 v_hidden_size,  # list of int
                 t_hidden_size,  # list of int
                 use_kl_loss=False,
                 use_lstm_ve=False):
        super(VisualSemanticEmbedding, self).__init__()
        assert v_hidden_size[1] == t_hidden_size[1]
        self.LSTM_V_1 = nn.LSTM(input_fv_dim, v_hidden_size[0], num_layers=1, batch_first=True)
        self.LSTM_V_2 = nn.LSTM(v_hidden_size[0], v_hidden_size[1], num_layers=1, batch_first=True)
        self.LSTM_T_1 = nn.LSTM(input_sen_dim, v_hidden_size[0], num_layers=1, batch_first=True)
        self.LSTM_T_2 = nn.LSTM(v_hidden_size[0], v_hidden_size[1], num_layers=1, batch_first=True)
        self.kl = use_kl_loss

    def forward(self, video_vec, sentence_vec):
        out_v, _ = self.LSTM_V_1(video_vec)
        out_v, _ = self.LSTM_V_2(out_v)
        f_v_ = out_v[:, -1, :]
        out_t, _ = self.LSTM_T_1(sentence_vec)
        out_t, _ = self.LSTM_T_2(out_t)
        f_t_ = out_t[:, -1, :]
        if self.kl:
            p_t = F.softmax(f_t_, dim=-1)
            p_v = F.softmax(f_v_, dim=-1)
            return f_v_, f_t_, p_v, p_t
        return f_v_, f_t_