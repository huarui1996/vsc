import torch
import torch.nn as nn


class TemporalAttentionModule(nn.Module):
    def __init__(self, video_embedding_size, max_vid_len, fvs_dim):
        super(TemporalAttentionModule, self).__init__()
        self.FC_ATT = nn.Linear(in_features=video_embedding_size, out_features=1)
        self.FC_FVS = nn.Linear(max_vid_len, fvs_dim)
        self.AvePool = nn.AvgPool2d(kernel_size=video_embedding_size)

    def forward(self, video_feature):
        batch_size, max_seq, _ = video_feature.shape
        video_feature = video_feature.view((-1, video_feature.shape[-1])).float()
        mat_a = self.FC_ATT(video_feature)
        att_out = mat_a * video_feature
        f_vs = torch.mean(att_out, -1, keepdim=True).view(batch_size, max_seq)
        f_vs = self.FC_FVS(f_vs)
        return att_out.view(batch_size, max_seq, _), mat_a.view(batch_size, max_seq), f_vs
