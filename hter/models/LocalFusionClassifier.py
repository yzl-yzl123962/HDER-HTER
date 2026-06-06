import torch
import torch.nn as nn
from .main_layer import CrossTransformer,Transformer,ConvModulOperationSpatialAttention
from .bert import BertTextEncoder
from .temporal_encoder import TemporalResidualEncoder
from einops import repeat

class LocalFusionClassifier(nn.Module):
    def __init__(self,args, dim=128, num_classes=1, fusion_depth=2, heads=8, mlp_dim=128):
        super(LocalFusionClassifier, self).__init__()
        args = args.model
        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args.bert_pretrained)
        self.audio_temporal_encoder = TemporalResidualEncoder(
            args.a_proj_dim,
            enabled=getattr(args, 'use_av_tcn', False),
            kernel_size=getattr(args, 'av_tcn_kernel_size', 3),
            dropout=getattr(args, 'av_tcn_dropout', 0.1),
            scale_init=getattr(args, 'av_tcn_scale_init', 0.01),
        )
        self.visual_temporal_encoder = TemporalResidualEncoder(
            args.v_proj_dim,
            enabled=getattr(args, 'use_av_tcn', False),
            kernel_size=getattr(args, 'av_tcn_kernel_size', 3),
            dropout=getattr(args, 'av_tcn_dropout', 0.1),
            scale_init=getattr(args, 'av_tcn_scale_init', 0.01),
        )

        self.proj_l = nn.Sequential(
            nn.Linear(args.l_proj_dim, args.proj_dst_dim),
            Transformer(num_frames=args.l_proj_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(args.a_proj_dim, args.proj_dst_dim),
            Transformer(num_frames=args.a_proj_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(args.v_proj_dim, args.proj_dst_dim),
            Transformer(num_frames=args.v_proj_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )

        self.conv_att_v = ConvModulOperationSpatialAttention(args.v_proj_dim, kernel_size=3)

        self.fusion_transformer = CrossTransformer(
            source_num_frames=args.l_proj_length + args.token_length,
            tgt_num_frames=max(args.a_proj_length, args.v_proj_length) + args.token_length,
            dim=dim,
            depth=fusion_depth,
            heads=heads,
            mlp_dim=mlp_dim
        )


        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self, video_feat, audio_feat, text_feat, return_modal=False):

        text_feat = self.bertmodel(text_feat)
        audio_feat = self.audio_temporal_encoder(audio_feat)
        video_feat = self.visual_temporal_encoder(video_feat)

        video_feat = video_feat.unsqueeze(-1)
        video_feat=video_feat.permute(0, 2, 1, 3)
        video_feat=self.conv_att_v(video_feat)
        video_feat =video_feat.squeeze(-1)
        video_feat= video_feat.permute(0, 2, 1)

        audio_feat = self.proj_a(audio_feat)  # (batch_size, seq_len, dim)
        video_feat = self.proj_v(video_feat)  # (batch_size, seq_len, dim)
        text_feat = self.proj_l(text_feat)     # (batch_size, seq_len, dim)

        fused_feat = self.fusion_transformer(text_feat, audio_feat, video_feat)  # (batch_size, seq_len, dim)

        local_feat = fused_feat[:, 0, :]  # (batch_size, dim)

        if return_modal:
            modal_feats = {
                "text": text_feat.mean(dim=1),
                "visual": video_feat.mean(dim=1),
                "audio": audio_feat.mean(dim=1),
                "av": torch.cat([video_feat.mean(dim=1), audio_feat.mean(dim=1)], dim=-1),
            }
            return local_feat, modal_feats

        return local_feat
def build_model():
    model =LocalFusionClassifier()
    return model
