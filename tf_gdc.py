import torch
import torch.nn as nn
import numpy as np
import os


# ==========================================================
# 🌟 核心创新模块：跨变量耦合补丁提取器
# ==========================================================
class CrossVariatePatchExtractor(nn.Module):
    def __init__(self, seq_len=512, patch_len=64, variates=3, d_model=256):
        super().__init__()
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.variates = variates
        self.d_model = d_model

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.variate_attention = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True, dim_feedforward=512, dropout=0.1
        )
        self.attn_weights = None  # 🚀 新增：专门用于临时存放 3x3 注意力矩阵

    def forward(self, x, ablation_mode='full'):
        B, L, V = x.shape
        x = x.permute(0, 2, 1)  # [B, 3, L]

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # [B, 3, patch_num, patch_len]
        x_emb = self.patch_embedding(x)  # [B, 3, patch_num, d_model]

        x_emb = x_emb.permute(0, 2, 1, 3)  # [B, patch_num, 3, d_model]
        x_emb = x_emb.reshape(B * self.patch_num, V, self.d_model)  # [B*patch_num, 3, d_model]

        if ablation_mode == 'Ablation_wo_CrossAttn':
            coupled_features = x_emb  # 独立变量，不做耦合
            self.attn_weights = None
        else:
            # 🚀 稳扎稳打提取法：只在测试阶段，单独调用底层的 self_attn 来获取 V-I-T 权重矩阵
            if not self.training:
                # 获取注意力权重，形状为 [B*patch_num, 3, 3]
                _, attn_weights = self.variate_attention.self_attn(x_emb, x_emb, x_emb, need_weights=True)
                self.attn_weights = attn_weights.detach()

            # 核心：物理逻辑校验 (正常的前向传播不变)
            coupled_features = self.variate_attention(x_emb)

        coupled_features = coupled_features.reshape(B, self.patch_num, V, self.d_model)
        return coupled_features  # [B, patch_num, 3, d_model]


# ==========================================================
# 👑 主打战车：TF-GDC 满血融合版 (支持全套消融 & 黄金参数)
# ==========================================================
class TF_GDC(nn.Module):
    def __init__(self, win_size=512, input_c=3, patch_size=64, d_model=256, ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.win_size = win_size
        self.input_c = input_c
        self.patch_size = patch_size
        self.patch_num = win_size // patch_size
        self.d_model = d_model

        self.viz_cache = {}  # 🚀 新增：可视化数据缓存字典

        # 1. 前端物理感知层
        self.variate_extractor = CrossVariatePatchExtractor(
            seq_len=win_size, patch_len=patch_size, variates=input_c, d_model=d_model
        )
        self.fusion_linear = nn.Linear(input_c * d_model, d_model)

        # 2. 双分支架构 (Dual-Branch)
        encoder_layer_A = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dim_feedforward=512)
        self.branch_a_encoder = nn.TransformerEncoder(encoder_layer_A, num_layers=2)

        self.branch_b_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )
        self.norm_b = nn.LayerNorm(d_model)

        # 🛡️ 门控层 (Gate)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # 3. 语义字典 (Semantic Dictionary)
        dict_path = 'dataset/Battery/semantic_anchors.npy'
        if os.path.exists(dict_path):
            anchors_data = np.load(dict_path)
            self.semantic_memory = nn.Parameter(torch.from_numpy(anchors_data).float(), requires_grad=False)
        else:
            self.semantic_memory = nn.Parameter(torch.randn(5, 768), requires_grad=False)

        self.text_projector = nn.Linear(768, d_model)
        self.dict_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.norm_fusion = nn.LayerNorm(d_model)

        # 4. 重构头与投影头
        self.projection_head = nn.Linear(d_model, d_model)
        self.reconstruction_head = nn.Linear(d_model, patch_size * input_c)

    def forward(self, x):
        B = x.shape[0]

        # --- 步骤 1：跨变量提取 ---
        coupled_feat = self.variate_extractor(x, self.ablation_mode)
        flat_feat = coupled_feat.reshape(B, self.patch_num, self.input_c * self.d_model)
        fused_feat = self.fusion_linear(flat_feat)

        # --- 步骤 2：双分支编码 ---
        out_branch_a = self.branch_a_encoder(fused_feat)

        x_permuted = fused_feat.permute(0, 2, 1)
        out_branch_b = self.branch_b_encoder(x_permuted).permute(0, 2, 1)
        out_branch_b = self.norm_b(out_branch_b)

        if self.ablation_mode == 'Ablation_wo_BranchA':
            out_branch_a = torch.zeros_like(out_branch_a)

        # --- 步骤 3：自适应门控融合 ---
        gate_weight = None  # 赋初值防报错
        if self.ablation_mode == 'Ablation_wo_Gate':
            combined_query = out_branch_a + out_branch_b
        else:
            gate_weight = self.gate(torch.cat([out_branch_a, out_branch_b], dim=-1))
            combined_query = gate_weight * out_branch_a + (1 - gate_weight) * out_branch_b

        # --- 步骤 4：语义字典增强 ---
        if self.ablation_mode == 'Ablation_wo_Dict':
            final_feat = combined_query
        else:
            if self.ablation_mode == 'Ablation_wo_Prompts':
                temp_memory = torch.randn_like(self.semantic_memory)
                memory_proj = self.text_projector(temp_memory)
            else:
                memory_proj = self.text_projector(self.semantic_memory)

            key_value = memory_proj.unsqueeze(0).repeat(B, 1, 1)
            dict_out, _ = self.dict_attention(combined_query, key_value, key_value)
            final_feat = self.norm_fusion(combined_query + dict_out)

        # --- 步骤 5：重构波形 ---
        out = self.reconstruction_head(final_feat)
        out = out.reshape(B, self.win_size, self.input_c)

        # 🚀 补救核心：让特征通过投影头，进入真正的语义对齐空间！
        proj_a = self.projection_head(out_branch_a)
        proj_b = self.projection_head(out_branch_b)

        # 🚀 稳扎稳打：将对齐后的核心变量打包进类的缓存字典！
        if not self.training:
            self.viz_cache = {
                'attn_weights': self.variate_extractor.attn_weights.cpu().numpy() if self.variate_extractor.attn_weights is not None else None,
                'gate_weights': gate_weight.detach().cpu().numpy() if gate_weight is not None else None,
                'H_patch': proj_a.detach().cpu().numpy(),  # 👈 已换成投影后的高级语义特征
                'H_in_patch': proj_b.detach().cpu().numpy(),  # 👈 已换成投影后的高级语义特征
                'anchors': memory_proj.detach().cpu().numpy() if self.ablation_mode != 'Ablation_wo_Dict' else None
            }

        # ⚠️ 原封不动返回 3 个值
        return out, proj_a, proj_b