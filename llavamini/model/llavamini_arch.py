#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llavamini.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llavamini.mm_utils import get_anyres_image_grid_shape

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """ 
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)

    中文说明（与视频/图像 token 压缩相关）：
    - 视觉 backbone（比如 ViT）先把一帧图像切成固定网格的 patch token，常见是 24x24=576 个 token。
    - 这些 576 个 token 作为 K/V 输入到本模块，内部用 "grid_size**2" 个可学习查询（Q）做一次交叉注意力，
      把所有 patch 的信息汇聚到少量查询 token 上（例如 grid_size=2 时只留下 4 个 token）。
    - 后续只会保留这 "grid_size**2" 个聚合后的 token，原始的 576 个 patch token 在这里就被压缩/丢弃掉，
      只通过注意力贡献到这少量 token 的表示中。
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm,
            # dtype=torch.half
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size))).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        self.query.data.normal_(mean=0.0, std=0.02)
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads,batch_first=True)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        nn.init.constant_(self.ln_q.bias, 0)
        nn.init.constant_(self.ln_q.weight, 1.0)
        nn.init.constant_(self.ln_kv.bias, 0)
        nn.init.constant_(self.ln_kv.weight, 1.0)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def init_weights(self):
        self.query.data.normal_(mean=0.0, std=0.02)
        nn.init.constant_(self.ln_q.bias, 0)
        nn.init.constant_(self.ln_q.weight, 1.0)
        nn.init.constant_(self.ln_kv.bias, 0)
        nn.init.constant_(self.ln_kv.weight, 1.0)

    def forward(self, x, attn_mask=None,text=None):
        pos_embed = get_abs_pos(self.pos_embed, x.size(1)).type_as(x)
        Q=self.query
        x = self.kv_proj(x)
        x = self.ln_kv(x)
        N = x.shape[1]
        Q = self.ln_q(Q)
        out,attn =self.attn((Q + self.pos_embed.type_as(x)).unsqueeze(0).expand(x.size(0),Q.size(0),Q.size(1)),x + pos_embed.unsqueeze(0).type_as(x), x,attn_mask=attn_mask)
        return out,attn

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

class LlavaMiniMetaModel:

    def __init__(self, config):
        super(LlavaMiniMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
                
        self.init_build_compressor=False
        if hasattr(config,'compressor_size'):
            self.build_compressor(config)
            self.init_build_compressor=True

    def build_compressor(self,config):
        self.prefusion_layer_num= getattr(config,'prefusion_layer_num', 4)
        
        self.prefusion_layers=nn.ModuleList([LlamaDecoderLayer(self.base_model.config,layer_idx=i) for i in range(self.prefusion_layer_num)])
        if self.base_model.device.type != 'meta':
            self.prefusion_layers.to(self.base_model.device).to(self.base_model.dtype)
            
        self.compressor_size= getattr(config,'compressor_size', 2)
        self.compressor=Resampler(
            grid_size=self.compressor_size,
            embed_dim=1024,
            num_heads=8,
        )
        temporal_router_hidden = getattr(config, 'temporal_router_hidden_size', 256)
        temporal_router_input_dim = getattr(config, 'hidden_size', self.base_model.config.hidden_size)
        self.temporal_router = nn.Sequential(
            nn.LayerNorm(temporal_router_input_dim),
            nn.Linear(temporal_router_input_dim, temporal_router_hidden),
            nn.GELU(),
            nn.Linear(temporal_router_hidden, 1),
        )
        self.buffer_query = nn.Parameter(torch.randn(16, self.base_model.config.hidden_size) * 0.02)
        self.buffer_retriever = nn.MultiheadAttention(
            embed_dim=self.base_model.config.hidden_size,
            num_heads=getattr(config, 'buffer_retriever_heads', 8),
            batch_first=True,
        )
        self.buffer_retriever_norm = nn.LayerNorm(self.base_model.config.hidden_size)
        if self.base_model.device.type != 'meta':
            self.compressor.to(self.base_model.device).to(self.base_model.dtype)
            self.temporal_router.to(self.base_model.device).to(self.base_model.dtype)
            self.buffer_retriever.to(self.base_model.device).to(self.base_model.dtype)
            self.buffer_retriever_norm.to(self.base_model.device).to(self.base_model.dtype)
            self.buffer_query.data = self.buffer_query.data.to(self.base_model.device).to(self.base_model.dtype)
        print("#Vision Tokens:",self.compressor_size*self.compressor_size)
        self.load_prefusion_layers=False

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if not self.init_build_compressor:
            self.build_compressor(model_args)
            self.init_build_compressor=True

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            self.mm_projector = build_vision_projector(self.config)
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        for p in self.compressor.parameters():
            p.requires_grad = True
        self.compressor.init_weights()

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            if 'base_model.model.model.prefusion_layers.0.self_attn.q_proj.weight' in mm_projector_weights.keys():
                for name, module in self.spatial_w_text_projector.named_parameters():
                    module.data=mm_projector_weights[f"base_model.model.model.prefusion_layers.{name}"].data.type_as(module.data)
                    # module.requires_grad = False
                self.load_spatial_w_text_projector=True
                print("load pretrained prefusion_layers")

        if not self.load_prefusion_layers:
            if getattr(model_args, 'pretrain_prefusion', None):
                model_weights = torch.load(model_args.pretrain_prefusion, map_location='cpu')
                for name, module in self.prefusion_layers.named_parameters():
                    module.data=model_weights[f"{name}"].data.type_as(module.data)
                    module.requires_grad = True
                print(f"load pretrain_prefusion from {model_args.pretrain_prefusion}")
                self.load_prefusion_layers=True


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMiniMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def _retrieve_buffer_tokens(self, image_features, global_image_features):
        buffer_source = image_features
        if buffer_source.size(1) < 16:
            pad = buffer_source[:, -1:, :].expand(-1, 16 - buffer_source.size(1), -1)
            buffer_source = torch.cat([buffer_source, pad], dim=1)
        query = self.get_model().buffer_query.unsqueeze(0).expand(image_features.size(0), -1, -1).type_as(image_features)
        retrieved_buffer, _ = self.get_model().buffer_retriever(
            query=query,
            key=buffer_source,
            value=buffer_source,
        )
        retrieved_buffer = self.get_model().buffer_retriever_norm(retrieved_buffer + query)
        return retrieved_buffer

    def _build_pyramid_outputs(self, image_features, global_image_features):
        anchor_tokens = global_image_features[:, :4, :]
        if anchor_tokens.size(1) < 4:
            pad = anchor_tokens[:, -1:, :].expand(-1, 4 - anchor_tokens.size(1), -1)
            anchor_tokens = torch.cat([anchor_tokens, pad], dim=1)

        buffer_tokens = self._retrieve_buffer_tokens(image_features, global_image_features)
        return anchor_tokens, buffer_tokens

    def _apply_temporal_pruning(self, frame_features, frame_scores=None, keep_frames=None):
        if frame_features is None or frame_features.ndim != 4:
            return frame_features, None

        batch_size, temporal_len, token_len, hidden_size = frame_features.shape
        if temporal_len <= 1:
            flat_features = frame_features.reshape(batch_size, temporal_len * token_len, hidden_size)
            single_scores = frame_features.new_ones(batch_size, temporal_len)
            return flat_features, {
                'frame_scores': single_scores,
                'routing_weights': single_scores,
                'selected_indices': torch.zeros(batch_size, 1, dtype=torch.long, device=frame_features.device),
                'selected_scores': single_scores,
            }

        if keep_frames is None:
            keep_frames = getattr(self.config, 'temporal_pruning_keep_frames', 4)
        keep_frames = max(1, min(int(keep_frames), temporal_len))

        router_input = frame_features.float().mean(dim=2)
        router_logits = self.get_model().temporal_router(router_input).squeeze(-1)
        if frame_scores is not None:
            router_logits = router_logits + frame_scores.type_as(router_logits)
        routing_weights = torch.softmax(router_logits, dim=1)
        topk_weights, topk_indices = torch.topk(routing_weights, k=keep_frames, dim=1)
        topk_indices, sort_order = torch.sort(topk_indices, dim=1)
        topk_weights = torch.gather(topk_weights, 1, sort_order)
        gather_index = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, token_len, hidden_size)
        pruned_features = torch.gather(frame_features, 1, gather_index)
        weighted_features = pruned_features * topk_weights.unsqueeze(-1).unsqueeze(-1).type_as(pruned_features)
        return weighted_features.reshape(batch_size, keep_frames * token_len, hidden_size), {
            'frame_scores': router_logits,
            'routing_weights': routing_weights,
            'selected_indices': topk_indices,
            'selected_scores': topk_weights,
        }

    def encode_images_mini(self, images,input_ids,labels=None,modal='image'):
        if modal=='video': 
            # Batch operations on video frames can further improve efficiency and will be implemented in the future.
            pass

        else:
            all_text_embedding=self.get_input_embeddings()(input_ids.clamp(min=0)).detach()
            input_ids=input_ids*(labels==-100).int() if labels is not None else input_ids
            padding_mask=(input_ids<=0)
            
            text_embedding=all_text_embedding

            # images 形状: [bsz, parts, 3, H, W]
            #   - 对于单张图像或单帧视频，这里通常是 parts == 1；
            #   - 视觉 backbone 会把每一帧映射成 spa_len 个 patch token，常见 spa_len=24*24=576。
            bsz,parts,rgb,height,width=images.size()

            if parts==1:
                # standard resolution
                images=images[:,0]
                # 视觉编码：一帧图像 -> spa_len 个 patch token（比如 576 个）
                clip_image_features = self.get_model().get_vision_tower()(images)
                _,spa_len,d_im=clip_image_features.size()
                clip_image_features=clip_image_features.view(bsz,spa_len,d_im)

                # org_grid 一般是 24，对应 24x24 = 576 个 patch token
                org_grid=int(math.sqrt(spa_len))
                split_ratio=1

                # image_features 仍然保留的是一帧的所有 patch token（例如 576 个），
                # global_image_features 则作为全局特征后面会一起送入 mm_projector。
                image_features=clip_image_features
                global_image_features=clip_image_features

                # 关键一步：用 Resampler 做空间压缩
                #   - compressor 内部以所有 patch token 为 K/V，以少量 learnable query 为 Q 做一次 cross-attention；
                #   - 输出的 compressed_image_features 只有 grid_size**2 个 token，
                #     原始的 spa_len（例如 576）个 patch token 不再单独保留下来，
                #     只通过注意力聚合到这少量 token 里，相当于在这里丢弃了「全量 576 个 token」的显式表示。
                compressed_image_features,attn=self.get_model().compressor(image_features)

                compressed_image_features=self.get_model().mm_projector(compressed_image_features)
                global_image_features=self.get_model().mm_projector(global_image_features)
                anchor_image_features, buffer_image_features = self._build_pyramid_outputs(compressed_image_features, global_image_features)

                x=torch.cat([anchor_image_features,buffer_image_features,global_image_features,compressed_image_features,text_embedding],dim=1)
                mask=torch.cat((torch.zeros((padding_mask.size(0),anchor_image_features.size(1)+buffer_image_features.size(1)+global_image_features.size(1)+compressed_image_features.size(1)),device=padding_mask.device).bool(),padding_mask),dim=1)

            else:
                # high resolution
                images=images.view(bsz*parts,rgb,height,width)
                clip_image_features = self.get_model().get_vision_tower()(images)
                _,spa_len,d_im=clip_image_features.size()
                clip_image_features=clip_image_features.view(bsz,-1,spa_len,d_im)

                hd_ratio=int(math.sqrt(parts-1))
                org_grid=int(math.sqrt(spa_len))
                split_ratio=1
            
                hd_image_features=clip_image_features[:,:hd_ratio*hd_ratio].view(bsz,hd_ratio,hd_ratio,org_grid,org_grid,d_im).transpose(2,3).reshape(bsz,hd_ratio*org_grid,hd_ratio*org_grid,d_im)
                hd_image_features=hd_image_features.view(bsz,split_ratio,hd_ratio*org_grid//split_ratio,split_ratio,hd_ratio*org_grid//split_ratio,d_im).transpose(2,3).reshape(bsz*split_ratio*split_ratio,-1,d_im)

                global_image_features=clip_image_features[:,-1]
                
                compressed_image_features,attn=self.get_model().compressor(hd_image_features)

                compressed_image_features=self.get_model().mm_projector(compressed_image_features)
                compressed_image_features=compressed_image_features.view(bsz,split_ratio*split_ratio,compressed_image_features.size(-2),compressed_image_features.size(-1)).reshape(bsz,-1,compressed_image_features.size(-1))
                global_image_features=self.get_model().mm_projector(global_image_features)
                anchor_image_features, buffer_image_features = self._build_pyramid_outputs(compressed_image_features, global_image_features)

                d=global_image_features.size(-1)
                hd_image_features_all=self.get_model().mm_projector(clip_image_features[:,:-1]).view(bsz,hd_ratio,hd_ratio,org_grid,org_grid,-1).transpose(2,3).reshape(bsz,-1,d)

                x=torch.cat([anchor_image_features,buffer_image_features,hd_image_features_all,global_image_features,compressed_image_features,text_embedding],dim=1)
                mask=torch.cat((torch.zeros((padding_mask.size(0),anchor_image_features.size(1)+buffer_image_features.size(1)+hd_image_features_all.size(1)+global_image_features.size(1)+compressed_image_features.size(1)),device=padding_mask.device).bool(),padding_mask),dim=1)

            if getattr(self.get_model().base_model, "_use_flash_attention_2", False) or getattr(self.get_model().base_model.config, "_attn_implementation", "") == "flash_attention_2":
                attention_mask=(~mask).int()
            else: 
                attention_mask=_prepare_4d_causal_attention_mask(~mask, (x.size(0), x.size(1)), x, 0)
            
            position_ids = (~mask).int().long().cumsum(-1) - 1
            position_ids.masked_fill_((~mask).int() == 0, 1)
            

            # modality pre-fusion
            for layer in self.get_model().prefusion_layers:
                x = layer(x, attention_mask=attention_mask, position_ids=position_ids)[0]

            fusion_text_features=x[:,-1*input_ids.size(1):,:]
            compressed_image_features=x[:,-1*input_ids.size(1)-1*compressed_image_features.size(1):-1*input_ids.size(1),:]
            fusion_text_features=fusion_text_features*(~padding_mask).unsqueeze(-1).int()+all_text_embedding*padding_mask.unsqueeze(-1)
            
            return compressed_image_features,fusion_text_features

    

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list:
            # video 模式：images 是一个列表，每个元素对应一个样本；
            #   - images[i].ndim == 5 时，表示该样本是视频，包含 temporal_len 帧；
            #   - 每一帧会单独调用 encode_images_mini，经由 vision_tower + compressor 从 576 patch token
            #     压缩成少量聚合 token，然后在时间维度上再做一次帧级剪枝（_apply_temporal_pruning）。
            image_features=[]
            text_features=[]
            for i in range(len(images)):
                if images[i].ndim==5:

                    image=images[i].unsqueeze(0)
                    temporal_len=image.size(1)
                    image_features_list = []
                    text_features_sum = 0
                    # 逐帧处理视频：
                    #   - 第一步：对每一帧调用 encode_images_mini，内部会先得到 spa_len（通常 576）个 patch token，
                    #     再通过 compressor 压缩成少量 token（grid_size**2 个），这一层是「空间维度」上的 token 压缩。
                    for frame_idx in range(temporal_len):
                        frame_image_features,frame_text_features = self.encode_images_mini(image[:,frame_idx],input_ids[i:i+1],labels[i:i+1] if labels is not None else None)
                        image_features_list.append(frame_image_features)
                        text_features_sum=text_features_sum+frame_text_features.float()
                    # 第二步：把所有帧的压缩后 token 堆叠成 [B, T, token_len, D]，
                    # 然后交给 _apply_temporal_pruning 在时间维度上做帧级剪枝：
                    #   - 通过 temporal_router 计算每一帧的重要性分数；
                    #   - 只保留 top-k 帧（keep_frames），其它帧的所有视觉 token 直接被丢弃；
                    #   - 这样在空间上已经从 576 -> grid_size**2，在时间上又从 T 帧 -> keep_frames 帧。
                    stacked_frame_features = torch.stack(image_features_list, dim=1)
                    pruned_image_features, _ = self._apply_temporal_pruning(stacked_frame_features)
                    image_features.append(pruned_image_features.squeeze(0))
                    text_features.append((text_features_sum/temporal_len).squeeze(0).type_as(frame_text_features))


                else:
                    image_feature,text_feature=self.encode_images_mini(images[i].unsqueeze(0),input_ids[i:i+1],labels[i:i+1] if labels is not None else None)
                    image_features.append(image_feature.squeeze(0))
                    text_features.append(text_feature.squeeze(0))
        else:
            # image
            if images.ndim==6:
                temporal_len=images.size(1)
                image_features_list = []
                text_features_sum = 0
                for frame_idx in range(temporal_len):
                    frame_image_features,frame_text_features = self.encode_images_mini(images[:,frame_idx],input_ids=input_ids,labels=labels)
                    image_features_list.append(frame_image_features)
                    text_features_sum=text_features_sum+frame_text_features.float()
                stacked_frame_features = torch.stack(image_features_list, dim=1)
                image_features, temporal_pruning_info = self._apply_temporal_pruning(stacked_frame_features)
                text_features=(text_features_sum/temporal_len).type_as(frame_text_features)

            else:
                image_features,text_features = self.encode_images_mini(images,input_ids=input_ids,labels=labels)


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        _labels=labels
        _attention_mask=attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            cur_text_features=text_features[batch_idx]
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = cur_text_features[_attention_mask[batch_idx]]
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_input_embeds_no_im=[]
            for i in range(len(image_token_indices) - 1):
                cur_input_embeds_no_im.append(cur_text_features[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            assert cur_new_input_embeds.size(0)==cur_new_labels.size(0)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    try:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                    except:
                        raise ValueError("new_labels_padded error")
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            # if model_args.pretrain_mm_mlp_adapter:
            if getattr(model_args, 'pretrain_mm_mlp_adapter', None):
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                if 'model.embed_tokens.weight' in mm_projector_weights.keys():
                    embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                    # assert num_new_tokens == 2
                    if input_embeddings.shape == embed_tokens_weight.shape:
                        input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                    elif embed_tokens_weight.shape[0] == num_new_tokens:
                        input_embeddings[-num_new_tokens:] = embed_tokens_weight
                    else:
                        raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
