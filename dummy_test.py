import importlib.util
import pathlib
import sys
import types
import torch

ROOT = pathlib.Path(__file__).resolve().parent
MODULE_PATH = ROOT / 'llavamini' / 'model' / 'llavamini_arch.py'


def stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class DummyDecoderLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(64, 64)

    def forward(self, x, attention_mask=None, position_ids=None):
        return (x + self.proj(x),)


def dummy_prepare_4d_causal_attention_mask(mask, shape, x, past_key_values_length):
    return mask


stub_module('llavamini')
stub_module(
    'llavamini.constants',
    IGNORE_INDEX=-100,
    IMAGE_TOKEN_INDEX=999,
    DEFAULT_IMAGE_PATCH_TOKEN='<im_patch>',
    DEFAULT_IM_START_TOKEN='<im_start>',
    DEFAULT_IM_END_TOKEN='<im_end>',
)
stub_module('llavamini.mm_utils', get_anyres_image_grid_shape=lambda *args, **kwargs: (1, 1))
stub_module('llavamini.model')
stub_module('llavamini.model.multimodal_encoder')
stub_module('llavamini.model.multimodal_projector')
stub_module('llavamini.model.multimodal_encoder.builder', build_vision_tower=lambda *args, **kwargs: None)
stub_module('llavamini.model.multimodal_projector.builder', build_vision_projector=lambda *args, **kwargs: None)
stub_module('transformers')
stub_module('transformers.models')
stub_module('transformers.models.llama')
stub_module('transformers.models.llama.modeling_llama', LlamaDecoderLayer=DummyDecoderLayer)
stub_module('transformers.modeling_attn_mask_utils', _prepare_4d_causal_attention_mask=dummy_prepare_4d_causal_attention_mask)

spec = importlib.util.spec_from_file_location('llavamini.model.llavamini_arch', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules['llavamini.model.llavamini_arch'] = module
spec.loader.exec_module(module)


class DummyVisionTower(torch.nn.Module):
    def forward(self, images):
        bsz = images.shape[0]
        tokens = 25
        hidden = 1024
        base = torch.arange(bsz * tokens * hidden, dtype=torch.float32).reshape(bsz, tokens, hidden)
        image_signal = images.float().mean(dim=(1, 2, 3), keepdim=True).view(bsz, 1, 1)
        return base / 1000.0 + image_signal


class DummyProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1024, 64)

    def forward(self, x):
        return self.proj(x)


class DummyBaseModel:
    def __init__(self):
        self._use_flash_attention_2 = False
        self.config = types.SimpleNamespace(_attn_implementation='eager', hidden_size=64)
        self.device = torch.device('cpu')
        self.dtype = torch.float32


class DummyInnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = DummyVisionTower()
        self.mm_projector = DummyProjector()
        self.prefusion_layers = torch.nn.ModuleList([DummyDecoderLayer()])
        self.base_model = DummyBaseModel()
        self.config = types.SimpleNamespace(hidden_size=64)
        self.build_compressor(types.SimpleNamespace(prefusion_layer_num=1, compressor_size=2, temporal_router_hidden_size=32, buffer_retriever_heads=8))

    def build_compressor(self, config):
        module.LlavaMiniMetaModel.build_compressor(self, config)

    def get_vision_tower(self):
        return self.vision_tower


class DummyMeta(module.LlavaMiniMetaForCausalLM):
    def __init__(self):
        self.device = torch.device('cpu')
        self.config = types.SimpleNamespace(temporal_pruning_keep_frames=2)
        self._model = DummyInnerModel()
        self.embedding = torch.nn.Embedding(4096, 64)

    def get_model(self):
        return self._model

    def get_input_embeddings(self):
        return self.embedding


def grad_norm(param):
    if param.grad is None:
        return None
    return float(param.grad.norm().item())


def main():
    torch.manual_seed(0)
    model = DummyMeta()
    input_ids = torch.tensor([[1, 999, 2, 3]])
    labels = torch.full_like(input_ids, -100)
    images = torch.randn(1, 3, 1, 3, 8, 8)

    frame_features = []
    for frame_idx in range(images.size(1)):
        image_feature, text_feature = model.encode_images_mini(images[:, frame_idx], input_ids=input_ids, labels=labels)
        frame_features.append(image_feature)
        if frame_idx == 0:
            print('single_frame_image_feature_shape', tuple(image_feature.shape))
            print('single_frame_text_feature_shape', tuple(text_feature.shape))

    stacked = torch.stack(frame_features, dim=1)
    pruned, info = model._apply_temporal_pruning(stacked)
    anchor, buffer_tokens = model._build_pyramid_outputs(frame_features[0], frame_features[0])

    target = torch.zeros_like(pruned)
    loss = torch.nn.functional.mse_loss(pruned, target) + 0.1 * buffer_tokens.pow(2).mean()
    loss.backward()

    print('stacked_frame_features_shape', tuple(stacked.shape))
    print('pruned_image_features_shape', tuple(pruned.shape))
    print('selected_frame_indices_shape', tuple(info['selected_indices'].shape))
    print('selected_frame_indices', info['selected_indices'].tolist())
    print('routing_weights', info['routing_weights'].detach().cpu().tolist())
    print('anchor_shape', tuple(anchor.shape))
    print('buffer_shape', tuple(buffer_tokens.shape))
    print('mse_loss', float(loss.item()))
    print('temporal_router.1.weight.grad_l2', grad_norm(model.get_model().temporal_router[1].weight))
    print('temporal_router.1.bias.grad_l2', grad_norm(model.get_model().temporal_router[1].bias))
    print('temporal_router.3.weight.grad_l2', grad_norm(model.get_model().temporal_router[3].weight))
    print('buffer_query.grad_l2', grad_norm(model.get_model().buffer_query))
    print('buffer_retriever.in_proj_weight.grad_l2', grad_norm(model.get_model().buffer_retriever.in_proj_weight))
    print('buffer_retriever.out_proj.weight.grad_l2', grad_norm(model.get_model().buffer_retriever.out_proj.weight))


if __name__ == '__main__':
    main()
