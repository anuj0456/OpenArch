from typing import Optional

import torch
from torch import nn

from siglip import SiglipVisionModel
from decoder import GemmaForCausalLM
from utils import KVCache


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.vision.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, x):
        hidden_states = self.linear(x)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision = SiglipVisionModel(config.vision_config)
        self.multimodal = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text.config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(self,
                                             image_features: torch.Tensor,
                                             input_embeds: torch.Tensor,
                                             input_ids: torch.Tensor,
                                             attention_mask: torch.Tensor,
                                             kv_cache:Optional[KVCache]=None):
        _, _, embed_dim = image_features.shape
        batch_size, seq_len, = input_ids.shape
        image_features = image_features.view(batch_size * seq_len, embed_dim)
        dtype, device = input_embeds.dtype, input_embeds.device

        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        final_embedding = torch.zeros(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        dtype, device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(self, input_ids, pixel_values, attention_mask, kv_cache=None):
        assert torch.all(attention_mask == 1), "the input cannot be padded"

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        selected_image_features = self.vision_tower(pixel_values.to(input_embeds.dtype))
        image_features = self.multi_modal_projector(selected_image_features)

        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(attention_mask=attention_mask, kv_cache=kv_cache, position_ids=position_ids, input_embeds=input_embeds)

        return outputs