siglip_config =   {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 16,
    "layer_norm_eps": 1e-6,
    "attention_dropout": 0.0,
    "num_image_tokens":  None,
}

gemma_config =  {
    "head_dim": 256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "pad_token_id": None,
}

pali_config = {
    "vision_config": None,
    "text_config": None,
    "ignore_index": -100,
    "image_token_index": 256000,
    "vocab_size": 257152,
    "projection_dim": 2048,
    "hidden_size": 2048,
    "pad_token_id": None,
}