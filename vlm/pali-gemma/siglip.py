import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, embed_dim, image_size, patch_size, num_channels):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid"
        )
        self.num_patches = (self.image_size // patch_size) ** 2
        self.num_positions = self.num_patches
        self.pos_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            'positions_ids',
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False,
        )

    def forward(self, pixel_values):
        _, _, h, w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embedding = patch_embeds.flatten(2)
        embedding = embedding.transpose(1, 2)
        embedding = embedding + self.pos_embedding(self.positions_ids)
        return embedding


class SiglipAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_score = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_weights = F.softmax(attn_score, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(attn_output)
        return out, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, intermediate_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_layer = SiglipAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_layer = SiglipMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.attn_layer(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp_layer(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, intermediate_size, num_hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList([SiglipEncoderLayer(embed_dim, num_heads, hidden_size, intermediate_size) for _ in range(num_hidden_layers)])

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, embed_dim, image_size, patch_size, num_channels, num_heads, hidden_size, intermediate_size, num_hidden_layers):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(embed_dim, image_size, patch_size, num_channels)
        self.encoder = SiglipEncoder(embed_dim, num_heads, hidden_size, intermediate_size, num_hidden_layers)
        self.post_layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_states = self.encoder(input_embeds=hidden_states)
        last_hidden_states = self.post_layer_norm(last_hidden_states)
        return last_hidden_states


class SiglipVisionModel(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, num_heads, hidden_size, intermediate_size, num_hidden_layers):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(image_size, patch_size, num_channels, num_heads, hidden_size, intermediate_size, num_hidden_layers)

    def forward(self, pixel_values):
        return self.vision_model(pixel_values)

