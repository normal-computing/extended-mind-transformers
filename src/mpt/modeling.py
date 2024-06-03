# Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code has been adapted from Mosaic ML and Huggingface and inherits the above lisence.
# The original code can be found here:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# We annotate the edited code below with 'EM' comments to indicate where we have made changes.
"""PyTorch MPT model."""

import math
from typing import Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from torch.linalg import vector_norm
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn import functional as F
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from src.mpt.configuration import ExtendedMptConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "mosaicml/mpt-7b"
_CONFIG_FOR_DOC = "MptConfig"


# Copied from transformers.models.bloom.modeling_bloom._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty(
        (target_length, target_length + past_key_values_length),
        dtype=torch.bool,
        device=device,
    )
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask


# Copied from transformers.models.bloom.modeling_bloom._expand_mask
def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def build_mpt_alibi_tensor(
    num_heads,
    sequence_length,
    sequence_length_with_past,
    alibi_bias_max=8,
    device=None,
    for_ae=False,
    topk=None,
):
    r"""
    Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292
    """
    if not for_ae:
        alibi = torch.arange(
            1 - sequence_length, 1, dtype=torch.int32, device=device
        ).view(1, 1, 1, sequence_length)
    else:  # EM: All memory tokens get same bias
        alibi = (
            torch.tensor(-sequence_length_with_past, dtype=torch.int32, device=device)
            .repeat(sequence_length * topk)
            .view(1, 1, 1, sequence_length * topk)
        )
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))

    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.float32, device=device)
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / torch.pow(2, base)
    slopes = slopes.view(1, num_heads, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:num_heads]

    alibi = alibi * slopes
    return alibi.squeeze(0)


class ExtendedMptAttention(nn.Module):
    """Multi-head self attention.
    Using torch or triton attention implemetation enables user to also use additive bias.
    """

    def __init__(self, config: ExtendedMptConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.head_dim = self.hidden_size // self.n_heads
        self.softmax_scale = config.attn_config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

        self.attn_dropout_p = config.attn_config.attn_pdrop
        self.Wqkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        long_range_past_key_value=None,
        topk=None,
        faiss_indexes=None,
        mask_by_sim=None,
        sim_threshold=None,
        position_bias_ae=None,
        current_layer=None,
        output_retrieved_memory_idx=False,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        mixed_qkv = self.Wqkv(hidden_states)
        query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
        query_states = query_states.reshape(
            batch_size, seq_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.reshape(
            batch_size, seq_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.reshape(
            batch_size, seq_length, self.n_heads, self.head_dim
        ).transpose(1, 2)

        if past_key_value is not None:
            if len(past_key_value) != 0:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states)
        bsz, nh, s_q, d = query_states.shape

        attention_scores = (
            torch.matmul(query_states, key_states.transpose(-1, -2))
            * self.softmax_scale
        )
        key_length = key_states.shape[-2]
        query_length = (
            seq_length
            if past_key_value is None
            else seq_length + past_key_value[0].shape[2]
        )
        if position_bias is not None:
            if len(position_bias.shape) != 3:
                raise ValueError(
                    f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}"
                )

            position_bias_query_index = max(0, position_bias.size(1) - query_length)
            position_bias_key_index = max(0, position_bias.size(2) - key_length)

            position_bias = position_bias[
                :, position_bias_query_index:, position_bias_key_index:
            ]

            attention_scores = attention_scores + position_bias

        # EM: Retrieve memories from cache or faiss indexes
        if long_range_past_key_value is not None or faiss_indexes is not None:
            if long_range_past_key_value is not None:  # Manual store
                k_cache, v_cache = long_range_past_key_value
                s_cache = k_cache.size(-2)

                k_cache = k_cache.to(key_states.device)
                v_cache = v_cache.to(key_states.device)

                # Normalize query and key vectors
                q_n = query_states / vector_norm(
                    query_states, ord=2, dim=-1, keepdim=True
                ) 
                k_n = k_cache / vector_norm(k_cache, ord=2, dim=-1, keepdim=True)
                sim = q_n.matmul(k_n.transpose(-1, -2))
                if s_cache < topk:   # number of tokens in cache < topk
                    topk = s_cache
                val, idx = torch.topk(sim, k=topk, dim=-1)  # Retrieve topk memories

                reshaped_idx = idx.reshape(bsz, nh, s_q * topk)
                selected_k = k_cache.gather(
                    dim=-2, index=reshaped_idx.unsqueeze(-1).expand(-1, -1, -1, d)
                )
                selected_v = v_cache.gather(
                    dim=-2, index=reshaped_idx.unsqueeze(-1).expand(-1, -1, -1, d)
                )

            elif faiss_indexes is not None:  # FAISS indexes
                kn_index, kv_index = faiss_indexes
                q_n = query_states / vector_norm(
                    query_states, ord=2, dim=-1, keepdim=True
                )
                # One-hot encoding for layer, head to only retrieve memories from the same layer, head
                one_hot_encodings = (
                    F.one_hot(
                        torch.arange(0, nh * self.n_layers, device=query_states.device)
                    )
                    * 10
                )
                q_n = torch.concat(
                    [
                        rearrange(q_n, "b h s d -> b (h s) d", h=nh),
                        one_hot_encodings[nh * current_layer : nh * (current_layer + 1)]
                        .unsqueeze(0)
                        .repeat_interleave(repeats=query_states.size(-2), dim=-2),
                    ],
                    dim=-1,
                ).squeeze()

                if kn_index.ntotal / (nh * self.n_layers) < topk:
                    topk = int(kn_index.ntotal / (nh * self.n_layers))

                val, idx = kn_index.search(q_n.to("cpu").detach().numpy(), k=topk)
                val = torch.tensor(val - 100).reshape(bsz, nh, s_q, topk)  #Similarity includes scale factor from one-hot encoding
                reshaped_idx = torch.tensor(
                    idx % (kn_index.ntotal / (nh * self.n_layers))
                ).reshape(bsz, nh, s_q * topk)

                # Retrieve tensors
                selected_k = rearrange(
                    torch.tensor(kv_index.reconstruct_batch(idx.flatten()))[:, :d],
                    "(h s) d -> 1 h s d",
                    h=nh,
                ).to(query_states.device)
                selected_v = rearrange(
                    torch.tensor(kv_index.reconstruct_batch(idx.flatten()))[:, d:],
                    "(h s) d -> 1 h s d",
                    h=nh,
                ).to(query_states.device)

            selected_key_length = selected_k.size(-2)
            key_length += selected_key_length
            attention_scores_cache = (
                query_states.matmul(selected_k.transpose(-1, -2)) * self.softmax_scale
            )
            # EM: Mask by similarity
            if mask_by_sim:
                sim_mask = (
                    rearrange(~(val > sim_threshold).bool(), "b h s i -> b h (s i)")
                    .unsqueeze(-2)
                    .expand(-1, -1, s_q, -1)
                ).to(query_states.device)

                attention_scores_cache = attention_scores_cache.masked_fill(
                    sim_mask, torch.finfo(query_states.dtype).min
                )

            # EM: Add position bias to cache
            if position_bias_ae is not None:
                if len(position_bias_ae.shape) != 3:
                    raise ValueError(
                        f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias_ae.shape)}"
                    )

                position_bias_query_index = max(
                    0, position_bias_ae.size(1) - query_length
                )
                position_bias_key_index = max(
                    0, position_bias_ae.size(2) - selected_key_length
                )

                position_bias_ae = position_bias_ae[
                    :, position_bias_query_index:, position_bias_key_index:
                ]

                attention_scores_cache = attention_scores_cache + position_bias_ae

            # EM: Concatenate cache and current attention weights, values
            attention_scores = torch.cat(
                [attention_scores_cache, attention_scores], dim=-1
            )  # Concat attention scores, values
            value_states = torch.cat([selected_v, value_states], dim=-2)

        # EM: Create mask for external memories, queries only attend to their own memories
        def _create_external_memories_mask(k, s_q, device):
            mask = torch.zeros(s_q, s_q * k, device=device, dtype=torch.bool)
            for i in range(s_q):
                mask[i, i * k : (i + 1) * k] = 1
            return ~mask

        if attention_mask is not None:
            # EM: Concatenate attention mask with external memories mask
            if long_range_past_key_value is not None or faiss_indexes is not None:
                mask = _create_external_memories_mask(
                    k=topk, s_q=s_q, device=attention_scores.device
                )
                attention_mask = attention_mask.squeeze(dim=0).squeeze(dim=0)
                attention_mask = torch.cat([mask, attention_mask], dim=1)
            attention_scores = attention_scores.masked_fill(
                attention_mask, torch.finfo(query_states.dtype).min
            )

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(
            value_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attn_dropout_p, training=self.training
        )

        context_states = torch.matmul(attn_weights, value_states)
        context_states = (
            context_states.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_length, -1)
        )
        attn_output = self.out_proj(context_states)

        if not output_retrieved_memory_idx or (long_range_past_key_value is None and faiss_indexes is None):
            reshaped_idx = None

        return attn_output, attn_weights, past_key_value, reshaped_idx


class MptMLP(nn.Module):
    def __init__(self, config: ExtendedMptConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.act = nn.GELU(approximate="none")
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=False)
        self.hidden_dropout = config.attn_config.attn_pdrop

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.act(self.up_proj(hidden_states))

        intermediate_output = self.down_proj(hidden_states)

        output = F.dropout(
            intermediate_output, p=self.hidden_dropout, training=self.training
        )
        output = output + residual

        return output


class MptBlock(nn.Module):
    """MPTBlock"""

    def __init__(self, config: ExtendedMptConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.norm_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # backward compatibility with weights on the Hub
        self.norm_1.bias = None

        self.num_heads = config.n_heads
        self.attn = ExtendedMptAttention(config)

        self.norm_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # backward compatibility with weights on the Hub
        self.norm_2.bias = None

        self.ffn = MptMLP(config)

        self.dropout_rate = config.attn_config.attn_pdrop
        self.resid_attn_dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_retrieved_memory_idx: bool = False,
        topk: int = None,
        long_range_past_key_value: Optional[Tuple[torch.Tensor]] = None,
        faiss_indexes: Tuple = None,
        position_bias_ae=None,
        current_layer: int = None,
        mask_by_sim: bool = False,
        sim_threshold: float = None,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.norm_1(hidden_states)

        residual = hidden_states

        # Self attention.
        attn_outputs, attn_weights, past_key_value, reshaped_idx = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past,
            long_range_past_key_value=long_range_past_key_value,
            topk=topk,
            faiss_indexes=faiss_indexes,
            position_bias_ae=position_bias_ae,
            current_layer=current_layer,
            mask_by_sim=mask_by_sim,
            sim_threshold=sim_threshold,
            output_retrieved_memory_idx=output_retrieved_memory_idx,
        )

        hidden_states = self.resid_attn_dropout(attn_outputs) + residual

        layernorm_output = self.norm_2(hidden_states)

        # Get residual
        residual = hidden_states

        # MLP.
        output = self.ffn(layernorm_output, residual)
        outputs = (output,)

        if use_cache:
            outputs += (past_key_value,)

        if output_attentions:
            outputs += (attn_weights,)
        if output_retrieved_memory_idx:
            outputs += (reshaped_idx,)

        return outputs  # hidden_states, present, attentions


class MptPreTrainedModel(PreTrainedModel):
    """MPT Pretrained Model"""

    config_class = ExtendedMptConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MptBlock"]
    _keys_to_ignore_on_load_missing = [r"lm_head.*."]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False):
        if isinstance(module, ExtendedMptConfig):
            module.gradient_checkpointing = value

    @staticmethod
    def _convert_to_mpt_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Mpt, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].reshape(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


MPT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ExtendedMptConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]`
            (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.

            Each element of `past_key_values` is a tuple (past_key, past_value):
            - past_key: [batch_size * num_heads, head_dim, kv_length]
            - past_value: [batch_size * num_heads, kv_length, head_dim]
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        use_external_mind (`bool`, *optional*, defaults to `True`):
            Whether to attend to external memories.
        long_range_past_key_values (`List[Tuple[torch.FloatTensor]]`, *optional*, defaults to None):
            Manual store for memories.
        faiss_indexes (`Tuple[faiss.swigfaiss_avx2.IndexFlatIP]`, *optional*, defaults to None):
            Vector store for memories.
        topk (`int`, *optional*, defaults to `10`):
            Number of external memories for each query token to retrieve and attend to.
"""


@add_start_docstrings(
    "The bare Mpt Model transformer outputting raw hidden-states without any specific head on top.",
    MPT_START_DOCSTRING,
)
class ExtendedMptModel(MptPreTrainedModel):
    """Extended MPT Model"""

    def __init__(self, config: ExtendedMptConfig):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_heads

        # Embedding + LN Embedding
        self.wte = nn.Embedding(config.vocab_size, self.hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([MptBlock(config) for _ in range(config.n_layers)])

        # Final Layer Norm
        self.norm_f = LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        # backward compatibility with weights on the Hub
        self.norm_f.bias = None

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        self.mask_by_sim = config.attn_config.mask_by_sim
        self.sim_threshold = config.attn_config.sim_threshold
        self.topk = config.attn_config.topk
        self.use_external_mind = config.use_external_mind
        self.use_external_mind_by_layer = config.attn_config.use_external_mind_by_layer

    def get_input_embeddings(self):
        return self.wte

    def build_mpt_alibi_tensor(
        self,
        num_heads,
        sequence_length,
        sequence_length_with_past,
        alibi_bias_max=8,
        device=None,
        for_ae=None,
        topk=None,
    ):
        return build_mpt_alibi_tensor(
            num_heads,
            sequence_length,
            sequence_length_with_past,
            alibi_bias_max,
            device,
            for_ae=for_ae,
            topk=topk,
        )

    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        if input_shape[1] + past_key_values_length != attention_mask.shape[1]:
            raise ValueError(
                "Attention mask shape should be (batch_size, seq_length + past_key_values_length)"
                f" but is {attention_mask.shape} with input_ids shape {input_shape} and past length"
                f" {past_key_values_length}."
            )
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved_memory_idx: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_external_mind: Optional[bool] = None,
        long_range_past_key_values: Optional[list[Tuple[torch.FloatTensor]]] = None,
        faiss_indexes: Tuple = None,
        topk: int = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_retrieved_memory_idx = (
            output_retrieved_memory_idx
            if output_retrieved_memory_idx is not None
            else False
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_external_mind = (
            use_external_mind
            if use_external_mind is not None
            else self.use_external_mind
        )
        topk = topk if topk is not None else self.topk

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.blocks))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_idx = () if output_retrieved_memory_idx else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), device=hidden_states.device
            )
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_mpt_alibi_tensor(
            self.num_heads,
            self.config.max_seq_len,
            seq_length_with_past,
            device=hidden_states.device,
        )
        # EM: Alibi tensor for retrieved kvs
        alibi_ae = self.build_mpt_alibi_tensor(
            self.num_heads,
            seq_length,
            seq_length_with_past,
            device=hidden_states.device,
            for_ae=True,
            topk=topk,
        )

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            long_range_past_key_value = (
                long_range_past_key_values[i]
                if (
                    long_range_past_key_values is not None
                    and self.use_external_mind_by_layer[i]
                    and use_external_mind is True
                )
                else None
            )
            if long_range_past_key_value is not None and faiss_indexes is not None:
                raise NotImplementedError(
                    """Using faiss and passing key value pairs
                    manually are mutually exclusive right now."""
                )
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        )

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_retrieved_memory_idx=output_retrieved_memory_idx,
                    position_bias=alibi,
                    position_bias_ae=alibi_ae,
                    topk=topk,
                    long_range_past_key_value=long_range_past_key_value,
                    faiss_indexes=faiss_indexes,
                    mask_by_sim=self.mask_by_sim,
                    sim_threshold=self.sim_threshold,
                    current_layer=i,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
            if output_retrieved_memory_idx:
                idx = (
                    3
                    if (use_cache & output_attentions)
                    else 2
                    if (use_cache or output_attentions)
                    else 1
                )
                all_idx = all_idx + (outputs[idx],)

        # Add last hidden state
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_idx,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=(all_self_attentions, all_idx),  # EM: Return idx of retrieved memories
        )


@add_start_docstrings(
    """
    The MPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MPT_START_DOCSTRING,
)
class ExtendedMptForCausalLM(MptPreTrainedModel):
    """Extended MPT for Causal LM."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ExtendedMptConfig, external_memories:list=None):
        super().__init__(config)
        self.transformer: ExtendedMptModel = ExtendedMptModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.use_external_mind = config.use_external_mind
        self.memory_type = config.attn_config.memory_type
        self.memory_ids = None
        self.memories = None
        self.memory_device = config.attn_config.memory_device
        self.remove_special_ids = config.attn_config.remove_special_ids
        self.tokenizer_all_special_ids = config.attn_config.tokenizer_all_special_ids

        # EM: Memory token ids
        if external_memories is not None:
            self.memory_ids = external_memories
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    # EM: Clear memory cache
    def clear_memory(self):
        """Clear memory cache."""
        self.memory_ids = None
        self.memories = None

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,  # NITS should it be layer_past?
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "use_external_mind": kwargs.get("use_external_mind"), # EM: Add config here
                "topk": kwargs.get("topk"),
                "output_retrieved_memory_idx": kwargs.get("output_retrieved_memory_idx"),
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(MPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_retrieved_memory_idx: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_external_mind: Optional[bool] = None,
        topk: int = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # EM: Generate key value cache once on first call
        if (
            self.memory_ids is not None and self.memories is None
        ): 
            self.memory_ids = torch.tensor([self.memory_ids], device=self.device) if type(self.memory_ids)==list else self.memory_ids
            self.memories = self.generate_cache(
                self.memory_ids, cache_type=self.memory_type,
            )
            # EM: Remove special tokens from memory cache
            if self.remove_special_ids:
                idx_to_remove = [
                    token_idx
                    for token_idx, token in enumerate(self.memory_ids[0])
                    if token in self.tokenizer_all_special_ids
                ]
                if self.memory_type == "manual":
                    mask = torch.ones(self.memories[0][0].size(), dtype=torch.bool)
                    mask[:, :, idx_to_remove, :] = False

                    new_size = (
                        self.memories[0][0].size(0),
                        self.memories[0][0].size(1),
                        -1,
                        self.memories[0][0].size(3),
                    )
                    self.memories = [
                        (ks[mask].view(new_size), vs[mask].view(new_size))
                        for ks, vs in self.memories
                    ]
                else:
                    kn_index, kv_index = self.memories
                    all_idx_to_remove = [
                        [
                            i
                            for i in range(0, kn_index.ntotal)
                            if (
                                i
                                % (
                                    kn_index.ntotal
                                    / (
                                        self.config.num_attention_heads
                                        * self.config.num_hidden_layers
                                    )
                                )
                            )
                            == j
                        ]
                        for j in idx_to_remove
                    ]
                    kn_index.remove_ids(
                        np.array(all_idx_to_remove).flatten().astype("int64")
                    )
                    kv_index.remove_ids(
                        np.array(all_idx_to_remove).flatten().astype("int64")
                    )

        use_external_mind = (
            use_external_mind
            if use_external_mind is not None
            else self.use_external_mind
        )
        topk = topk if topk is not None else None

        long_range_past_key_values = None
        faiss_indexes = None
        if hasattr(self, "memories") and isinstance(self.memories, list):
            long_range_past_key_values = self.memories
        elif hasattr(self, "memories"):
            faiss_indexes = self.memories

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_retrieved_memory_idx=output_retrieved_memory_idx,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            long_range_past_key_values=long_range_past_key_values,
            faiss_indexes=faiss_indexes,
            use_external_mind=use_external_mind,
            topk=topk,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length),
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self,
        past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        beam_idx: torch.LongTensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device)
            for layer_past in past
            for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in past
        )
        return reordered_past

    # EM: Add method to generate key-value cache
    def generate_cache(
        self,
        input_ids: torch.LongTensor,
        stride: int = 512,
        max_len: int = 3072,
        cache_type: str = "manual",
    ):
        """Generate cache for long range attention."""
        if cache_type not in ["manual", "faiss"]:
            raise NotImplementedError(f"Cache type {cache_type} not implemented.")

        prev_end_loc = 0
        long_range_past_key_values = None
        faiss_indexes = None
        for b_idx in range(
            0, input_ids.size(-1), stride
        ):  # generate kv-pairs using stride
            end_loc = min(b_idx + max_len, input_ids.size(-1))
            trg_len = end_loc - prev_end_loc
            subseq = input_ids[:, b_idx:end_loc].to(self.device)
            with torch.no_grad():
                outputs = self.transformer(
                    subseq, use_cache=True, use_external_mind=False
                )
            to_cache = [
                (kv[0][:, :, -trg_len:], kv[1][:, :, -trg_len:])
                for kv in outputs.past_key_values
            ]
            long_range_past_key_values, faiss_indexes = self.cache(
                to_cache,
                cache_type,
                long_range_past_key_values=long_range_past_key_values,
                faiss_indexes=faiss_indexes,
            )

            prev_end_loc = end_loc
            if end_loc == input_ids.size(-1):
                break
        if long_range_past_key_values is not None:
            return long_range_past_key_values
        else:
            return faiss_indexes
        
    # EM: Add method to cache key value pairs
    def cache(
        self,
        to_cache: list,
        cache_type: str = "manual",
        long_range_past_key_values: list = None,
        faiss_indexes: faiss.IndexFlatIP = None,
        max_length_cache=100000,
        verbose=False,
    ):
        """Cache long range attention."""
        if (long_range_past_key_values is not None) & (faiss_indexes is not None):
            raise NotImplementedError(
                "Using faiss and passing key value pairs manually are mutually exclusive right now."
            )
        
        # To avoid spinning up a new index for each layer, we add one-hot encodings to the keys so that queries match with the appropriate layer, head
        if cache_type == "faiss":  # add one-hot encoding to match layer, head indices
            one_hot_encodings = (
                F.one_hot(torch.arange(0, self.config.n_heads * self.config.n_layers))
                * 10
            )
            # New indices, one to store normalized keys with one-hot encodings, another to retrieve kv pairs without normalization
            if faiss_indexes is None:
                faiss_indexes = (
                    faiss.IndexFlatIP(
                        to_cache[0][0].size(-1) + one_hot_encodings.size(-1)
                    ),
                    faiss.IndexFlatIP(to_cache[0][0].size(-1) * 2),
                )
            kn_index, kv_index = faiss_indexes
            for l_idx, (k, v) in enumerate(to_cache):
                k_n = (k / vector_norm(k, ord=2, dim=-1, keepdim=True)).to("cpu") #Normalize keys for cosine sim
                
                # Indices are 2 dimensional, so flatten 
                # Add normalized keys with one-hot encodings
                k_n = torch.concat(
                    [
                        rearrange(k_n, "b h s d -> b (h s) d", h=self.config.n_heads),
                        one_hot_encodings[
                            self.config.n_heads
                            * l_idx : self.config.n_heads
                            * (l_idx + 1)
                        ]
                        .unsqueeze(0)
                        .repeat_interleave(repeats=k.size(-2), dim=-2),
                    ],
                    dim=-1,
                )
                kn_index.add(k_n.squeeze().numpy())

                # Add unnormalized keys and values
                k = rearrange(k, "b h s d -> b (h s) d", h=self.config.n_heads)
                v = rearrange(v, "b h s d -> b (h s) d", h=self.config.n_heads)
                kv_index.add(
                    torch.concat([k.squeeze(), v.squeeze()], dim=1).to("cpu").numpy()
                )
        else:
            # Simply use list to store key value pairs
            if long_range_past_key_values is None:
                long_range_past_key_values = [
                    (k.to(self.memory_device), v.to(self.memory_device))
                    for k, v in to_cache
                ]
            else:
                long_range_past_key_values = [
                    (
                        torch.concat(
                            [kv[0], to_cache[ind][0].to(self.memory_device)], dim=2
                        ),
                        torch.concat(
                            [kv[1], to_cache[ind][1].to(self.memory_device)], dim=2
                        ),
                    )
                    for ind, kv in enumerate(long_range_past_key_values)
                ]
        if (
            long_range_past_key_values is not None
        ):  # set a limit on manual memory length
            if long_range_past_key_values[0][0].size(-2) > max_length_cache:
                long_range_past_key_values = [
                    (
                        kv[0][:, :, -max_length_cache:],
                        kv[1][:, :, -max_length_cache:],
                    )
                    for kv in long_range_past_key_values
                ]
        if verbose:
            if cache_type == "faiss":
                print(f"{kn_index.ntotal} keys in faiss index")
            else:
                print(f"{long_range_past_key_values[0][0].size(-2)} cached kvs")

        return (
            long_range_past_key_values,
            (kn_index, kv_index) if cache_type == "faiss" else None,
        )
