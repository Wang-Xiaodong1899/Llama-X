"""
LLaMA model from transformers, following stanford's Alpaca
"""

MODEL_PATH = "/workspace/GODIVA/model/emu/llama_config"

from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig

IGNORE_INDEX = -100
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMG_START_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"
DEFAULT_IMG_TOKEN = "<image>"
USER_TOKEN = '[USER]'
ASSISTANT_TOKEN = '[ASSISTANT]'


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        resize_output: bool = True,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg

        if resize_output:
            output_embeddings = model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class RegressCausalLMOutputWithPast(CausalLMOutputWithPast):
    llm_loss: Optional[torch.FloatTensor] = None
    regression_loss: Optional[torch.FloatTensor] = None

from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, apply_rotary_pos_emb, repeat_kv
import torch.nn.functional as F
import math

class LlamaAttention_adapter(LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.gate = torch.nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        adapter=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.k_proj(adapter).view(1, adapter_len, self.num_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.v_proj(adapter).view(1, adapter_len, self.num_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_k = adapter_k.transpose(1, 2)
            adapter_v = adapter_v.transpose(1, 2)
            adapter_scores = torch.matmul(query_states, adapter_k.transpose(2,3)) / math.sqrt(self.num_heads)
            adapter_scores = self.gate * F.softmax(adapter_scores.float(), dim=-1).type_as(query_states)
            attn_output = attn_output + torch.matmul(adapter_scores, adapter_v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaDecoderLayer_adapter(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.self_attn = LlamaAttention_adapter(config=config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        adapter=None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            adapter=adapter
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    

class LlamModel_adapter(transformers.LlamaModel):
    """
    add adapter layer to Llama model
    """
    def __init__(self, config):
        super().__init__(config)
        self.adapter_len = 30
        self.adapter_layer = 30
        self.adapter_query = nn.Embedding(self.adapter_len * self.adapter_layer, config.hidden_size)
        self.n_layers = config.num_hidden_layers
        
        self.layers = nn.ModuleList([LlamaDecoderLayer_adapter(config) for _ in range(config.num_hidden_layers)])
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        adapter_prompt = self.adapter_query.weight.reshape(self.adapter_layer, self.adapter_len,
                                                   self.hidden_size).unsqueeze(1) # 8, 1, 10, 4049

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            adapter_index = idx - (self.n_layers - self.adapter_layer)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward
                if adapter_index < 0:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None,
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None,
                        adapter_prompt[adapter_index],
                    )
            else:
                if adapter_index < 0:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        adapter=adapter_prompt[adapter_index]
                    )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForReg(transformers.LlamaForCausalLM):
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                regress_mask: torch.Tensor = None,
                img_length: int = None,
                args=None,
                regress_labels=None
                ):
        """
        :param self:
        :param inputs_embeds: shape [B, 1 + n_image + n_token, C]
        :param img_length: length of image tokens, not include the special `[IMG]` token
        :return:
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        print(labels)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return RegressCausalLMOutputWithPast(
            llm_loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LLaMAForClsAndRegression(nn.Module):
    def __init__(self, args, model_name_or_path=MODEL_PATH):
        super(LLaMAForClsAndRegression, self).__init__()
        self.args = args
        print(f'tokenizer: {model_name_or_path}')
        # self.lm = LlamaForReg.from_pretrained(model_name_or_path)
        self.lm = LlamaForReg(config=LlamaConfig.from_pretrained(model_name_or_path))

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=2048,
            padding_side="right",
            truncation=True,
            use_fast=False,
        )

        if args.instruct:  # for instruction tuning, [USER] and [ASSISTANT] tokens are added
            special_token_list = [DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_END_TOKEN, DEFAULT_IMG_TOKEN, USER_TOKEN,
                                  ASSISTANT_TOKEN]
        else:
            special_token_list = [DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_END_TOKEN, DEFAULT_IMG_TOKEN]

        special_tokens_dict = dict(
            pad_token=DEFAULT_PAD_TOKEN,
            bos_token=DEFAULT_BOS_TOKEN,
            eos_token=DEFAULT_EOS_TOKEN,
            unk_token=DEFAULT_UNK_TOKEN,
            additional_special_tokens=special_token_list
        )

        if self.tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=self.tokenizer,
                model=self.lm,
            )
        
        self.lm.model.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        print(f"The Special Tokens: {self.tokenizer.special_tokens_map}")
        print(f"Vocab Size: {len(self.tokenizer)}")

        # student head
        self.lm.stu_regress_head = nn.Linear(self.lm.config.hidden_size, self.lm.config.hidden_size, bias=False)

        self.config = self.lm.config
        self.lm.config.d_model = self.lm.config.hidden_size
        self.lm.half()

        self.prompt = None

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(['<image>'])[0]
        print(f"image_token_id: {self.image_token_id}")

        self.img_token_id = self.tokenizer.convert_tokens_to_ids(['[IMG]'])[0]
        print(f"[IMG] token id: {self.img_token_id}")

        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(['[/IMG]'])[0]
        print(f"[/IMG] token id: {self.img_end_token_id}")

    def get_num_layers(self):
        return len(self.lm.model.layers)

    def forward(self, image_embeds, text_input, text_mask, text_output=None, output_mask=None):
        """
        Process:
        1. image_embeds & text_tokens as input
        2. prepend [IMG] token to img features or replace <ImagePatches> in <img><ImagePatches></img> with img features
        3. concat image and text features
        4. prepend <BOS> to sequence and append <EOS> to end of sequence
        4. feed into forward and return two losses

        :param image_embeds: [B, n_causal, C], after projected into Language shape
        :param text_input: [B, seq_len]
        :param text_mask: [B, seq_len]
        :return:
        """
        B, n_causal, _ = image_embeds.shape

        # mask [PAD]
        targets = text_input.masked_fill(
            text_input == self.tokenizer.pad_token_id, -100
        )
        # mask <image>
        targets = targets.masked_fill(
            targets == self.image_token_id, -100
        )
        # mask [IMG]
        targets = targets.masked_fill(
            targets == self.img_token_id, -100
        )
        # mask [/IMG]
        targets = targets.masked_fill(
            targets == self.img_end_token_id, -100
        )

        if self.args.instruct:
            text_embeds = self.lm.model.model.embed_tokens(text_input)  # [B, seq_len, C]
        else:
            text_embeds = self.lm.model.embed_tokens(text_input)  # [B, seq_len, C]

        all_image_indices = (text_input == self.image_token_id).to(image_embeds.device)

        assert (text_input[all_image_indices].shape[0] == image_embeds.shape[0] * image_embeds.shape[1]), \
            f"{text_input[text_input == self.image_token_id].shape[0]} != {image_embeds.shape[0]}*{image_embeds.shape[1]}"
        assert (image_embeds.shape[-1] == text_embeds.shape[-1]), f"{image_embeds.shape[-1]} != {text_embeds.shape[-1]}"

        image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])

        text_embeds[all_image_indices] = image_embeds

        regress_label_mask = ((text_input == self.image_token_id) + (text_input == self.img_end_token_id)).to(
            image_embeds.device)

        regress_labels = text_embeds[regress_label_mask]
        regress_mask = ((text_input == self.image_token_id) + (text_input == self.img_token_id)).to(image_embeds.device)

        outputs = self.lm(
            inputs_embeds=text_embeds,
            attention_mask=text_mask,
            return_dict=True,
            labels=targets,
            regress_mask=regress_mask,
            img_length=n_causal,
            args=self.args,
            regress_labels=regress_labels.detach()
            # regress_labels=text_embeds
        )

        return outputs

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.lm.gradient_checkpointing_enable()
