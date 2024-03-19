import torch

from transformers.models.clip.modeling_clip import CLIPTextTransformer,_make_causal_mask,_expand_mask
from typing import Any, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPooling


def forward_texttransformer(
	self,
	input_ids: Optional[torch.Tensor] = None,
	hidden_states: Optional[torch.Tensor] = None,
	pos_eot: Optional[torch.Tensor] = None,
	attention_mask: Optional[torch.Tensor] = None,
	position_ids: Optional[torch.Tensor] = None,
	output_attentions: Optional[bool] = None,
	output_hidden_states: Optional[bool] = None,
	return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
	r"""
	Returns:

	"""
	if hidden_states is None:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if input_ids is None:
			raise ValueError("You have to specify input_ids")

		input_shape = input_ids.size()
		input_ids = input_ids.view(-1, input_shape[-1])

		hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
	else:
		input_shape = hidden_states.size()[:2]
	# CLIP's text model uses causal mask, prepare it here.
	# https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
	causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
	# expand attention_mask
	if attention_mask is not None:
		# [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
		attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

	encoder_outputs = self.encoder(
		inputs_embeds=hidden_states,
		attention_mask=attention_mask,
		causal_attention_mask=causal_attention_mask,
		output_attentions=output_attentions,
		output_hidden_states=output_hidden_states,
		return_dict=return_dict,
	)

	last_hidden_state = encoder_outputs[0]
	last_hidden_state = self.final_layer_norm(last_hidden_state)

	# text_embeds.shape = [batch_size, sequence_length, transformer.width]
	# take features from the eot embedding (eot_token is the highest number in each sequence)
	# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
	if hidden_states is None:
		pos_eot = input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1)
	pooled_output = last_hidden_state[
		torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
		pos_eot,
	]

	if not return_dict:
		return (last_hidden_state, pooled_output) + encoder_outputs[1:]

	return BaseModelOutputWithPooling(
		last_hidden_state=last_hidden_state,
		pooler_output=pooled_output,
		hidden_states=encoder_outputs.hidden_states,
		attentions=encoder_outputs.attentions,
	)

def forward_textmodel(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        pos_eot: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            hidden_states = hidden_states,
            pos_eot = pos_eot,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )