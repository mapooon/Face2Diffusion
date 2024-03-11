import os
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import numpy as np
import cv2
from torch import nn
import argparse
import torchvision
from skimage import transform as trans
import face_alignment
from PIL import Image
from torch.nn import functional as F
from src import modules


def get_clip_hidden_states(input_ids,text_encoder):
	output_attentions = text_encoder.text_model.config.output_attentions
	output_hidden_states = (
		text_encoder.text_model.config.output_hidden_states
	)
	return_dict = text_encoder.text_model.config.use_return_dict

	if input_ids is None:
		raise ValueError("You have to specify input_ids")

	input_shape = input_ids.size()
	input_ids = input_ids.view(-1, input_shape[-1])

	hidden_states = text_encoder.text_model.embeddings(input_ids=input_ids, position_ids=None)
	return hidden_states



def main(args):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
	pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16,safety_checker=None).to("cuda")
	pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

	img2text = modules.IMG2TEXTwithEXP(384*4,384*4,768)
	img2text.load_state_dict(torch.load(args.weight,map_location='cpu'))
	img2text=img2text.to(device)
	img2text.eval()

	identifier='f'

	ids = pipe.tokenizer(
					args.prompt,
					padding="do_not_pad",
					truncation=True,
					max_length=pipe.tokenizer.model_max_length,
				).input_ids
	placeholder_token_id=pipe.tokenizer(
					identifier,
					padding="do_not_pad",
					truncation=True,
					max_length=pipe.tokenizer.model_max_length,
				).input_ids[1]
	assert placeholder_token_id in ids,'identifier does not exist in prompt'
	pos_id = ids.index(placeholder_token_id)
	

	input_ids = pipe.tokenizer.pad(
			{"input_ids": [ids]},
			padding="max_length",
			max_length=pipe.tokenizer.model_max_length,
			return_tensors="pt",
		).input_ids


	with torch.no_grad():
		idvec = torch.tensor(np.load(args.input)).unsqueeze(0).to(device)
		tokenized_identity_first, tokenized_identity_last = img2text(idvec,exp=None)
		hidden_states = get_clip_hidden_states(input_ids.to(device),pipe.text_encoder).to(dtype=torch.float32)
		hidden_states[[0], [pos_id]]=tokenized_identity_first.to(dtype=torch.float32)
		hidden_states[[0], [pos_id+1]]=tokenized_identity_last.to(dtype=torch.float32)
		pos_eot = input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1)
	
	#text encoding
	with torch.autocast("cuda"):
		with torch.no_grad():
			encoder_hidden_states = pipe.text_encoder(hidden_states=hidden_states, pos_eot=pos_eot)[0]

	#diffusion process
	generator = torch.Generator(device).manual_seed(0)
	image = pipe(prompt_embeds=encoder_hidden_states, num_inference_steps=30, guidance_scale=7,generator=generator,num_images_per_prompt=args.n_samples).images#[0]
	image = np.concatenate([np.array(image[i]) for i in range(len(image))],1)
	image = Image.fromarray(image.astype(np.uint8))

	#save output
	image.save(args.output)

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-p',dest='prompt',required=True)
	parser.add_argument('-i',dest='input',required=True)
	parser.add_argument('-w',dest='weight',required=True)
	parser.add_argument('-o',dest='output',required=True)
	parser.add_argument('-n',dest='n_samples',default=8,type=int)
	args=parser.parse_args()
	main(args)