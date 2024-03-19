import os
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import numpy as np
import cv2
from torch import nn
import argparse
import torchvision
import face_alignment
from PIL import Image
from torch.nn import functional as F
from src import modules
from src import utils
from src.msid import msid_base_patch8_112
from transformers.models.clip.modeling_clip import CLIPTextTransformer,CLIPTextModel
from src import mod



def main(args):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
	pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16,safety_checker=None).to("cuda")
	pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

	#build f2d pipeline
	pipe.text_encoder.text_model.forward = mod.forward_texttransformer.__get__(pipe.text_encoder.text_model, CLIPTextTransformer)
	pipe.text_encoder.forward = mod.forward_textmodel.__get__(pipe.text_encoder, CLIPTextModel)

	img2text = modules.IMG2TEXTwithEXP(384*4,384*4,768)
	img2text.load_state_dict(torch.load(args.w_map,map_location='cpu'))
	img2text=img2text.to(device)
	img2text.eval()

	msid = msid_base_patch8_112(ext_depthes=[2,5,8,11])
	msid.load_state_dict(torch.load(args.w_msid))
	msid=msid.to(device)
	msid.eval()

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


	#identity encoding
	detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cuda' if torch.cuda.is_available() else 'cpu')
	lmk=np.array(detector.get_landmarks(args.input))[0]
	img = np.array(Image.open(args.input).convert('RGB'))
	with torch.no_grad():
		M=utils.align(lmk)
		img=utils.warp_img(img,M,(112,112))/255
		img=torch.tensor(img).permute(2,0,1).unsqueeze(0)
		img=(img-0.5)/0.5
		idvec = msid.extract_mlfeat(img.to(device).float(),[2,5,8,11])
		tokenized_identity_first, tokenized_identity_last = img2text(idvec,exp=None)
		hidden_states = utils.get_clip_hidden_states(input_ids.to(device),pipe.text_encoder).to(dtype=torch.float32)
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
	parser.add_argument('-i',dest='input',required=True,help='path for the input facial image')
	parser.add_argument('--w_map',required=True,help='weight path for the mapping network')
	parser.add_argument('--w_msid',required=True,help='weight path for the msid encoder')
	parser.add_argument('-o',dest='output',required=True)
	parser.add_argument('-n',dest='n_samples',default=8,type=int)
	args=parser.parse_args()
	main(args)