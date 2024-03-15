import numpy as np
import torch
from skimage import transform as trans
import cv2

src = np.array(
	[[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
	 [41.5493, 92.3655], [70.7299, 92.2041]],
	dtype=np.float32)

def align(lmk):
	lmk5=to5lmk(lmk)
	tform = trans.SimilarityTransform()
	tform.estimate(lmk5, src)
	M=tform.params.copy()
	return M

def to5lmk(lmk):
	eye_right=lmk[36:42].mean(0)
	eye_left=lmk[42:48].mean(0)
	nose=lmk[30]
	mouth_right=lmk[48]
	mouth_left=lmk[54]
	lmk5=np.stack([eye_right,eye_left,nose,mouth_right,mouth_left],0)
	return lmk5

def warp_img(frame,M,size):
	return cv2.warpPerspective(frame, M, (size[1],size[0]),flags=cv2.INTER_AREA)


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