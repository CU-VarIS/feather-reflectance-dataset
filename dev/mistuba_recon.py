
# %% [markdown]
# # CU-Varis Feather - material fitting

# %%

import sys

sys.path.append("..")

import mitsuba as mi    
import drjit as dr

from varis_feather.Utilities.show_image import show, image_montage_same_shape
from varis_feather.Utilities.ImageIO import readImage, writeImage, RGL_tonemap_uint8, RGL_tonemap
from varis_feather.Previs.MitsubaInit import mitsuba_set_mode
from varis_feather.Space.capture import CaptureFrame
from varis_feather.Space.olat import OLATCapture

from varis_feather import load_standard_capture


# %%

retro, olat = load_standard_capture("FeatherHyacinthMacaw")

# %%

import numpy as np
from matplotlib import pyplot
from tqdm import tqdm

class MitsubaSceneForCapture:

	param_light = "emitter.intensity.value"
	param_ggx_alpha = "shape.bsdf.nested_bsdf.alpha.data"
	param_reflectance = "shape.bsdf.nested_bsdf.specular_reflectance.data"

	def __init__(self, capture: OLATCapture):
		self.capture = capture

		h, w, _ = self.capture.stage_poses[(0, 0)].anchor_image.shape
		self.shape = (h, w)

		self.normals_textures = {
			wiid: (pose.normals_smooth[:, ::-1, :] + 1.) * 0.5
			for wiid, pose in capture.stage_poses.items()
		}

		self.scene = self.build_scene()

	@classmethod
	def params_to_optimize(cls):
		return [
			cls.param_light,
			cls.param_ggx_alpha,
			cls.param_reflectance,
		]

	def build_scene_dict(self) -> dict:
		h, w = self.shape
		
		scene_dict = {
			"type": "scene",
		    "integrator": {"type": "path"},
			"sensor": {
				"type": "orthographic",
				"to_world": mi.ScalarTransform4f().look_at(
					origin=[0, 0, 1],
					target=[0, 0, 0],
					up=[0, -1, 0]
				),
				"film": {
					"type": "hdrfilm",
					"width": w,
					"height": h,
					"rfilter": {"type": "box"},
					"pixel_format": "rgba",
				},
			},
			"emitter": {
				"type": "point",
				"position": [0.5, 0.5, 3.5],
				"intensity": {
					"type": "rgb",
					"value": [1e4]*3,
				}
			},
			"shape": {
				"type": "rectangle",
				# TODO aspect ratio
		        "to_world": mi.ScalarTransform4f()
					.translate([0, 0, 0])
					# .rotate([0, 0, 1], 180)
					.scale([1., h/w, 1.]),
					#.rotate([1, 0, 0], 10),
				"bsdf": {
					"type": "normalmap",
					"normalmap": {
						"type": "bitmap",
						"data": self.normals_textures[(0, 0)],
					},
					"bsdf": {
						"type": "roughdielectric",
						"distribution": "ggx",
						"alpha": {
							"type": "bitmap",
							"data": np.full((h, w, 1), fill_value=0.25), 
						},
						"specular_reflectance": {
							"type": "bitmap",
							"data": np.full((h, w, 3), fill_value=0.25),
						},
					},
				},          
			},
		}
	
		return scene_dict
	
	def build_scene(self) -> mi.Scene:
		self.scene_dict = self.build_scene_dict()
		self.scene = mi.load_dict(self.scene_dict)

		# Make differentiable
		self.params = mi.traverse(self.scene)


		print(self.params)
		

		return self.scene
	
	def setup_for_frame(self, frame: CaptureFrame):

		# self.params["sensor.to_world"] = mi.ScalarTransform4f().look_at(
		# 	# origin=frame.sample_wi.tolist(),
		# 	origin=[0, 0, 1],
		# 	target=[0, 0, 0],
		# 	up=[0, 1, 0]
		# )
		# print(self.params["sensor.to_world"])


		# Set the light direction

		self.params["emitter.position"] = (frame.sample_wo * 10).tolist()

		self.params.update()
	
	def render(self, frame: CaptureFrame):
		self.setup_for_frame(frame)
		return mi.render(self.scene, spp=2)

	def render_compare_to_frame(self, frame: CaptureFrame, b_show=True, save=None):
		res = np.array(self.render(frame))
		res_rgb = res[:, :, :3]

		img_gt = olat.read_measurement_image(frame)

		print("Render", res.shape, res.dtype, np.min(res_rgb), np.mean(res_rgb), np.max(res_rgb))

		print("GT", img_gt.shape, img_gt.dtype, np.min(img_gt), np.mean(img_gt), np.max(img_gt))

		res_rgb_tn = RGL_tonemap_uint8(res_rgb)
		res_rgb_tn[res[:, :, 3] == 0] = (0, 255, 0)

		img_compare = image_montage_same_shape([
			res_rgb_tn,
			RGL_tonemap_uint8(img_gt)
		], num_cols=2, border=8)

		if b_show:
			show(img_compare)

		if save:
			writeImage(img_compare, save)

		# show([res_rgb_tn, RGL_tonemap_uint8(olat.read_measurement_image(frame))])

		return img_compare
	
	# Objective function: MSE to reference
	@staticmethod
	def _mse(a, b):
		return dr.mean(dr.square(a - b))

	def optimize(self, frames, num_epochs=1, lr=0.03, batch_size=8):
		opt = mi.ad.Adam(lr=lr)

		for key in self.params_to_optimize():
			dr.enable_grad(self.params[key])
			opt[key] = self.params[key]

		self.params.update(opt)

		errors = []
		frames_epoch = list(frames)
		frames_all = []

		for epoch in range(num_epochs):
			# iterate over shuffled frames
			np.random.shuffle(frames_epoch)
			frames_all += frames_epoch



		# divide into sub-lists of batch_size

		batches = [
			frames_all[i:i + batch_size]
			for i in range(0, len(frames_all), batch_size)
		]

		pb = tqdm(enumerate(batches), total=len(batches), desc="Optimizing")

		for it, frame_batch in pb:
			loss = 0

			for frame in frame_batch:
				self.setup_for_frame(frame)

				# Reference image
				im_ref = self.capture.read_measurement_image(frame, cache=True)
				im_ref = dr.detach(mi.TensorXf(im_ref))

				# Perform a (noisy) differentiable rendering of the scene
				image = mi.render(self.scene, self.params, spp=2)

				# Evaluate the objective function from the current rendered image
				loss = self._mse(image[:, :, :3], im_ref) + loss
			# print(loss)

			# print("opt", opt)

			# Backpropagate through the rendering process
			dr.backward(loss)

			# Optimizer: take a gradient descent step
			opt.step()

			# Post-process the optimized parameters to ensure legal color values.
			try:
				for key in [self.param_reflectance, self.param_ggx_alpha]:
					opt[key] = dr.clip(opt[key], 0.0, 1.0)
			except KeyError as e:
				raise ValueError(f"No key {e} in {opt}") from e

			self.params.update(opt)


			# Update the scene state to the new optimized values

			# Track the difference between the current color and the true value
			# err_ref = dr.sum(dr.square(param_ref - params[key]))
			# print(f"Iteration {it:02d}: error = {loss}")
			err = float(np.array(dr.slice(loss)))
			pb.set_postfix_str(f"Error {err:.6f}")

			errors.append(err)

		errors = np.array(errors)
		
		fig, ax = pyplot.subplots(1, 1)
		ax.plot(errors)
		ax.set_xlabel('Step')
		ax.set_ylabel('MSE(rgb)')
		# ax.title('Parameter error plot');

		return errors

		print('\nOptimization complete.')


	def reflectance_image(self) -> np.ndarray:
		return np.array(self.params[self.param_reflectance])

	def reflectance_image_byte(self) -> np.ndarray:
		return (self.reflectance_image() * 255).astype(np.uint8)
	
	def ggx_alpha(self) -> np.ndarray:
		return np.array(self.params[self.param_ggx_alpha]).squeeze()
	

miscene = MitsubaSceneForCapture(olat)

# %%

miscene.render_compare_to_frame(olat.frames[20], save="mi_rec_20_dramatic2.jpg")
miscene.render_compare_to_frame(olat.frames[30], save="mi_rec_30.jpg")
miscene.render_compare_to_frame(olat.frames[40])
miscene.render_compare_to_frame(olat.frames[50])
miscene.render_compare_to_frame(olat.frames[60])

None

# %%

miscene.optimize(olat.stage_poses[(0, 0)].frames, num_epochs=16)

# %%

miscene.params[miscene.param_light]
# %%

show([miscene.reflectance_image_byte(), miscene.ggx_alpha()])

m kk# %%

miscene.render_compare_to_frame(olat.frames[20], save="mi_rec_20_opt.jpg")
miscene.render_compare_to_frame(olat.frames[30], save="mi_rec_30_opt.jpg")
None
# %%
