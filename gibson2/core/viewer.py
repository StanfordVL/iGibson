import cv2
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from transforms3d.quaternions import quat2mat

class CustomizedViewer(object):
	def __init__(self, windows=[]):
		self.px = 0
		self.py = 0
		self._mouse_ix, self._mouse_iy = -1, -1
		self.down = False
		self.view_direction = np.array([1, 0, 0])
		self.windows = windows
		cv2.destroyAllWindows()
		# for window in self.windows:
		cv2.namedWindow('sensor', cv2.WINDOW_NORMAL)
		cv2.setMouseCallback('sensor', self.change_dir)

		self.renderer = None


	def change_dir(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self._mouse_ix, self._mouse_iy = x, y
			self.down = True
		if event == cv2.EVENT_MOUSEMOVE:
			if self.down:
				dx = (x - self._mouse_ix) / 100.0
				dy = (y - self._mouse_iy) / 100.0
				self._mouse_ix = x
				self._mouse_iy = y
				r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0, np.cos(dy)]])
				r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx), np.cos(-dx), 0], [0, 0, 1]])
				self.view_direction = r1.dot(r2).dot(self.view_direction)

			elif event == cv2.EVENT_LBUTTONUP:
				self.down = False


	def update(self, state, pose_camera=None):
		sensor_frames = []
		for window in self.windows:
			if window == 'rgb':
				sensor_frames.append(state[window])
			elif window == 'depth':
				frame = np.repeat(state[window], 3, axis=2)
				sensor_frames.append(frame)
			elif window == 'scan':
				raise NotImplemented
				# world_xyz = state[window]
				# transform_matrix = quat2mat([pose_camera[-1], pose_camera[3], pose_camera[4], pose_camera[5]])
				# camera_frame = frame.dot(transform_matrix)
				# print('camera frame shape" {}'.format(camera_frame.shape))
				# camera_matrix = np.array([[1, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 1.0]])
				# print('camera matrix: {}'.format(camera_matrix.shape))
				# rvec, _ = cv2.Rodrigues(transform_matrix)
				# print('rvec: {}'.format(rvec))
				# tvec = np.expand_dims(pose_camera[:3], axis=0)
				# print('tvec: {}'.format(tvec))
				# image_xy, _ = cv2.projectPoints(world_xyz, rvec, tvec, camera_matrix, distCoeffs=None)
				# image_xy = np.squeeze(np.round(image_xy), axis=1)
				# print('image xy: {}'.format(image_xy))
				# frame = np.zeros((128, 128, 3))
				
				# print('frame shape: {}'.format(frame.shape))
				# print('frame: {}'.format(frame))
				# sensor_frames.append(frame)
				# sensor_frames.append(frame)


			# frame = cv2.cvtColor(state[window], cv2.COLOR_RGB2BGR)
		if len(sensor_frames) > 0:
			sensor_frames = cv2.cvtColor(np.concatenate(sensor_frames, axis=1), cv2.COLOR_RGB2BGR)
			cv2.waitKey(1)
			cv2.putText(sensor_frames, "px {:1.1f} py {:1.1f}".format(self.px, self.py), (10, 20),\
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
			cv2.putText(sensor_frames, "[{:1.1f} {:1.1f} {:1.1f}]".format(*self.view_direction), (10, 40),\
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
			cv2.imshow('sensor', sensor_frames)


		# for window in self.windows:
			# cv2.imshow(window, frame)