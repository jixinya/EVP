import cv2
#import pickle
import time
import numpy as np
import copy
import imageio
from matplotlib import pyplot as plt
from tqdm import tqdm
import open3d as o3d



class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
      s = value
    else:
      s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s


class OneEuroFilter:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    cutoff = self.mincutoff + self.beta * np.abs(edx)
    return self.x_filter.process(x, self.compute_alpha(cutoff))


def render_sequence_3d(verts, faces, width, height, video_path, fps=30,
                       visible=False, need_norm=True):
  if type(verts) == list:
    verts = np.stack(verts, 0)

  scale = np.max(np.max(verts, axis=(0, 1)) - np.min(verts, axis=(0, 1)))
  mean = np.mean(verts)
  verts = (verts - mean) / scale

  cam_offset = 1.2

  writer = VideoWriter(video_path, width, height, fps)

  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.vertices = o3d.utility.Vector3dVector(verts[0])

  vis = o3d.visualization.Visualizer()
  vis.create_window(width=width, height=height, visible=visible)
  vis.add_geometry(mesh)
  view_control = vis.get_view_control()
  cam_params = view_control.convert_to_pinhole_camera_parameters()
  cam_params.extrinsic = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, cam_offset],
    [0, 0, 0, 1],
  ])
  view_control.convert_from_pinhole_camera_parameters(cam_params)

  for v in tqdm(verts, ascii=True):
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    frame = (np.asarray(vis.capture_screen_float_buffer()) * 255).astype(np.uint8)
    writer.write_frame(frame)

  writer.close()