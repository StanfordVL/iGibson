import os
import numpy as np
import pickle
from PIL import Image
import networkx as nx 
import cv2
import matplotlib.pyplot as plt

class GlobalPlanner(object):
	def __init__(self, map_path=None,  
		trav_map_resolution=0.1, 
		trav_map_erosion=0.8,
		waypoint_resolution=0.5,
		alg='astar'):
		
		self.trav_map_resolution = trav_map_resolution  # each pixel=0.1m = 10cm.
		self.trav_map_default_resolution = 0.01
		self.waypoint_interval = int(waypoint_resolution / self.trav_map_resolution)
		self.trav_map_erosion = int(trav_map_erosion / trav_map_resolution)
		print('trav map erosion: {}'.format(self.trav_map_erosion))
		self.waypoint_resolution = waypoint_resolution
		self.map_path = map_path
		if not self.map_path:
			self.map_path = 'maps/basic'
		
		if alg == 'astar':
			self.alg = nx.astar_path
		else:
			raise NotImplemented

		self.floor_graphs = []
		self.floor_maps = []
		

	def load_traversability_map(self, trav_map_path):
		print('load traversability map')
		assert os.path.exists(trav_map_path), 'No traversability map'
		trav_map = np.array(Image.open(trav_map_path))
		# plt.imshow(self.trav_map)
		self.ylen, self.xlen = trav_map.shape
		self.ylen = int(self.ylen * self.trav_map_default_resolution / self.trav_map_resolution) # Resacle.
		self.xlen = int(self.xlen * self.trav_map_default_resolution / self.trav_map_resolution)
		trav_map = cv2.resize(trav_map, (self.xlen, self.ylen))
		trav_map = cv2.erode(trav_map, np.ones((self.trav_map_erosion, self.trav_map_erosion)))
		trav_map[trav_map < 255] = 0
		# plt.imshow(trav_map)
		# plt.show()
		return trav_map


	def build_connectivity_graph(self, graph_path, trav_map_path):
		'''
		Inputs: 
			graph_path: path to connectivity graph, relative to map_path.
			trav_map_path: path to traversability map, relative to map_path.
		'''
		graph_path = os.path.join(self.map_path, graph_path)
		trav_map_path = os.path.join(self.map_path, trav_map_path)
		trav_map = self.load_traversability_map(trav_map_path)

		if os.path.exists(graph_path):
			print('load connectivity graph')
			with open(graph_path, 'rb') as pfile:
				g = pickle.load(pfile)
		
		else:
			assert trav_map_path is not None, "trav_map_path missing"
			print('build connectivity graph')
			self.ylen, self.xlen = trav_map.shape
			g = nx.Graph()
			for i in range(self.ylen):
				for j in range(self.xlen):
					if trav_map[i,j] > 0:
						print('add ({}, {})'.format(i, j))
						g.add_node((j, i))
						neighbors = [(i-1, j-1), (i, j-1), (i+1, j-1), (i-1, j)]
						for n in neighbors:
							if 0 <= n[0] < self.ylen and 0 <= n[1] < self.xlen \
							and trav_map[n[0], n[1]] > 0:
								print('add edge to: {}'.format(n))
								g.add_edge((n[1], n[0]), (j, i), weight=self.l2_distance(n, (i, j)))

			# Only take the largets connected component.
			# cc = sort(nx.connected_components(g), key=len)
			# largest_cc = max(nx.connected_components(g), key=len)
			# g = g.subgraph(largest_cc).copy()

			with open(graph_path, 'wb') as pfile:
				pickle.dump(g, pfile, protocol=pickle.HIGHEST_PROTOCOL)
			print('connectivity graph saved at: {}'.format(graph_path))
		self.floor_graphs.append(g)
		# Update trav_map according to largest cc.


	def get_shortest_path(self, source, target, component=0):
		'''
		inputs:
			source, target: (x, y) coordinates in world frame.
			component: index of the connected component being queried, if several are built. 
		'''
		g = self.floor_graphs[component]
		source_map = tuple(self.world_to_map(source))
		target_map = tuple(self.world_to_map(target))
		nodes = np.array(g.nodes)
		# if not g.has_node(source_map):
		source_closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - source_map, axis=1))])
			# g.add_edge(closest_node, source_map, weight=self.l2_distance(closest_node, source_map))
		# if not g.has_node(target_map):
		target_closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
			# g.add_edge(closest_node, target_map, weight=self.l2_distance(closest_node, target_map))
		
		# print('ALGO: {}'.format(self.alg))
		# print('source: {}'.format(source_map))
		# path_map = np.array(self.alg(g, source_map, target_map, heuristic=self.l2_distance))
		path_map = np.array(self.alg(g, source_closest_node, target_closest_node, heuristic=self.l2_distance))
		path_map = np.concatenate([[source_map], path_map, [target_map]])
		# plt.imshow(self.trav_map, cmap='summer')
		# print('source map: {}'.format(source_map))
		# print('path map: {}'.format(path_map))
		# plt.plot(*source_map, 'bo')
		# plt.plot(*target_map, 'go')
		# plt.scatter(path_map[:, 0], path_map[:, 1], s=1, c='r')
		# plt.show()
		path_world = self.map_to_world(path_map)
		geodesic_distance = np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1)) 
		path_world = path_world[::self.waypoint_interval]
		# print('geodesic distance: {}'.format(geodesic_distance))
		# print('distance between two consecutive waypoints: {}'.format(self.l2_distance(path_world[0], path_world[1])))

		# TODO: not entire_path mode.
		return path_world, geodesic_distance

	def map_to_world(self, xy):
		axis = 0 if len(xy.shape) == 0 else 1
		return np.flip((xy - np.array([self.xlen, self.ylen]) / 2.0) * self.trav_map_resolution, axis=axis)


	def world_to_map(self, xy):
		# TODO: depends on how the traversability map is defined; flipping might be needed.
		return np.flip((xy / self.trav_map_resolution + np.array([self.ylen, self.xlen]) / 2.0)).astype(np.int)

	def add_edge_to_closest_node(self, g, node):
		nodes = np.array(g.nodes)
		closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - node, axis=1))])
		g.add_edge(closest_node, node, weight=self.l2_distance(closest_node, node))


	def l2_distance(self, source, target):
		return ((source[0] - target[0]) ** 2 + (source[1] - target[1]) ** 2) ** 0.5