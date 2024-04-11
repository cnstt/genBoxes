import torch
import logging
import numpy as np
import trimesh
import plotly.graph_objects as go
from open3d.geometry import TriangleMesh
from open3d.utility import Vector3dVector, Vector3iVector
from torch_geometric.data import Data, InMemoryDataset
from .utils import box_surface, check_collision, sphere_surface, cylinder_surface
from operator import itemgetter

logger = logging.getLogger(__name__)

class BoxesDataset(InMemoryDataset):
    def __init__(self, cfg, mode):
        super().__init__()
        
        # mode is 'train', 'test' or 'val'
        self.mode = mode
        self.visu = cfg.settings.visu
        num_boxes = cfg.settings.num_boxes
        
        # Adapt max coordinates: size of the box needs to considered after offset
        xmax = cfg.settings.xmax - cfg.settings.wmax
        ymax = cfg.settings.ymax - cfg.settings.lmax
        
        # Configure sampling
        sampling = cfg.settings.sampling # proportional according to surface
        if sampling == "uniform_proportional":
            total_points = cfg.settings.total_points # used if proportional sampling
        elif sampling == "uniform":
            samples_per_box = cfg.settings.samples_per_box # used if non-proportional sampling
        elif sampling == "velodyne":
            box_meshes = []
        else:
            raise ValueError("Sampling parameter not correct.")

        # Initialise boxes
        x = np.random.uniform(low=cfg.settings.xmin, high=xmax, size=num_boxes)
        y = np.random.uniform(low=cfg.settings.ymin, high=ymax, size=num_boxes)
        z = np.random.uniform(low=cfg.settings.zmin, high=cfg.settings.zmax, size=num_boxes)
        length = np.random.uniform(low=cfg.settings.lmin, high=cfg.settings.lmax, size=num_boxes)
        width = np.random.uniform(low=cfg.settings.wmin, high=cfg.settings.wmax, size=num_boxes)
        height = np.random.uniform(low=cfg.settings.hmin, high=cfg.settings.hmax, size=num_boxes)
        total_surface = sum(2*(length*width+length*height+width*height))
        
        # Choose the objects contained in the scene
        # Objects available: standard box (0), car (1), sphere (2), cylinder (3)
        if cfg.settings.add_obj:
            if cfg.settings.add_obj_type == 'complex':
                # 0 standard box obj, 2 sphere, 3 cylinder
                types_ids = [0,2,3]
                obj_type = np.random.choice(types_ids, num_boxes)
            elif cfg.settings.add_obj_type == 'complex_car':
                types_ids = [0,1,2,3]
                # add id 1 (for cars)
                obj_type = np.random.choice(types_ids, num_boxes)
            else:
                # 0 if standard obj, 1 if added special object type (car)
                obj_type = np.random.randint(2,size=num_boxes)
        else:
            obj_type = np.zeros(10,dtype=int)
        
        if self.visu:
            self._data = []
        boxes = []
        count_cols=0
        pos_list = []
        
        # prepare the clearing of the central area
        # objects shouldn't block the lidar
        if sampling == "velodyne":
            # size hardcoded for now
            minLidarDist = 6
            lidar_clearing = {
                'x': (cfg.settings.xmax - cfg.settings.xmin)/2 + cfg.settings.xmin - minLidarDist,
                'y': (cfg.settings.ymax - cfg.settings.ymin)/2 + cfg.settings.ymin - minLidarDist,
                'z': 0,
                'length': minLidarDist*2,
                'width': minLidarDist*2,
                'height': minLidarDist*2
            }

        for i in range(num_boxes):
            # Check collision
            new_box = {
                'x': x[i],
                'y': y[i],
                'z': z[i],
                'length': length[i] if obj_type[i] in [0,2] else cfg.settings.add_l2,
                'width': width[i] if obj_type[i] in [0,2] else cfg.settings.add_w2,
                'height': height[i] if obj_type[i] in [0,2] else cfg.settings.add_h1+cfg.settings.add_h2
            }

            while check_collision(boxes, new_box, cfg.settings.min_dist if hasattr(cfg.settings, "min_dist") else None) \
                or (sampling == "velodyne" and check_collision([lidar_clearing], new_box, cfg.settings.min_dist if hasattr(cfg.settings, "min_dist") else None)):
                count_cols+=1
                x[i] = np.random.uniform(low=cfg.settings.xmin, high=xmax)
                y[i] = np.random.uniform(low=cfg.settings.ymin, high=ymax)
                z[i] = np.random.uniform(low=cfg.settings.zmin, high=cfg.settings.zmax)

                new_box = {
                    'x': x[i],
                    'y': y[i],
                    'z': z[i],
                    'length': length[i] if obj_type[i] in [0,2] else cfg.settings.add_l2,
                    'width': width[i] if obj_type[i] in [0,2] else cfg.settings.add_w2,
                    'height': height[i] if obj_type[i] in [0,2] else cfg.settings.add_h1+cfg.settings.add_h2
            }
            # Add box to boxes list
            boxes.append(new_box)
            if obj_type[i]==0:
                # Create an Open3D box mesh
                box_mesh = TriangleMesh.create_box(width=width[i], height=height[i], depth=length[i])
                box_mesh.translate([x[i], y[i], z[i]])
                surface = box_surface(length[i], width[i], height[i])
            # Sphere
            elif obj_type[i]==2:
                box_mesh = TriangleMesh.create_sphere(radius=width[i]/2)
                box_mesh.translate([x[i]+width[i]/2, y[i]+width[i]/2, z[i]+width[i]/2])
                surface = sphere_surface(width[i]/2)
            # Cylinder
            elif obj_type[i]==3:
                radius = cfg.settings.add_w2/4 # Even smaller radius
                box_mesh = TriangleMesh.create_cylinder(radius=radius, height=cfg.settings.add_h2**2)
                box_mesh.translate([x[i]+radius, y[i]+radius, z[i]+cfg.settings.add_h2**2/2])
                surface = cylinder_surface(radius=radius, height=cfg.settings.add_h2)
            # Car
            else:
                ori = np.random.choice([0, 1])
                box_mesh1 = TriangleMesh.create_box(width=cfg.settings.add_w1,
                                                    height=cfg.settings.add_h1,
                                                    depth=cfg.settings.add_l1)
                box_mesh1.translate([x[i], y[i]+ori*cfg.settings.add_h2, z[i]])
                box_mesh2 = TriangleMesh.create_box(width=cfg.settings.add_w2,
                                                    height=cfg.settings.add_h2,
                                                    depth=cfg.settings.add_l2)
                box_mesh2.translate([x[i], y[i]+(1-ori)*cfg.settings.add_h1, z[i]])
                surface = box_surface(cfg.settings.add_l1, cfg.settings.add_w1, cfg.settings.add_h1)
                surface += box_surface(cfg.settings.add_l2, cfg.settings.add_w2, cfg.settings.add_h2)

            # Perform sampling or store mesh
            if sampling == "uniform_proportional":
                if obj_type[i] == 1:
                    num_points = int(surface * total_points / total_surface)
                    # Convert the meshes to numpy arrays
                    vertices1 = np.asarray(box_mesh1.vertices)
                    vertices2 = np.asarray(box_mesh2.vertices)
                    faces1 = np.asarray(box_mesh1.triangles)
                    faces2 = np.asarray(box_mesh2.triangles)
                    # Shift the indices of the faces of the second mesh
                    faces2 += len(vertices1)
                    # Combine the vertices and faces
                    vertices = np.concatenate((vertices1, vertices2), axis=0)
                    faces = np.concatenate((faces1, faces2), axis=0)
                    mesh_fused = TriangleMesh()
                    mesh_fused.vertices = Vector3dVector(vertices)
                    mesh_fused.triangles = Vector3iVector(faces)
                    sampled_points = mesh_fused.sample_points_uniformly(number_of_points=num_points)
                else:
                    num_points = int(surface * total_points / total_surface)
                    sampled_points = box_mesh.sample_points_uniformly(number_of_points=num_points)
            elif sampling == "uniform":
                sampled_points = box_mesh.sample_points_uniformly(number_of_points=samples_per_box)
            elif sampling == "velodyne":
                if obj_type[i] in [0, 2, 3]:
                    box_meshes.append(box_mesh)
                else:
                    box_meshes.append(box_mesh1)
                    box_meshes.append(box_mesh2)
                continue
            
            points = np.asarray(sampled_points.points)  # Convert to numpy array
            if self.visu:
                trace = go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color='blue'
                    )
                )
                self._data.append(trace)
            pos_list.append(points)
        
        logger.info("Collisions: %d", count_cols)
        logger.info("Nb boxes: %d", len(boxes))

        if sampling == "velodyne":
            # Combine all the meshes into a single mesh
            combined_mesh = TriangleMesh()
            for box_mesh in box_meshes:
                combined_mesh += box_mesh
            # Convert Open3D mesh to trimesh
            vertices = combined_mesh.vertices
            triangles = combined_mesh.triangles
            mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(triangles))
            
            # create some rays
            # ~ Velodyne HDL-64 caracteristics
            x0 = (cfg.settings.xmax - cfg.settings.xmin)/2 + cfg.settings.xmin
            y0 = (cfg.settings.ymax - cfg.settings.ymin)/2 + cfg.settings.ymin
            z0 = cfg.settings.lmax + 1
            if cfg.settings.add_obj_type == 'complex_car' and cfg.settings.add_obj:
                z0 = cfg.settings.add_l2 + 1
            ray_origins = np.array([[x0, y0, z0]]*4500*64)
            azimuth = np.radians(np.arange(0,360,0.08))
            elevation = np.radians(np.arange(-30, 2, 0.5)) # np.arange(-24.8, 2, 0.4)
            x_dir = np.cos(elevation) * np.tile(np.cos(azimuth), (64, 1)).T
            y_dir = np.cos(elevation) * np.tile(np.sin(azimuth), (64, 1)).T
            z_dir = np.tile(np.sin(elevation), (4500, 1))
            ray_directions = np.column_stack((x_dir.flatten(), y_dir.flatten(), z_dir.flatten()))

            # Get the intersections
            locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)
            
            def keep_first_location(locations, index_ray):
                """Only keeps the first location for each index ray."""
                unique_index_ray, first_indices = np.unique(index_ray, return_index=True)
                first_locations = locations[first_indices]
                return first_locations
            
            first_locations = keep_first_location(locations, index_ray)
            pos_array = first_locations
            if self.visu:
                trace = go.Scatter3d(
                    x=pos_array[:, 0],
                    y=pos_array[:, 1],
                    z=pos_array[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color='blue'
                    )
                )
                self._data.append(trace)
        else:
            pos_array = np.concatenate(pos_list, axis=0)
        
        data_list = []
        #dummy zeros
        dummy_rgb = np.zeros(pos_array.shape)
        data_list.append(Data(
            pos=torch.from_numpy(pos_array).float(),
            rgb=torch.from_numpy(dummy_rgb).float(),
            intensity=torch.tensor([[1.]]*pos_array.shape[0]).float(),
            point_y=torch.tensor([[1.]]*pos_array.shape[0]).float(),
        ))
        if self.mode in ["train", "val"]:
            data_list[-1].pos_maxi = data_list[-1].pos[:, :2].max(0)[0]
        if cfg.settings.saveGT:
            keys = boxes[0].keys()
            tensor_list = list(map(lambda obj: torch.tensor(itemgetter(*keys)(obj)), boxes))
            gt_tensor = torch.stack(tensor_list)
            self.gt_coord = gt_tensor
            data_list[-1].gt = gt_tensor
        self.dataset = self.collate(data_list)
    
    def visualize(self):
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data',
                aspectratio=dict(x=1, y=1, z=1)
            ),
            title='3D Point Cloud of Sampled Boxes',
        )
        if self.visu:
            fig = go.Figure(data=self._data, layout=layout)
            fig.show()
        else:
            raise ValueError("Visualisation turned off")
    
    def get_data(self):
        return self.dataset
