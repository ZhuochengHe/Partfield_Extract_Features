import torch
import lightning.pytorch as pl
from .dataloader import Demo_Dataset, Demo_Remesh_Dataset, Correspondence_Demo_Dataset
from torch.utils.data import DataLoader
from partfield.model.UNet.model import ResidualUNet3D
from partfield.model.triplane import TriplaneTransformer, get_grid_coord #, sample_from_planes, Voxel2Triplane
from partfield.model.model_utils import VanillaMLP
import torch.nn.functional as F
import torch.nn as nn
import os
import trimesh
import skimage
import numpy as np
import h5py
import torch.distributed as dist
from partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat, sample_triplane_feat_chunked
import json
import gc
import time
from plyfile import PlyData, PlyElement


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.automatic_optimization = False
        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_channels_low = cfg.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim=cfg.triplane_channels_low * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=cfg.triplane_channels_high,
        )
        self.sdf_decoder = VanillaMLP(input_dim=64,
                                      output_dim=1, 
                                      out_activation="tanh", 
                                      n_neurons=64, #64
                                      n_hidden_layers=6) #6
        self.use_pvcnn = cfg.use_pvcnnonly
        self.use_2d_feat = cfg.use_2d_feat
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                cfg.pvcnn,
                device="cuda",
                shape_min=-1, 
                shape_length=2,
                use_2d_feat=self.use_2d_feat) #.cuda()
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.grid_coord = get_grid_coord(256)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss(reduction='none')

        if cfg.regress_2d_feat:
            self.feat_decoder = VanillaMLP(input_dim=64,
                                output_dim=192, 
                                out_activation="GELU", 
                                n_neurons=64, #64
                                n_hidden_layers=6) #6
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")

    def predict_dataloader(self):
        if self.cfg.remesh_demo:
            dataset = Demo_Remesh_Dataset(self.cfg)        
        elif self.cfg.correspondence_demo:
            dataset = Correspondence_Demo_Dataset(self.cfg)
        else:
            dataset = Demo_Dataset(self.cfg)

        dataloader = DataLoader(dataset, 
                            num_workers=self.cfg.dataset.val_num_workers,
                            batch_size=self.cfg.dataset.val_batch_size,
                            shuffle=False, 
                            pin_memory=True,
                            drop_last=False)
        
        return dataloader           


    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        save_dir = self.cfg.result_name
        os.makedirs(save_dir, exist_ok=True)

        uid = batch['uid'][0]
        view_id = 0
        starttime = time.time()
        
        if uid == "car" or uid == "complex_car":
        # if uid == "complex_car":
            print("Skipping this for now.")
            print(uid)
            return

        ### Skip if model already processed
        if os.path.exists(f'{save_dir}/part_feat_{uid}_{view_id}.npy') or os.path.exists(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy'):
            print("Already processed "+uid)
            return

        N = batch['pc'].shape[0]
        assert N == 1

        if self.use_2d_feat: 
            print("ERROR. Dataloader not implemented with input 2d feat.")
            exit()
        else:
            pc_feat = self.pvcnn(batch['pc'], batch['pc'])

        planes = pc_feat
        planes = self.triplane_transformer(planes)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        if self.cfg.is_pc:
            tensor_vertices = batch['pc'].reshape(1, -1, 3).cuda().to(torch.float16)
            point_feat = sample_triplane_feat_chunked(part_planes, tensor_vertices) # N, M, C
            point_feat = point_feat.cpu().detach().numpy().reshape(-1, 448)

            np.save(f'{save_dir}/part_feat_{uid}_{view_id}.npy', point_feat)
            print(f"Exported part_feat_{uid}_{view_id}.npy")

            ###########
            from sklearn.decomposition import PCA
            data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

            pca = PCA(n_components=3)

            data_reduced = pca.fit_transform(data_scaled)
            data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
            colors_255 = (data_reduced * 255).astype(np.uint8)

            points = batch['pc'].squeeze().detach().cpu().numpy()

            if colors_255 is None:
                colors_255 = np.full_like(points, 255)  # Default to white color (255,255,255)
            else:
                assert colors_255.shape == points.shape, "Colors must have the same shape as points"
            
            # Convert to structured array for PLY format
            vertex_data = np.array(
                [(*point, *color) for point, color in zip(points, colors_255)],
                dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
            )

            # Create PLY element
            el = PlyElement.describe(vertex_data, "vertex")
            # Write to file
            filename = f'{save_dir}/feat_pca_{uid}_{view_id}.ply'
            PlyData([el], text=True).write(filename)
            print(f"Saved PLY file: {filename}")
            np.save(f'{save_dir}/part_feat_coord_{uid}_{view_id}.npy', np.concatenate([points, point_feat], axis=1))
            ############
        
        else:
            use_cuda_version = True
            if use_cuda_version:

                def sample_points(vertices, faces, n_point_per_face):
                    # Generate random barycentric coordinates
                    # borrowed from Kaolin https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/mesh/trianglemesh.py#L43
                    n_f = faces.shape[0]
                    u = torch.sqrt(torch.rand((n_f, n_point_per_face, 1),
                                                device=vertices.device,
                                                dtype=vertices.dtype))
                    v = torch.rand((n_f, n_point_per_face, 1),
                                    device=vertices.device,
                                    dtype=vertices.dtype)
                    w0 = 1 - u
                    w1 = u * (1 - v)
                    w2 = u * v

                    face_v_0 = torch.index_select(vertices, 0, faces[:, 0].reshape(-1))
                    face_v_1 = torch.index_select(vertices, 0, faces[:, 1].reshape(-1))
                    face_v_2 = torch.index_select(vertices, 0, faces[:, 2].reshape(-1))
                    points = w0 * face_v_0.unsqueeze(dim=1) + w1 * face_v_1.unsqueeze(dim=1) + w2 * face_v_2.unsqueeze(dim=1)
                    return points
                
                def per_face_coord_feat(part_planes, tensor_vertices, n_point_per_face, n_sample_each):
                    """
                    Args:
                        part_planes: triplane feature, shape (1, 3, C, H, W)
                        tensor_vertices: vertex coordinates, shape (1, N_vertices, 3)
                        n_point_per_face: 每个面采样的点数
                        n_sample_each: 每次 slice 采样的顶点数，避免 OOM
                    Returns:
                        coord_feat: (N_faces_total, 3 + latent_dim)
                    """
                    all_face_feat = []
                    all_face_coord = []

                    n_v = tensor_vertices.shape[1]
                    n_sample = n_v // n_sample_each + 1

                    for i_sample in range(n_sample):
                        # slice 顶点
                        v_slice = tensor_vertices[:, i_sample * n_sample_each : (i_sample + 1) * n_sample_each, :]
                        if v_slice.shape[1] == 0:
                            continue

                        # sample feature
                        sampled_feature = sample_triplane_feat_chunked(part_planes, v_slice)  # (1, n_slice_points, latent_dim)
                        B, n_points, latent_dim = sampled_feature.shape

                        # reshape 按 n_point_per_face 聚合
                        n_faces_in_slice = n_points // n_point_per_face
                        if n_faces_in_slice == 0:
                            continue

                        sampled_feature = sampled_feature[:, :n_faces_in_slice * n_point_per_face, :]  # trim 多余点
                        v_slice = v_slice[:, :n_faces_in_slice * n_point_per_face, :]

                        # reshape
                        sampled_feature = sampled_feature.reshape(B, n_faces_in_slice, n_point_per_face, latent_dim)
                        v_slice = v_slice.reshape(B, n_faces_in_slice, n_point_per_face, 3)

                        # 求平均
                        face_feat = sampled_feature.mean(dim=2)  # (B, n_faces_in_slice, latent_dim)
                        face_coord = v_slice.mean(dim=2)         # (B, n_faces_in_slice, 3)

                        all_face_feat.append(face_feat)
                        all_face_coord.append(face_coord)

                    # cat 所有 slice
                    all_face_feat = torch.cat(all_face_feat, dim=1)  # (B, N_faces_total, latent_dim)
                    all_face_coord = torch.cat(all_face_coord, dim=1)  # (B, N_faces_total, 3)

                    # 拼接坐标和特征
                    coord_feat = torch.cat([all_face_coord, all_face_feat], dim=-1)  # (B, N_faces_total, 3 + latent_dim)

                    # 如果你只想要 N_faces_total × (3 + latent_dim) 去掉 batch 维
                    coord_feat = coord_feat.squeeze(0)
                    print(coord_feat.shape)

                    return coord_feat  # (N_faces_total, 3 + latent_dim)

                def sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face):
                    n_sample_each = self.cfg.n_sample_each # we iterate over this to avoid OOM
                    n_v = tensor_vertices.shape[1]
                    n_sample = n_v // n_sample_each + 1
                    all_sample = []
                    for i_sample in range(n_sample):
                        sampled_feature = sample_triplane_feat_chunked(part_planes, tensor_vertices[:, i_sample * n_sample_each: i_sample * n_sample_each + n_sample_each,])
                        assert sampled_feature.shape[1] % n_point_per_face == 0
                        sampled_feature = sampled_feature.reshape(1, -1, n_point_per_face, sampled_feature.shape[-1])
                        sampled_feature = torch.mean(sampled_feature, axis=-2)
                        all_sample.append(sampled_feature)
                    return torch.cat(all_sample, dim=1)
                
                if self.cfg.vertex_feature:
                    tensor_vertices = batch['vertices'][0].reshape(1, -1, 3).to(torch.float32)
                    point_feat = sample_and_mean_memory_save_version(part_planes, tensor_vertices, 1)
                else:
                    n_point_per_face = self.cfg.n_point_per_face
                    tensor_vertices = sample_points(batch['vertices'][0], batch['faces'][0], n_point_per_face)
                    tensor_vertices = tensor_vertices.reshape(1, -1, 3).to(torch.float32)
                    point_feat = sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face)  # N, M, C

                #### Take mean feature in the triangle
                print("Time elapsed for feature prediction: " + str(time.time() - starttime))
                point_feat = point_feat.reshape(-1, 448).cpu().numpy()
                # np.save(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy', point_feat)
                np.save(f'{save_dir}/part_feat_coord_{uid}_{view_id}_batch.npy', per_face_coord_feat(part_planes, tensor_vertices, n_point_per_face, self.cfg.n_sample_each).cpu().numpy())
                # print(f"Exported part_feat_{uid}_{view_id}.npy")

                ###########
                from sklearn.decomposition import PCA
                data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                pca = PCA(n_components=3)

                data_reduced = pca.fit_transform(data_scaled)
                data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
                colors_255 = (data_reduced * 255).astype(np.uint8)
                V = batch['vertices'][0].cpu().numpy()
                F = batch['faces'][0].cpu().numpy()
                if self.cfg.vertex_feature:
                    colored_mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=colors_255, process=False)
                else:
                    colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)
                colored_mesh.export(f'{save_dir}/feat_pca_{uid}_{view_id}.ply')
                ############
                torch.cuda.empty_cache()

            else:
                ### Mesh input (obj file)
                V = batch['vertices'][0].cpu().numpy()
                F = batch['faces'][0].cpu().numpy()

                ##### Loop through faces #####
                num_samples_per_face = self.cfg.n_point_per_face

                all_point_feats = []
                for face in F:
                    # Get the vertices of the current face
                    v0, v1, v2 = V[face]

                    # Generate random barycentric coordinates
                    u = np.random.rand(num_samples_per_face, 1)
                    v = np.random.rand(num_samples_per_face, 1)
                    is_prob = (u+v) >1
                    u[is_prob] = 1 - u[is_prob]
                    v[is_prob] = 1 - v[is_prob]
                    w = 1 - u - v
                    
                    # Calculate points in Cartesian coordinates
                    points = u * v0 + v * v1 + w * v2 

                    tensor_vertices = torch.from_numpy(points.copy()).reshape(1, -1, 3).cuda().to(torch.float32)
                    point_feat = sample_triplane_feat_chunked(part_planes, tensor_vertices) # N, M, C 

                    #### Take mean feature in the triangle
                    point_feat = torch.mean(point_feat, axis=1).cpu().detach().numpy()
                    all_point_feats.append(point_feat)                  
                ##############################
                
                all_point_feats = np.array(all_point_feats).reshape(-1, 448)
                
                point_feat = all_point_feats

                np.save(f'{save_dir}/part_feat_{uid}_{view_id}.npy', point_feat)
                print(f"Exported part_feat_{uid}_{view_id}.npy")
                
                ###########
                from sklearn.decomposition import PCA
                data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                pca = PCA(n_components=3)

                data_reduced = pca.fit_transform(data_scaled)
                data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
                colors_255 = (data_reduced * 255).astype(np.uint8)

                colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)
                colored_mesh.export(f'{save_dir}/feat_pca_{uid}_{view_id}.ply')
                ############

        print("Time elapsed: " + str(time.time()-starttime))
            
        return 