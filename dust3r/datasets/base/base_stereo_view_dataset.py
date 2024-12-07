# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# base class for implementing datasets
# --------------------------------------------------------
import PIL
import numpy as np
import torch

from dust3r.datasets.base.easy_dataset import EasyDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dust3r.datasets.utils.cropping as cropping


class BaseStereoViewDataset(EasyDataset):
    """ Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(self, *,  # only keyword arguments
                 split=None,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=ImgNorm,
                 aug_crop=False,
                 aug_f=False,
                 seed=None,
                 depth_prior_name='depthpro'):
        self.num_views = 2
        self.split = split
        self.depth_prior_name = depth_prior_name
        self._set_resolutions(resolution)
        self.aug_f = aug_f
        self.transform = transform
        if isinstance(transform, str):
            transform = eval(transform)

        self.aug_crop = aug_crop
        self.seed = seed

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = '['+';'.join(f'{w}x{h}' for w, h in self._resolutions)+']'
        return f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace('self.', '').replace('\n', '').replace('   ', '')

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()

    def pixel_to_pointcloud(self, depth_map, focal_length_px):
        """
        Convert a depth map to a 3D point cloud.

        Args:
        depth_map (numpy.ndarray): The input depth map with shape (H, W), where each value represents the depth at that pixel.
        focal_length_px (float): The focal length of the camera in pixels.

        Returns:
        numpy.ndarray: The resulting point cloud with shape (H, W, 3), where each point is represented by (X, Y, Z).
        """
        height, width = depth_map.shape
        cx = width / 2
        cy = height / 2

        # Create meshgrid for pixel coordinates
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        #depth_map[depth_map>100]=0
        # Convert pixel coordinates to camera coordinates
        Z = depth_map
        X = (u - cx) * Z / focal_length_px
        Y = (v - cy) * Z / focal_length_px
        
        # Stack the coordinates into a point cloud (H, W, 3)
        point_cloud = np.dstack((X, Y, Z)).astype(np.float32)
        point_cloud = self.normalize_pointcloud(point_cloud)
        # Optional: Filter out invalid depth values (if necessary)
        # point_cloud = point_cloud[depth_map > 0]
        #print(point_cloud)
        return point_cloud

    def normalize_pointcloud(self, point_cloud):
        min_vals = np.min(point_cloud, axis=(0, 1))
        max_vals = np.max(point_cloud, axis=(0, 1))
        #print(min_vals, max_vals)
        normalized_point_cloud = (point_cloud - min_vals) / (max_vals - min_vals)
        return normalized_point_cloud

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        #print(ar_idx, self.dataset_label,resolution)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & (np.isfinite(pts3d).all(axis=-1))[..., None]

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            assert width >= height
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(self, image, depthmap, pred_depth, intrinsics, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        #print(cx, W-cx,cy, H-cy)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        # scale = rng.choice([0.5, 0.75, 1, 1.25], size=1, replace=False)[0]
        # #print(scale)
        # crop_resolution = (resolution[0]*scale, resolution[1]*scale)
        # #print(crop_resolution)
        # assert min_margin_x > W/5, f'Bad principal point in view={info}'
        # assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # if rng.choice([0, 1], size=1, replace=False)[0]==0:
        #   min_margin_x = min(min_margin_x, int(crop_resolution[0]/2))
        #   min_margin_y = min(min_margin_y, int(crop_resolution[1]/2))
        

        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        #print(resolution, crop_resolution,crop_bbox)
        # print(crop_bbox)
        image, depthmap, pred_depth, intrinsics = cropping.crop_image_depthmap(image, depthmap, pred_depth,  intrinsics, crop_bbox)
        #print(image.size)
        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1*W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        # center-crop
        target_resolution = np.array(resolution)
        if self.aug_f:
            crop_scale = rng.choice([0.8, 0.9, 1.0], size=1, replace=False)[0]

            image, depthmap, pred_depth, intrinsics = cropping.center_crop_image_depthmap(image, depthmap, pred_depth, intrinsics, crop_scale)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, pred_depth, intrinsics = cropping.rescale_image_depthmap(image, depthmap, pred_depth, intrinsics, target_resolution)
        #print(image.size)
        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, pred_depth, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, pred_depth, intrinsics, crop_bbox)
        #print(image.size)
        return image, depthmap, pred_depth, intrinsics2


def is_good_type(key, v):
    """ returns (is_good, err_msg) 
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view['dataset'])
    label = sel(view['label'])
    instance = sel(view['instance'])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        # rectify portrait to landscape
        assert view['img'].shape == (3, height, width)
        view['img'] = view['img'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width)
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3)
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)

        assert view['pred_depth'].shape == (height, width)
        view['pred_depth'] = view['pred_depth'].swapaxes(0, 1)
        # transpose x and y pixels
        view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]
