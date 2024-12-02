import open3d as o3d
import numpy as np
import cv2
import re

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask


blender2opencv = np.float32([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

def depth_pts3d(frame_path):

    depth = readPFM(frame_path + "_depth.pfm")
    mask = cv2.imread(frame_path + "_mask.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0 > 0.1
    depth *= mask
    # depth *= 6.5

    input_metadata = np.load(frame_path + "_metadata.npz")

    camera_pose = input_metadata['camera_pose'].astype(np.float32)
    intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

    X_cam, valid_mask = depthmap_to_camera_coordinates(depth, intrinsics)


    return X_cam, camera_pose


frame_1_path = "../data/pointodyssey/train/ani/0182"
frame_2_path = "../data/pointodyssey/train/ani/0172"

# coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2,origin=[0, 0, 0])


pts_1, w_T_cam1 = depth_pts3d(frame_1_path)
pts_1_reshape = pts_1.reshape(-1, 3)
pts_2, w_T_cam2 = depth_pts3d(frame_2_path)

cam1_T_cam2 = np.linalg.inv(w_T_cam1) @ w_T_cam2
R = cam1_T_cam2[:3, :3]
t = cam1_T_cam2[:3, 3]

# Express in absolute coordinates (invalid depth values)
pts_2_trans = (np.einsum("ik, vuk -> vui", R, pts_2) + t[None, None, :]).reshape(-1, 3)


pcd_1 = o3d.geometry.PointCloud()
pcd_1.points = o3d.utility.Vector3dVector(pts_1_reshape)

pcd_2 = o3d.geometry.PointCloud()
pcd_2.points = o3d.utility.Vector3dVector(pts_2_trans)

pcd_1.paint_uniform_color([0.0, 0.0, 1.0])
pcd_2.paint_uniform_color([1.0, 0.0, 0.0])

# o3d.visualization.draw_geometries([coor,pcd_1, pcd_2])



# 创建一个无头渲染器 (Offscreen Renderer)
width, height = 640, 480
render = o3d.visualization.rendering.OffscreenRenderer(width, height)

# 设置点云的渲染材质，包括点大小
material = o3d.visualization.rendering.MaterialRecord()
material.point_size = 5.0  # 设置点的大小

# 将点云添加到渲染器中
render.scene.add_geometry("point_cloud", pcd_1, material)
render.scene.add_geometry("point_cloud", pcd_2, material)
# 设置相机内参 (fx, fy, cx, cy 分别为焦距和主点坐标)
fx, fy, cx, cy = 288.9353025, 288.9353025, width/2-0.5, height/2-0.5
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)



#extrinsics = np.linalg.inv(extrinsics)
extrinsics = np.array([
    [0.866025, -0.500000, -0.000000, 20.000000],
    [0.000000, 0.000000, -1.000000, 0.000000],
    [0.500000, 0.866025, 0.000000, -20.000000],
    [0.000000, 0.000000, 0.000000, 1.000000]
])
# 设置相机视角
render.scene.camera.look_at([0, 0, 0], [0, 0, -1], [0, -1, 0])
render.setup_camera(intrinsics, extrinsics)

# 渲染并捕获图像

image = render.render_to_image()
image = np.asarray(image)  # 转换为 numpy 数组
image = gamma_correction(image)  # 进行 gamma 校正
#image = image.astype(np.uint8)

#output_filename = path+'/rendered_image.png'
o3d.io.write_image('1.png', o3d.geometry.Image(image.astype(np.uint8)))
