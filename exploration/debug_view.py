import numpy as np
import point_cloud_utils as pcu
from point_cloud import *
import pyrender
import trimesh
import random
from pyrender_view_render import *
from trimesh.creation import cone

DATA_DIR = "/Datasets/ModelNet10/ModelNet10"

scale = 1
image_height = 500 * scale
image_width = 500 * scale
fx = 250 * scale
fy = 250 * scale
cx = image_width / 2
cy = image_height / 2

# image_height = 2000
# image_width = 2000
# fx = 500
# fy = 500
# cx = image_width / 2
# cy = image_height / 2
k = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape((3, 3))

W_R_C = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
C_R_W = W_R_C.T
PY_R_W = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

if __name__ == "__main__":
    # Load Object Instance
    object_class = "chair"
    file = "train"
    instance = 2
    path_to_file = path_generator(DATA_DIR, object_class, file, instance)

    # Load vertices and faces
    v, f = pcu.load_mesh_vf(path_to_file)

    # Sample points from mesh
    pointcloud = sample_mesh_random(v, f, num_samples=600)
    pointcloud_centered, center = center_pointcloud_v2(pointcloud)
    center = np.reshape(center, (1, 3))

    # Camera Pose
    camera_center = np.array([30, -30, 30]) + center
    pose = transformation_matrix(W_R_C, camera_center)

    draw_point_cloud_with_cameras(
        pointcloud,
        title="Point Cloud",
        cameras=[(k, [image_width, image_height], pose)],
    )

    # Calculate the rotation to align the camera with the focal point
    rotation = calculate_aligning_rotation(
        camera_center - center, k, image_width, image_height, c_R_w=C_R_W
    )

    # Camera Pose in World Coordinates
    cal_cam_pose = transformation_matrix(rotation, camera_center)
    draw_point_cloud_with_cameras(
        pointcloud,
        title="Point Cloud",
        cameras=[(k, [image_width, image_height], cal_cam_pose)],
    )

    print("Camera Given Pose: ", pose)
    # print("Camera Calculated Pose: ", cal_cam_pose)

    # Rendering
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.show()
    scene = pyrender.Scene()
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    renderer = pyrender.OffscreenRenderer(image_width, image_height)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    # Camera Pose in Pyrender Coordinates
    py_T_w = transformation_matrix(PY_R_W, np.zeros([1, 3]))
    rotation = C_R_W @ pose[:3, :3]
    translation = PY_R_W @ camera_center.T

    print("Camera Rotation in World Coordinates: ", pose[:3, :3])

    # r = R.from_euler("y", -180, degrees=True).as_matrix()
    camera_pose = transformation_matrix(rotation, translation)

    # camera_pose = transformation_matrix(r @ PY_R_W @ W_R_C, translation)

    # Scene from cam visualization
    scene.add(mesh, pose=py_T_w)
    scene.add(camera, pose=camera_pose)

    pyrender.Viewer(
        scene, use_raymond_lighting=True, viewport_size=(image_width, image_height)
    )
