import open3d as o3d
import numpy as np


# Sample a point cloud on the surface of the mesh
def sample_point_cloud_from_shape(shape, num_points=1000):
    # Sample points on the surface of the mesh
    pcd = shape.sample_points_poisson_disk(num_points)
    return pcd


def to_pcd(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color(np.random.rand(3))

    return point_cloud

import copy


# Function to generate random shapes (e.g., spheres, boxes, and cylinders)
def generate_random_shape():
    shape_type = np.random.choice(["sphere", "box", "cylinder", "cone"])
    # shape_type = np.random.choice(["cylinder"])

    if shape_type == "sphere":
        shape = o3d.geometry.TriangleMesh.create_sphere(
            radius=np.random.uniform(0.5, 1.0)
        )
    elif shape_type == "box":
        shape = o3d.geometry.TriangleMesh.create_box(
            width=np.random.uniform(0.5, 1.0),
            height=np.random.uniform(0.5, 1.0),
            depth=np.random.uniform(0.5, 1.0),
        )
    elif shape_type == "cylinder":
        shape = o3d.geometry.TriangleMesh.create_cylinder(
            radius=np.random.uniform(0.3, 0.7), height=np.random.uniform(0.5, 1.50)
        )
    elif shape_type == "cone":
        shape = o3d.geometry.TriangleMesh.create_cone(
            radius=np.random.uniform(0.3, 0.7), height=np.random.uniform(0.5, 1.5)
        )
    elif shape_type == "icosahedron":
        shape = o3d.geometry.TriangleMesh.create_icosahedron(
            radius=np.random.uniform(0.5, 1.0)
        )

    # Apply random translation and rotation
    shape.translate(np.random.uniform(-2, 2, size=3))
    shape.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz(
            np.random.uniform(0, 2 * np.pi, size=3)
        )
    )

    return shape


def generate_cylinders():
    rad=np.random.uniform(4, 15)
    shape = o3d.geometry.TriangleMesh.create_cylinder(
        radius=rad, height=np.random.uniform(0.5, 150)
    )
    # Apply random translation and rotation

    shape.translate(np.random.uniform(-20, 20, size=3))

    shape.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz(
            np.random.uniform(0, 2 * np.pi, size=3)
        )
    )

    return shape


def sample_point_cloud_from_shape(shape, num_points=1000):
    pcd = shape.sample_points_poisson_disk(num_points)
    return pcd


def point_cloud_to_numpy_array(pcd):
    points = np.asarray(pcd.points)
    return points


def save_point_clouds_with_labels_as_numpy(num_shapes=10, num_points=1000):
    point_cloud_arrays = []
    instance_label_arrays = []
    geometries = []

    for shape_id in range(num_shapes):
        # shape = generate_random_shape()
        shape = generate_cylinders()
        shape.compute_vertex_normals()
        pcd = sample_point_cloud_from_shape(shape, num_points)
        points_array = point_cloud_to_numpy_array(pcd)
        point_cloud_arrays.append(points_array)
        geometries.append(shape)
        geometries.append(pcd)

        labels_array = np.full((points_array.shape[0],), shape_id, dtype=np.int32)
        instance_label_arrays.append(labels_array)

    all_point_clouds = np.vstack(point_cloud_arrays)
    all_instance_labels = np.hstack(instance_label_arrays).reshape(-1,1)

    return all_point_clouds, all_instance_labels



def random_rotation_matrix():
    random_matrix = np.random.randn(3, 3)
    q, r = np.linalg.qr(random_matrix)
    if np.linalg.det(q) < 0:
        q[:, 2] = -q[:, 2]
    return q

# Step 2: Generate a random scaling matrix
def random_scaling_matrix():
    # Generate random scaling factors for each axis (non-negative values)
    # scaling_factors = np.random.uniform(0.5, 2.0, size=(3,))
    # scaling_matrix = np.diag(scaling_factors)
    scaling_matrix=np.identity(3)
    return scaling_matrix

# Step 3: Generate a random translation (offset)
def random_translation_vector():
    # Generate a random 3D translation vector
    translation_vector = np.random.uniform(-50, 50, size=(3,))
    return translation_vector

# Step 4: Apply all transformations (scaling, rotation, and translation)
def apply_transformations(points,rotation_matrix,scaling_matrix,translation_vector):

    scaled_points = points @ scaling_matrix.T  # Nx3 points scaled
    rotated_points = scaled_points @ rotation_matrix.T  # Nx3 points rotated
    transformed_points = rotated_points + translation_vector

    return transformed_points

def generate_cylinder(rad, scaling, offset, rot, off=np.array([0,0,0])):
    
    offset += off

    shape = o3d.geometry.TriangleMesh.create_cylinder(
        radius =  rad, height=np.random.uniform(20, 150)
    )

    n_samples=1000

    pcd = shape.sample_points_poisson_disk(n_samples)

    pcd_proj = copy.copy(np.asarray(pcd.points))
    for i in range(len(pcd_proj)):
        pcd_proj[i,0]=0
        pcd_proj[i,1]=0

    # pcd_proj = to_pcd(pcd_proj)
    pcd=np.asarray(pcd.points)

    orient = np.array([[0,0,1]]).repeat(n_samples,axis=0)

    # orient.rotate(rot,center=(0, 0, 0))


    # pcd.translate(trans)
    # pcd.rotate(rot)

    # pcd_proj.translate(trans)
    # pcd_proj.rotate(rot)

    pcd=apply_transformations(pcd,rot,scaling,offset)
    pcd_proj=apply_transformations(pcd_proj,rot,scaling,offset)
    orient=apply_transformations(orient,rot,np.identity(3),offset)

    # o3d.visualization.draw_geometries([to_pcd(pcd),to_pcd(pcd_proj)])
    

    return pcd,pcd_proj,orient


def get_all_cylinders():
    pc,pcp,ori=[],[],[]
    instance_label_arrays=[]
    for i in range(10):

        # offset=np.random.uniform(-20,20,size=(3,1))
        # rad = np.random.uniform(0.3, 50)
        # trans = np.random.uniform(-20, 20, size=3) 

        rad = np.random.uniform(0.3, 5)
        rot = random_rotation_matrix()
        offset = random_translation_vector()
        scaling = random_scaling_matrix()


        a1,b1,c1=generate_cylinder(rad,scaling,offset,rot,np.array([0,0,0]))
        pc.append(a1)
        pcp.append(b1)
        ori.append(c1)

        a2,b2,c2=generate_cylinder(rad,scaling,offset,rot,np.array([2*rad,0,0]))
        pc.append(a2)
        pcp.append(b2)
        ori.append(c2)

        # o3d.visualization.draw_geometries([to_pcd(a1),to_pcd(b1),to_pcd(a2),to_pcd(b2)])

        labels_array = np.full((a1.shape[0],), i, dtype=np.int32)
        instance_label_arrays.append(labels_array)
        labels_array = np.full((a1.shape[0],), i, dtype=np.int32)
        instance_label_arrays.append(labels_array)
    

    pc=np.vstack(pc)
    pcp=np.vstack(pcp)
    ori=np.vstack(ori)
    all_instance_labels = np.hstack(instance_label_arrays).reshape(-1,1)

    return pc,pcp,all_instance_labels,ori


import os
import sys
from tqdm import tqdm

if __name__=='__main__':

    num=sys.argv[1]
    for j in tqdm(range(0, 10)):
        # point_clouds, instance_labels = save_point_clouds_with_labels_as_numpy()
        point_clouds, pointproj, instance, orient = get_all_cylinders()
        dir_path = f"data/cylinders/Area_{int(num)+1}/scene_{j}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


        np.save(os.path.join(dir_path,"coord.npy"), point_clouds)
        np.save(os.path.join(dir_path,"color.npy"), pointproj)
        np.save(os.path.join(dir_path,"instance.npy"), instance)
        np.save(os.path.join(dir_path,'normal.npy'), orient)
        # np.save(os.path.join(dir_path,'segment.npy'),np.ones((point_clouds.shape[0],1)))
        

