import numpy as np
import pyviz3d.visualizer as viz

scale = 5

def generate_random_translation_rotation_matrix():
    translation = np.random.uniform(-scale/3, scale/3, size=3)
    
    angles = np.random.uniform(0, 2 * np.pi, size=3)  # Random angles for rotation around x, y, and z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    
    rotation_matrix = Rz @ Ry @ Rx
    
    return translation, rotation_matrix

def generate_random_cylinder_with_points(radius = 1, height = 1, points_density=2000*scale):

    angle = np.random.uniform(np.pi / 2, 2 * np.pi)
    points_num = max(int(points_density * radius * height * (angle / (2 * np.pi))),100)
    
    points = []
    normals = []
    towards = []


    for _ in range(points_num):
        theta = np.random.uniform(0, angle)
        
        z = np.random.uniform(0, height)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        normal = np.array([-radius*np.cos(theta), -radius*np.sin(theta), 0])  # Normal vector is tangential to the cylinder surface
        toward = np.array([0, 0, 1]) # toward vector is along the cylinder axis
        
        points.append([x, y, z])
        normals.append(normal)
        towards.append(toward)
    
    points = np.array(points)
    normals = np.array(normals)
    towards = np.array(towards)
    
    return points, normals, towards

def generate_cylinders(num = 10, factor = 40):
    coords = []
    normals = []
    towards = []

    rs = []
    ts = []
    rr = []

    for i in range(num):
        r = np.random.uniform(0.001, scale/factor)
        rr.append(r)
        h = np.random.uniform(0.5*scale,scale)

        coord, normal, toward = generate_random_cylinder_with_points(r,h)

        translation, rotation = generate_random_translation_rotation_matrix()
        rs.append(rotation)
        ts.append(translation)

        coord = np.dot(coord, rotation.T) + translation
        normal = np.dot(normal, rotation.T)
        toward = np.dot(toward, rotation.T)

        coords.append(coord)
        normals.append(normal)
        towards.append(toward)


    for i in range(int(num/2)):
        indx =np.random.randint(0, num)
        r = np.random.uniform(0.001, scale/factor)
        h = np.random.uniform(0.5*scale,scale)

        coord, normal, toward = generate_random_cylinder_with_points(r,h)

        translation, rotation = ts[indx], rs[indx]

        coord = np.dot(coord+np.array([r+rr[indx],0,0]), rotation.T) + translation
        normal = np.dot(normal, rotation.T)
        toward = np.dot(toward, rotation.T)

        coords.append(coord)
        normals.append(normal)
        towards.append(toward)

    for i in range(int(num/2)):
        indx =np.random.randint(0, num)
        r = np.random.uniform(0.001, scale/factor)
        h = np.random.uniform(0.5*scale,scale)

        coord, normal, toward = generate_random_cylinder_with_points(r,h)

        translation, rotation = ts[indx], rs[indx]

        coord = np.dot(coord+np.array([-(r+rr[indx]),0,0]), rotation.T) + translation
        normal = np.dot(normal, rotation.T)
        toward = np.dot(toward, rotation.T)

        coords.append(coord)
        normals.append(normal)
        towards.append(toward)

    coords = np.concatenate(coords, axis=0)
    normals = np.concatenate(normals, axis=0)
    towards = np.concatenate(towards, axis=0)
    v = np.array([1, 1, 1])
    dot_product = np.dot(towards, v)
    towards[dot_product < 0] *=-1

    return coords, normals, towards

import sys,os
from tqdm import tqdm
factor = 100
if __name__=='__main__':

    num=sys.argv[1]
    for j in tqdm(range(0, 10*factor)):
        # point_clouds, instance_labels = save_point_clouds_with_labels_as_numpy()
        dir_path = f"data/cylinders_normal/Area_{int(num)+1}/scene_{j}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        coord, normal, towards = generate_cylinders(10)

        color = np.ones((coord.shape[0], 3))
        np.save(os.path.join(dir_path,"color.npy"), color)
        np.save(os.path.join(dir_path,"coord.npy"), coord)
        np.save(os.path.join(dir_path,"normal.npy"), normal)
        np.save(os.path.join(dir_path,"vector_attr_0.npy"), towards)
    
    for j in tqdm(range(10*factor, 15*factor)):
        # point_clouds, instance_labels = save_point_clouds_with_labels_as_numpy()
        dir_path = f"data/cylinders_normal/Area_{int(num)+1}/scene_{j}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        coord, normal, towards = generate_cylinders(10,5)

        color = np.ones((coord.shape[0], 3))
        np.save(os.path.join(dir_path,"color.npy"), color)
        np.save(os.path.join(dir_path,"coord.npy"), coord)
        np.save(os.path.join(dir_path,"normal.npy"), normal)
        np.save(os.path.join(dir_path,"vector_attr_0.npy"), towards)

    for j in tqdm(range(15*factor, 20*factor)):
        # point_clouds, instance_labels = save_point_clouds_with_labels_as_numpy()
        dir_path = f"data/cylinders_normal/Area_{int(num)+1}/scene_{j}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        coord, normal, towards = generate_cylinders(10,160)

        color = np.ones((coord.shape[0], 3))
        np.save(os.path.join(dir_path,"color.npy"), color)
        np.save(os.path.join(dir_path,"coord.npy"), coord)
        np.save(os.path.join(dir_path,"normal.npy"), normal)
        np.save(os.path.join(dir_path,"vector_attr_0.npy"), towards)
    
    # for j in tqdm(range(20*factor, 30*factor)):
    #     # point_clouds, instance_labels = save_point_clouds_with_labels_as_numpy()
    #     dir_path = f"data/cylinders_normal/Area_{int(num)+1}/scene_{j}"
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)

    #     coord, normal, towards = generate_cylinders(10,300)

    #     color = np.ones((coord.shape[0], 3))
    #     np.save(os.path.join(dir_path,"color.npy"), color)
    #     np.save(os.path.join(dir_path,"coord.npy"), coord)
    #     np.save(os.path.join(dir_path,"normal.npy"), normal)
    #     np.save(os.path.join(dir_path,"vector_attr_0.npy"), towards)

    coord, normal, towards = generate_cylinders(10)
    v = viz.Visualizer()
    v.add_points('RGB Color', coord, normal)
    v.add_lines('Normals', coord, coord + normal, visible=True)
    v.add_lines('towards ', coord, coord + towards, visible=True)
    v.save('visualization/example')
