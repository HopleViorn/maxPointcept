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


    for _ in range(points_num):
        theta = np.random.uniform(0, angle)
        
        z = np.random.uniform(0, height)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        normal = np.array([-radius*np.cos(theta), -radius*np.sin(theta), 0])  # Normal vector is tangential to the cylinder surface
        
        points.append([x, y, z])
        normals.append(normal)
    
    points = np.array(points)
    normals = np.array(normals)
    
    return points, normals

def generate_cylinders(num = 10):
    coords = []
    normals = []

    rs = []
    ts = []
    rr = []

    for i in range(num):
        r = np.random.uniform(0.001, scale/40)
        rr.append(r)
        h = np.random.uniform(0.5*scale,scale)

        coord, normal = generate_random_cylinder_with_points(r,h)

        translation, rotation = generate_random_translation_rotation_matrix()
        rs.append(rotation)
        ts.append(translation)

        coord = np.dot(coord, rotation.T) + translation
        normal = np.dot(normal, rotation.T)
        coords.append(coord)
        normals.append(normal)

    for i in range(int(num/2)):
        indx =np.random.randint(0, num)
        r = np.random.uniform(0.001, scale/40)
        h = np.random.uniform(0.5*scale,scale)

        coord, normal = generate_random_cylinder_with_points(r,h)

        translation, rotation = ts[indx], rs[indx]

        coord = np.dot(coord+np.array([r+rr[indx],0,0]), rotation.T) + translation
        normal = np.dot(normal, rotation.T)
        coords.append(coord)
        normals.append(normal)

    for i in range(int(num/2)):
        indx =np.random.randint(0, num)
        r = np.random.uniform(0.001, scale/40)
        h = np.random.uniform(0.5*scale,scale)

        coord, normal = generate_random_cylinder_with_points(r,h)

        translation, rotation = ts[indx], rs[indx]

        coord = np.dot(coord+np.array([-(r+rr[indx]),0,0]), rotation.T) + translation
        normal = np.dot(normal, rotation.T)
        coords.append(coord)
        normals.append(normal)


    coords = np.concatenate(coords, axis=0)
    normals = np.concatenate(normals, axis=0)
    return coords, normals



import sys,os
from tqdm import tqdm

if __name__=='__main__':

    num=sys.argv[1]
    for j in tqdm(range(0, 1000)):
        # point_clouds, instance_labels = save_point_clouds_with_labels_as_numpy()
        dir_path = f"data/cylinders_normal/Area_{int(num)+1}/scene_{j}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        coord, normal = generate_cylinders(20)

        color = np.ones((coord.shape[0], 3))

        np.save(os.path.join(dir_path,"color.npy"), color)
        np.save(os.path.join(dir_path,"coord.npy"), coord)
        np.save(os.path.join(dir_path,"normal.npy"), normal)
        
        # np.save(os.path.join(dir_path,"instance.npy"), instance)
        # np.save(os.path.join(dir_path,'normal.npy'), orient)
        # np.save(os.path.join(dir_path,'segment.npy'),np.ones((point_clouds.shape[0],1)))

    coord, normal = generate_cylinders(10)
    v = viz.Visualizer()
    v.add_points('RGB Color', coord, normal)
    v.add_lines('Normals', coord, coord + normal, visible=True)
    v.save('visualization/example')
