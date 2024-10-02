import numpy as np
import pyviz3d.visualizer as viz


def create_color_palette():
    return np.array([
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ], dtype=np.uint8)



def generate_random_translation_rotation_matrix():
    translation = np.random.uniform(-10, 10, size=3)
    
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

def generate_random_cylinder_with_points(radius = 1, height = 1, points_density=1000):

    points_num = int(points_density * radius * height)
    
    points = []
    normals = []
    
    for _ in range(points_num):
        theta = np.random.uniform(0, 2 * np.pi)
        
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
    r = np.random.uniform(0,0.5)
    h = np.random.uniform(10,50)
    coords = []
    normals = []

    for i in range(num):
        if np.random.rand() < 0.5: #stay aside
            pass

        coord, normal = generate_random_cylinder_with_points(r,h)
        translation, rotation = generate_random_translation_rotation_matrix()
        coord = np.dot(coord, rotation.T) + translation
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
    for j in tqdm(range(0, 10)):
        # point_clouds, instance_labels = save_point_clouds_with_labels_as_numpy()
        dir_path = f"data/cylinders_normal/Area_{int(num)+1}/scene_{j}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        coord, normal = generate_cylinders(10)

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
