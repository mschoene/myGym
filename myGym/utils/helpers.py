import numpy as np

def get_workspace_dict():
    ws_dict = {'baskets':  {'urdf': 'baskets.urdf', 'texture': 'baskets.jpg',
                                            'transform': {'position':[3.18, -3.49, -1.05], 'orientation':[0.0, 0.0, -0.4*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.56, -1.71, 0.6], [-1.3, 3.99, 0.6], [-3.43, 0.67, 1.0], [2.76, 2.68, 1.0], [-0.54, 1.19, 3.4]],
                                                        'target': [[0.53, -1.62, 0.59], [-1.24, 3.8, 0.55], [-2.95, 0.83, 0.8], [2.28, 2.53, 0.8], [-0.53, 1.2, 3.2]]},
                                            'borders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]},
                                'collabtable': {'urdf': 'collabtable.urdf', 'texture': 'collabtable.jpg',
                                            'transform': {'position':[0.45, -5.1, -1.05], 'orientation':[0.0, 0.0, -0.35*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.25, 3.24, 1.2], [-0.44, -1.34, 1.0], [-1.5, 2.6, 1.0], [1.35, -1.0, 1.0], [-0.1, 1.32, 1.4]],
                                                        'target': [[-0.0, 0.56, 0.6], [-0.27, 0.42, 0.7], [-1, 2.21, 0.8], [-0.42, 2.03, 0.2], [-0.1, 1.2, 0.7]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.2, 0.2, 0.2]},
                                'darts':    {'urdf': 'darts.urdf', 'texture': 'darts.jpg',
                                            'transform': {'position':[-1.4, -6.7, -1.05], 'orientation':[0.0, 0.0, -1.0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.0, 2.1, 1.0], [0.0, -1.5, 1.2], [2.3, 0.5, 1.0], [-2.6, 0.5, 1.0], [-0.0, 1.1, 4.9]],
                                                        'target': [[0.0, 0.0, 0.7], [-0.0, 1.3, 0.6], [1.0, 0.9, 0.9], [-1.6, 0.9, 0.9], [-0.0, 1.2, 3.1]]},
                                            'borders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]},
                                'drawer':   {'urdf': 'drawer.urdf', 'texture': 'drawer.jpg',
                                            'transform': {'position':[-4.81, 1.75, -1.05], 'orientation':[0.0, 0.0, 0.0*np.pi]},
                                            'robot': {'position': [0.0, 0.2, 0.0], 'orientation': [0, 0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.14, -1.63, 1.0], [-0.14, 3.04, 1.0], [-1.56, -0.92, 1.0], [1.2, -1.41, 1.0], [-0.18, 0.88, 2.5]],
                                                        'target': [[-0.14, -0.92, 0.8], [-0.14, 2.33, 0.8], [-0.71, -0.35, 0.7], [0.28, -0.07, 0.6], [-0.18, 0.84, 2.1]]},
                                            'borders':[-0.7, 0.7, 0.4, 1.3, 0.8, 0.1]},
                                'football': {'urdf': 'football.urdf', 'texture': 'football.jpg',
                                            'transform': {'position':[4.2, -5.4, -1.05], 'orientation':[0.0, 0.0, -1.0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.0, 2.1, 1.0], [0.0, -1.7, 1.2], [3.5, -0.6, 1.0], [-3.5, -0.7, 1.0], [-0.0, 2.0, 4.9]],
                                                        'target': [[0.0, 0.0, 0.7], [-0.0, 1.3, 0.2], [3.05, -0.2, 0.9], [-2.9, -0.2, 0.9], [-0.0, 2.1, 3.6]]},
                                            'borders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]},
                                'fridge':   {'urdf': 'fridge.urdf', 'texture': 'fridge.jpg',
                                            'transform': {'position':[1.6, -5.95, -1.05], 'orientation':[0.0, 0.0, 0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, -1.3, 1.0], [0.0, 2.35, 1.2], [-1.5, 0.85, 1.0], [1.4, 0.85, 1.0], [0.0, 0.55, 2.5]],
                                                        'target': [[0.0, 0.9, 0.7], [0.0, 0.9, 0.6], [0.0, 0.55, 0.5], [0.4, 0.55, 0.7], [0.0, 0.45, 1.8]]},
                                            'borders':[-0.7, 0.7, 0.3, 0.5, -0.9, -0.9]},
                                'maze':     {'urdf': 'maze.urdf', 'texture': 'maze.jpg',
                                            'transform': {'position':[6.7, -3.1, 0.0], 'orientation':[0.0, 0.0, -0.5*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.1], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, -1.4, 2.3], [-0.0, 5.9, 1.9], [4.7, 2.7, 2.0], [-3.2, 2.7, 2.0], [-0.0, 3.7, 5.0]],
                                                        'target': [[0.0, -1.0, 1.9], [-0.0, 5.6, 1.7], [3.0, 2.7, 1.5], [-2.9, 2.7, 1.7], [-0.0, 3.65, 4.8]]},
                                            'borders':[-2.5, 2.2, 0.7, 4.7, 0.05, 0.05]},
                                'stairs':   {'urdf': 'stairs.urdf', 'texture': 'stairs.jpg',
                                            'transform': {'position':[-5.5, -0.08, -1.05], 'orientation':[0.0, 0.0, -0.20*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.04, -1.64, 1.0], [0.81, 3.49, 1.0], [-2.93, 1.76, 1.0], [4.14, 0.33, 1.0], [2.2, 1.24, 3.2]],
                                                        'target': [[0.18, -1.12, 0.85], [0.81, 2.99, 0.8], [-1.82, 1.57, 0.7], [3.15, 0.43, 0.55], [2.17, 1.25, 3.1]]},
                                            'borders':[-0.5, 2.5, 0.8, 1.6, 0.1, 0.1]},
                                'table':    {'urdf': 'table.urdf', 'texture': 'table.jpg',
                                            'transform': {'position':[-0.0, -0.0, -1.05], 'orientation':[0.0, 0.0, 0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0., 0.85, 1.4],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]],
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.80, 1.],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]},
                                'table_tiago': {'urdf': 'table_tiago.urdf', 'texture': 'table.jpg',
                                            'transform': {'position':[-0.0, -0.0, -1.05], 'orientation':[0.0, 0.0, 0*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[0.0, 2.4, 1.0], [-0.0, -1.5, 1.0], [1.8, 0.9, 1.0], [-1.8, 0.9, 1.0], [0., 0.85, 1.4],
                                                                    [0.0, 1.6, 0.8], [-0.0, -0.5, 0.8], [0.8, 0.9, 0.6], [-0.8, 0.9, 0.8], [0.0, 0.9, 1.]],
                                                        'target': [[0.0, 2.1, 0.9], [-0.0, -0.8, 0.9], [1.4, 0.9, 0.88], [-1.4, 0.9, 0.88], [0.0, 0.80, 1.],
                                                                   [0.0, 1.3, 0.5], [-0.0, -0.0, 0.6], [0.6, 0.9, 0.4], [-0.6, 0.9, 0.5], [0.0, 0.898, 0.8]]},
                                            'borders':[-0.7, 0.7, 0.5, 1.3, 0.1, 0.1]},
                                'verticalmaze': {'urdf': 'verticalmaze.urdf', 'texture': 'verticalmaze.jpg',
                                            'transform': {'position':[-5.7, -7.55, -1.05], 'orientation':[0.0, 0.0, 0.5*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.0, -1.25, 1.0], [0.0, 1.35, 1.3], [1.7, -1.25, 1.0], [-1.6, -1.25, 1.0], [0.0, 0.05, 2.5]],
                                                        'target': [[-0.0, -1.05, 1.0], [0.0, 0.55, 1.3], [1.4, -0.75, 0.9], [-1.3, -0.75, 0.9], [0.0, 0.15, 2.1]]},
                                            'borders':[-0.7, 0.8, 0.65, 0.65, 0.7, 1.4]},
                                'modularmaze': {'urdf': 'modularmaze.urdf', 'texture': 'verticalmaze.jpg',
                                            'transform': {'position':[-7, 8, 0.0], 'orientation':[0.0, 0.0, 0.0]},
                                            'robot': {'position': [0.0, -0.5, 0.05], 'orientation': [0.0, 0.0, 0.5*np.pi]},
                                            'camera': {'position': [[-0.0, -1.25, 1.0], [0.0, 1.35, 1.3], [1.7, -1.25, 1.0], [-1.6, -1.25, 1.0], [0.0, 0.7, 2.1], [-0.0, -0.3, 0.2]],
                                                        'target': [[-0.0, -1.05, 0.9], [0.0, 0.55, 1.3], [1.4, -0.75, 0.9], [-1.3, -0.75, 0.9], [0.0, 0.71, 1.8], [-0.0, -0.25, 0.199]]},
                                            'borders':[-0.7, 0.8, 0.65, 0.65, 0.7, 1.4]}}
    return ws_dict


def get_robot_dict():
    r_dict =   {'kuka': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_magnetic.urdf', 'position': np.array([0.0, 0.0, -0.041]), 'orientation': [0.0, 0.0, 0*np.pi]},
                             'kuka_push': {'path': '/envs/robots/kuka_magnetic_gripper_sdf/kuka_push.urdf', 'position': np.array([0.0, 0.0, -0.041]), 'orientation': [0.0, 0.0, 0*np.pi]},
                             'pepper' : {'path': '/envs/robots/pepper/pepper.urdf', 'position':  np.array([-0.0, -0.18, -0.721]), 'orientation': [0.0, 0.0, 0*np.pi]},
                             'kuka_gripper': {'path': '/envs/robots/kuka_gripper/kuka_gripper.urdf', 'position': np.array([0.0, 0.0, -0.041]), 'orientation': [0.0, 0.0, 0*np.pi]},
                             'panda1': {'path': '/envs/robots/franka_emika/panda/urdf/panda1.urdf', 'position': np.array([0.0, -0.05, -0.04])},
                             'panda_boxgripper': {'path': '/envs/robots/franka_emika/panda/urdf/panda_cgripper.urdf', 'position': np.array([0.0, -0.05, -0.04])},
                             'panda2': {'path': '/envs/robots/franka_emika/panda_moveit/urdf/panda2.urdf', 'position': np.array([0.0, -0.05, -0.04])},
                             'panda': {'path': '/envs/robots/franka_emika/panda_bullet/panda.urdf', 'position': np.array([0.0, -0.05, -0.04])},
                             'jaco': {'path': '/envs/robots/jaco_arm/jaco/urdf/jaco_robotiq.urdf', 'position': np.array([0.0, 0.0, -0.041])},
                             'jaco_fixed': {'path': '/envs/robots/jaco_arm/jaco/urdf/jaco_robotiq_fixed.urdf', 'position': np.array([0.0, 0.0, -0.041])},
                             'nico': {'path': '/envs/robots/nico/complete.urdf', 'position': np.array([0.0, 0.0, -0.041])},
                             'nico_upper': {'path': '/envs/robots/nico/nico_upper.urdf', 'position': np.array([-0.0, 0.1, -0.475])},
                             'nico_upper_rh6d': {'path': '/envs/robots/nico/nico_upper_rh6d.urdf', 'position': np.array([-0.0, 0.1, -0.475])},
                             'reachy': {'path': '/envs/robots/pollen/reachy/urdf/reachy.urdf', 'position': np.array([0.0, 0.0, 0.32]), 'orientation': [0.0, 0.0, 0.0]},
                             'leachy': {'path': '/envs/robots/pollen/reachy/urdf/leachy.urdf', 'position': np.array([0.0, 0.0, 0.32]), 'orientation': [0.0, 0.0, 0.0]},
                             'reachy_and_leachy': {'path': '/envs/robots/pollen/reachy/urdf/reachy_and_leachy.urdf', 'position': np.array([0.0, 0.0, 0.32]), 'orientation': [0.0, 0.0, 0.0]},
                             'gummi': {'path': '/envs/robots/gummi_arm/urdf/gummi.urdf', 'position': np.array([0.0, 0.0, 0.021]), 'orientation': [0.0, 0.0, 0.5*np.pi]},
                             'gummi_fixed': {'path': '/envs/robots/gummi_arm/urdf/gummi_fixed.urdf', 'position': np.array([-0.1, 0.0, 0.021]), 'orientation': [0.0, 0.0, 0.5*np.pi]},
                             'ur3': {'path': '/envs/robots/universal_robots/urdf/ur3.urdf', 'position': np.array([0.0, -0.02, -0.041]), 'orientation': [0.0, 0.0, 0.0]},
                             'ur5': {'path': '/envs/robots/universal_robots/urdf/ur5.urdf', 'position': np.array([0.0, -0.03, -0.041]), 'orientation': [0.0, 0.0, 0.0]},
                             'ur10': {'path': '/envs/robots/universal_robots/urdf/ur10.urdf', 'position': np.array([0.0, -0.04, -0.041]), 'orientation': [0.0, 0.0, 0.0]},
                             'yumi': {'path': '/envs/robots/abb/yumi/urdf/yumi.urdf', 'position': np.array([0.0, 0.15, -0.042]), 'orientation': [0.0, 0.0, 0.0]},
                             'icub': {'path': '/envs/robots/iCub/robots/iCubGenova04_plus/model.urdf', 'position': np.array([0.0, 0.15, -0.042]), 'orientation': [0.0, 0.0, 0.0]},
                             'human': {'path': '/envs/robots/real_hands/humanoid_with_hands.urdf', 'position': np.array([0.0, 2, 0.45]), 'orientation': [0.0, 0.0, 0.0]},
                             'tiago': {'path': '/envs/robots/tiago/tiago_pal_gripper.urdf', 'position': np.array([0.0, -0.4, -0.2]), 'orientation': [0.0, 0.0, 0*np.pi]},
                             'tiago_simple': {'path': '/envs/robots/tiago/tiago_simple.urdf', 'position': np.array([0.0, -0.4, -0.2]), 'orientation': [0.0, 0.0, 0*np.pi]},
                             'tiago_dual': {'path': '/envs/robots/tiago_dualhand/tiago_dual_hey5.urdf', 'position': np.array([0.0, -0.4, -0.2]),
                                      'orientation': [0.0, 0.0, 0 * np.pi]},
                             'tiago4': {'path': '/envs/robots/tiago/tiago4/tiago_dual_hand.urdf',
                               'position': np.array([0.0, -0.4, -0.2]),'orientation': [0.0, 0.0, 0 * np.pi]},
                              'hsr': {'path': '/envs/robots/hsr/hsrb4s.urdf',
                                       'position': np.array([0.0, -0.15, -0.4]), 'orientation': [0.0, 0.0, 0 * np.pi]},
                             }
    return r_dict
