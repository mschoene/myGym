from collections import defaultdict
import os, inspect
import atexit

import xml.etree.ElementTree as ET
from shutil import copyfile
import pybullet
import random
import glob
import numpy as np
import sys, shutil
from datetime import datetime
import pkg_resources
from lxml import etree


currentdir = pkg_resources.resource_filename("myGym", "envs")


class EnvObject:
    """
    Env object class for dynamic object in PyBullet environment

    Parameters:
        :param urdf_path: (string) Path to model of object
        :param position: (list) Position of object in the coordinate frame of the environment ([x,y,z])
        :param orientation: (list) Orientation of object in the coordinate frame of the environment (quaternion [x,y,z,w])
        :param fixed: (bool) Whether the object should have fixed position and orientation
        :param pybullet_client: Which pybullet client the environment should refere to in case of parallel existence of multiple instances of this environment
    """
    def __init__(self, urdf_path, position=[0, 0, 0],
                 orientation=[0, 0, 0, 0], fixed=False, com_pos =[0,0,0],
                 pybullet_client=None):
        self.p = pybullet_client
        self.urdf_path = urdf_path
        self.init_position = position
        self.init_orientation = orientation
        self.init_com_pos = com_pos #center of mass shift
        self.fixed = fixed
        self.name = os.path.splitext(os.path.basename(self.urdf_path))[0]
        self.virtual = True if "virtual" in self.name else False
        self.object_ldamping = 1
        self.object_adamping = 1
        self.object_lfriction = 100
        self.object_rfriction = 100
        self.object_mass = self.read_mass_from_urdf()
        self.object_stiffness = 1
        if not self.virtual:
            if( com_pos != [0,0,0]):
                #this will create a new temporary urdf file with a shifted com.. beautiful solution, I know...
                self.update_urdf_com() # Sets URDF path to temporary one with shifted COM
            self.uid = self.load()

            #self.update_collision_and_visual_shapes()
            self.bounding_box = self.get_bounding_box()
            self.centroid = self.get_centroid()

        else:
            self.uid = 1
            self.bounding_box = None
            self.centroid = None
        self.debug_line_ids = []
        self.cuboid_dimensions = None

    def __eq__(self, other):
        if isinstance(other, EnvObject):
            return self.name == other.name and self.uid == other.uid
        return False
    
    def __hash__(self):
        return hash((self.name, self.uid))

    def __repr__(self):
        return self.name

    def read_mass_from_urdf(self):
        # Parse the URDF file to find the mass
        tree = etree.parse(self.urdf_path)
        root = tree.getroot()

        # Find the <inertial> tag
        inertial_tag = root.find(".//inertial")
        if inertial_tag is not None:
            # Find the <mass> tag within <inertial>
            mass_tag = inertial_tag.find("mass")
            if mass_tag is not None:
                # Extract the mass value from the 'value' attribute
                mass_value = mass_tag.attrib.get("value")
                if mass_value is not None:
                    return float(mass_value)

        # Return a default mass if not found
        return 1.0  # Default mass
    
    def set_color(self, color):
        """
        Set desired color of object

        Parameters:
            :param color: (list) RGB color to set
        """
        self.color_rgba = color
        if not self.virtual:
            self.p.changeVisualShape(self.uid, -1, rgbaColor=self.color_rgba)

    def get_color_rgba(self):
        """
        Get object's color

        Returns:
            :return self.color_rgba: (list) RGB color of object
        """
        return self.color_rgba

    def set_random_texture(self, obj_id, patternPath="dtd/images"):
        """
        Apply texture to object
        
        Parameters:
            :param obj_id: (int) ID of object
            :param patternPath: (string) relative path to *.jpg (recursive) with textures
        """
        pp = os.path.abspath(os.path.join(currentdir, str(patternPath)))
        texture_paths = glob.glob(os.path.join(pp, '**', '*.jpg'), recursive=True)
        texture_paths += glob.glob(os.path.join(pp, '**', '*.png'), recursive=True)
        random_texture_path = random.choice(texture_paths)
        if not self.virtual:
            texture_id = self.p.loadTexture(random_texture_path)
            try:
                self.p.changeVisualShape(obj_id, -1, rgbaColor=[1, 1, 1, 1])
                self.p.changeVisualShape(obj_id, -1, textureUniqueId=texture_id)
            except:
                print("Failed to apply texture to obj ID:" + str(obj_id) + " from path=" + str(pp))

    def set_init_position(self, position):
        """
        Set object's initial position in world coordinates

        Parameters:
            :param position: (list) Desired initial position ([x,y,z])
        """
        self.init_position = position

    def set_init_orientation(self, orientation):
        """
        Set object's initial orientation in world coordinates

        Parameters:
            :param orientation: (list) Desired initial orientation (quaternion [x,y,z,w])
        """
        self.init_orientation = orientation

    def get_position(self):
        """
        Get object's position in world coordinates

        Returns:
            :return position: (array) Position ([x,y,z]) of object
        """
        return self.get_position_and_orientation()[0]

    def get_orientation(self):
        """
        Get object's orientation in quaterion in world coordinates

        Returns:
            :return orientation: (array) Orientation of object (quaternion [x,y,z,w])
        """
        return self.get_position_and_orientation()[1]

    def get_orientation_euler(self):
        """
        Get object's orientation in Euler angles in world coordinates

        Returns:
            :return orientation: (array) Orientation of object in Euler angles (degrees)
        """
        return self.p.getEulerFromQuaternion(self.get_position_and_orientation()[1])

    def get_position_and_orientation(self):
        """
        Get object's position and orientation in quaterion in world coordinates

        Returns:
            :return orientation: (array) Position ([x,y,z]) Orientation of object (quaternion [x,y,z,w])
        """
        if not self.virtual:
            return self.p.getBasePositionAndOrientation(self.uid)
        else:
            return (self.init_position, self.init_orientation)

    def set_position(self, position):
        """
        Set object's position in world coordinates

        Parameters:
            :param position: (array) Position ([x,y,z]) of object
        """
        if not self.virtual:
            self.p.resetBasePositionAndOrientation(self.uid, position, self.get_orientation())

    def set_orientation(self, orientation):
        """
        Set object's orientation in quaterion in world coordinates

        Parameters:
            :param orientation: (array) Orientation of object in quaternion ([x,y,z,w])
        """
        if not self.virtual:
            self.p.resetBasePositionAndOrientation(self.uid, self.get_position(), orientation)

    def move(self, movement):
        """
        Move object to new position relative to current position

        Parameters:
            :param movement: (array) By how much to move from current position ([x,y,z])
        """
        current_position = self.get_position()
        self.set_position(np.add(current_position, movement))

    def rotate_euler(self, rotation):
        """
        Rotate object to new orientation relative to current orientation in Euler angles

        Parameters:
            :param rotation: (array) By how much to rotate from current rotation (degrees)
        """
        current_euler= self.p.getEulerFromQuaternion(self.get_orientation())
        next_euler = np.add(current_euler, rotation)
        self.set_orientation(self.p.getQuaternionFromEuler(next_euler))

    def get_file_path(self):
        """
        Get path to object's model

        Returns:
            :return self.file_path: (string) Path to object's model
        """
        return self.file_path

    def set_texture(self, texture):
        pass


    def update_urdf_com(self):
        # Get the directory of the original URDF file
        urdf_dir = os.path.dirname(self.urdf_path)

        # Create a temporary file path for the modified URDF file
        tmp_urdf_path = os.path.join(urdf_dir, "obj_temp.urdf")

        # Copy the original URDF file to the temporary file
        shutil.copyfile(self.urdf_path, tmp_urdf_path)

        # Parse the copied URDF file to identify collision shapes associated with the base link
        tree = etree.parse(tmp_urdf_path)
        root = tree.getroot()
        link = root.find(".//link[@name='baseLink']")
        inertial = link.find("inertial")
        origin = inertial.find("origin")
        if origin is not None:
            xyz = origin.attrib.get("xyz", "0 0 0")
            xyz = [float(x) for x in xyz.split()]
            new_xyz = [xyz[i] - self.init_com_pos[i] for i in range(3)]
            origin.attrib["xyz"] = " ".join(str(x) for x in new_xyz)

        # Write the modified URDF to the temporary file
        tree.write(tmp_urdf_path)

        # Load the object using the modified URDF path
        self.urdf_path = tmp_urdf_path

    def load(self):
        """
        Load object from it's model to the scene

        Returns:
            :return self.uid: (int) ID of loaded object
        """
        self.uid = self.p.loadURDF(self.urdf_path, self.init_position, self.init_orientation, useFixedBase=self.fixed,  flags=self.p.URDF_USE_SELF_COLLISION)
        if self.fixed:
            self.p.changeDynamics(self.uid, 0, collisionMargin=0., contactProcessingThreshold=0.0, ccdSweptSphereRadius=0)
        else:
            self.p.changeDynamics(self.uid, 0, collisionMargin=0., contactProcessingThreshold=0.0, 
                                ccdSweptSphereRadius=0, linearDamping=self.object_ldamping, 
                                angularDamping=self.object_adamping, lateralFriction=self.object_lfriction,
                                rollingFriction=self.object_rfriction, mass=self.object_mass)
        
        return self.uid
    
    def get_bounding_box(self):
        """
        Get 3D axis-aligned bounding box of object

        Returns:
            :return bounding_box: (list) Center and 8 coordinates of vertices of the bounding box
        """
        bounding_box = []
        diag = self.p.getAABB(self.uid)
        bounding_box.append(diag[0])
        bounding_box.append((diag[0][0], diag[1][1], diag[0][2]))
        bounding_box.append((diag[1][0], diag[0][1], diag[0][2]))
        bounding_box.append((diag[1][0], diag[1][1], diag[0][2]))
        bounding_box.append(diag[1])
        bounding_box.append((diag[0][0], diag[0][1], diag[1][2]))
        bounding_box.append((diag[1][0], diag[0][1], diag[1][2]))
        bounding_box.append((diag[0][0], diag[1][1], diag[1][2]))
        bounding_box.append(list(np.divide(np.add(diag[0], diag[1]), 2)))
        return bounding_box

    def get_centroid(self):
        """
        Get position of object's centroid

        Returns:
            :return centeroid: (array) Position of object's centroid (center of mass) ([x,y,z])
        """
        return self.p.getBasePositionAndOrientation(self.uid)[0]

    def draw_bounding_box(self):
        """
        Draw object's 3D bounding box in the scene GUI
        """
        diagonal_points = self.p.getAABB(self.uid)
        lines = self.get_lines(diagonal_points)
        if self.debug_line_ids:
            for i in range(len(lines)):
                self.p.addUserDebugLine(lines[i][0], lines[i][1], replaceItemUniqueId = self.debug_line_ids[i], lineColorRGB=(0.31, 0.78, 0.47), lineWidth = 2)
        else:
            for i in range(len(lines)):
                self.debug_line_ids.append(self.p.addUserDebugLine(lines[i][0], lines[i][1], lineColorRGB=(0.31, 0.78, 0.47), lineWidth = 2))

    def get_lines(self, diag):
        """
        Get lines connecting vertices of 3D bounding box

        Parameters:
            :param diag: (list) Diagonal points of 3D bounding box
        Returns:
            :return lines: (list) List of lines connecting vertices of 3D bounding box
        """
        lines = []

        lines.append((diag[0], (diag[0][0], diag[0][1], diag[1][2])))
        lines.append((diag[0], (diag[0][0], diag[1][1], diag[0][2])))
        lines.append((diag[0], (diag[1][0], diag[0][1], diag[0][2])))
        lines.append((diag[1], (diag[1][0], diag[1][1], diag[0][2])))
        lines.append((diag[1], (diag[1][0], diag[0][1], diag[1][2])))
        lines.append((diag[1], (diag[0][0], diag[1][1], diag[1][2])))

        lines.append(((diag[0][0], diag[0][1], diag[1][2]), (diag[1][0], diag[0][1], diag[1][2])))
        lines.append(((diag[0][0], diag[0][1], diag[1][2]), (diag[0][0], diag[1][1], diag[1][2])))
        lines.append(((diag[1][0], diag[1][1], diag[0][2]), (diag[1][0], diag[0][1], diag[0][2])))
        lines.append(((diag[1][0], diag[1][1], diag[0][2]), (diag[0][0], diag[1][1], diag[0][2])))
        lines.append(((diag[0][0], diag[1][1], diag[1][2]), (diag[0][0], diag[1][1], diag[0][2])))
        lines.append(((diag[1][0], diag[0][1], diag[0][2]), (diag[1][0], diag[0][1], diag[1][2])))

        return lines

    def get_cuboid_dimensions(self):
        """
        Get dimensions of cuboid defined by object's 3D bounding box

        Returns:
            :return self.cuboid_dimension: (list) Dimensions of cuboid
        """
        if self.cuboid_dimensions is None:
            diag = self.p.getAABB(self.uid)
            self.cuboid_dimensions = np.absolute(np.subtract(diag[0], diag[1])).tolist()
        return self.cuboid_dimensions

    def get_name(self):
        """
        Get object's name. Uses data from URDF.

        Returns:
            :return self.name: (string) Object's name
        """
        return self.name

    def get_uid(self):
        """
        Get object's unique ID

        Returns:
            :return self.uid: Object's unique ID
        """
        return self.uid

    @staticmethod
    def get_random_object_position(boarders):
        """
        Generate random position in defined volume

        Parameters:
            :param boarders: (list) Volume, where position may be generated ([x,x,y,y,z,z])
        Returns:
            :return pos: (list) Position in specified volume ([x,y,z])
        """
        if any(isinstance(i, list) for i in boarders):
            boarders = boarders[random.randint(0,len(boarders)-1)]
        pos = []
        pos.append(random.uniform(boarders[0], boarders[1])) #x
        pos.append(random.uniform(boarders[2], boarders[3])) #y
        pos.append(random.uniform(boarders[4], boarders[5])) #z
        return pos

    @staticmethod
    def get_random_object_com(boarders):
        """
        Generate random center of mass (com). make sure it is inside the volume

        Parameters:
            :param boarders: (list) xyz coordinate bounds, where position may be generated ([x,x,y,y,z,z])
        Returns:
            :return pos: (list) Position in specified volume ([x,y,z])
        """
        if any(isinstance(i, list) for i in boarders):
            boarders = boarders[random.randint(0,len(boarders)-1)]
        com_pos = []
        com_pos.append(random.uniform(boarders[0], boarders[1])) #x
        com_pos.append(random.uniform(boarders[2], boarders[3])) #y
        com_pos.append(random.uniform(boarders[4], boarders[5])) #z
        return com_pos
    
    @staticmethod
    def get_random_object_orientation():
        """
        Generate random orientation

        Returns:
            :return orientation: (array) Orientation (quaternion [x,y,z,w])
        """
        angleX = 3.14 * 0.5 + 3.14 * random.random()
        angleY = 3.14 * 0.5 + 3.14 * random.random()
        angleZ = 3.14 * 0.5 + 3.14 * random.random()
        return pybullet.getQuaternionFromEuler([angleX, angleY, angleZ])

    @staticmethod
    def get_random_z_rotation():
        angleX = np.pi
        angleY = 0
        angleZ = np.pi + (np.pi * random.random())
        return pybullet.getQuaternionFromEuler([angleX, angleY, angleZ])
