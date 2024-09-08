import logging
from abc import ABC, abstractmethod
from Interfaces.robot_interface.robots.CRX10.fanucpy.robot import Robot as fanucRobot
from Interfaces.robot_interface.robots.UR5.urx.urrobot import URRobot as urx
from Interfaces.robot_interface.robots.UR5.urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import pandas as pd
import math


class Robot(ABC):
    """
    Abstract base class for all robots. It defines the interface that any specific robot implementation should follow.
    """

    def __new__(cls, data):
        """
        Creates an instance of a specific robot type based on the data provided.

        Args:
            data (str): The type of the robot.

        Returns:
            Robot: An instance of a specific robot subclass.
        """
        if cls is Robot:
            if data == 'CRX10':
                return super(Robot, cls).__new__(CRX10)
            elif data == 'UR5':
                return super(Robot, cls).__new__(UR5)
        return super(Robot, cls).__new__(cls)

    def __init__(self, data):
        pass

    @abstractmethod
    def move_to_coords(self, coords, velocity, acceleration, cnt_val=0, linear=0):
        """
        Moves the robot to the specified coordinates.

        Args:
            coords (list): The coordinates to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        pass

    @abstractmethod
    def move_joints(self, joints, velocity, acceleration, cnt_val=0, linear=0):
        """
        Moves the robot's joints to the specified positions.

        Args:
            joints (list): The joint positions to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        pass

    @abstractmethod
    def move_xs(self, command, pose_list, acceleration, velocity, radius):
        """
        Moves the robot based on a list of poses or joint positions.

        Args:
            command (str): The movement command ('movel' or 'movej').
            pose_list (list): List of poses or joint positions.
            acceleration (float): The acceleration of the robot movement.
            velocity (float): The velocity at which to move the robot.
            radius (float): The blending radius.
        """
        pass

    @abstractmethod
    def get_coords(self):
        """
        Returns the current coordinates of the robot.
        """
        pass

    @abstractmethod
    def get_joints(self):
        """
        Returns the current joint positions of the robot.
        """
        pass

    @abstractmethod
    def freedrive(self, enable=True, timeout=5000):
        """
        Enables or disables the freedrive mode.
        
        Args:
            enable (bool): Enable or disable freedrive mode.
            timeout (int): Timeout for the freedrive mode in milliseconds.
        """
        pass

    @abstractmethod
    def connect(self):
        """
        Connects to the robot.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Disconnects from the robot.
        """
        pass

    @abstractmethod
    def gripper_action(self, action):
        """
        Performs an action with the gripper.

        Args:
            action (str): The action to perform.
        """
        pass

    @abstractmethod
    def gripper_open(self):
        """
        Opens the gripper.
        """
        pass

    @abstractmethod
    def gripper_close(self):
        """
        Closes the gripper.
        """
        pass
    
    @abstractmethod
    def convertCameraToRobot(self, x, y, z, angle):
        """
        Converts camera coordinates to robot coordinates.

        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            z (float): Z coordinate.
            angle (float): Angle in degrees.
        """
        pass


class CRX10(Robot):
    """
    CRX10 robot implementation.
    """

    def __init__(self, data: str):
        """
        Initializes the CRX10 robot with the given data.
        
        Args:
            data (str): The data containing the type of the robot.
        """
        super().__init__(data)
        self._robot = fanucRobot(
            robot_model="Fanuc",
            host="192.168.0.111",
            port=18735,
            ee_DO_type="RDO",
            ee_DO_num=7
        )

    def move_to_coords(self, coords: list, velocity: float, acceleration: float, cnt_val: int = 50, linear: int = 0):
        """
        Moves the CRX10 robot to the specified coordinates.
        
        Args:
            coords (list): The coordinates to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 50.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        self._robot.move("pose", vals=coords, velocity=velocity, acceleration=acceleration, cnt_val=cnt_val, linear=linear)

    def move_joints(self, coords: list, velocity: float, acceleration: float, cnt_val: int = 50, linear: int = 0):
        """
        Moves the joints of the CRX10 robot to the specified positions.
        
        Args:
            coords (list): The joint positions to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 50.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        self._robot.move("joint", vals=coords, velocity=velocity, acceleration=acceleration, cnt_val=cnt_val, linear=linear)

    def get_coords(self) -> list:
        """
        Returns the current coordinates of the CRX10 robot.
        
        Returns:
            list: The current coordinates of the robot.
        """
        return self._robot.get_curpos()

    def get_joints(self) -> list:
        """
        Returns the current joint positions of the CRX10 robot.
        
        Returns:
            list: The current joint positions of the robot.
        """
        return self._robot.get_curjpos()

    def connect(self):
        """
        Connects to the CRX10 robot.
        """
        self._robot.connect()

    def disconnect(self):
        """
        Disconnects from the CRX10 robot.
        """
        self._robot.disconnect()

    def gripper_action(self, action: str):
        """
        Performs an action with the gripper.
        
        Args:
            action (str): The action to perform with the gripper.
        """
        pass

    def gripper_open(self):
        """
        Opens the gripper.
        """
        pass

    def gripper_close(self):
        """
        Closes the gripper.
        """
        pass
    
    def freedrive(self, enable: bool = True, timeout: int = 60):
        """
        Enables or disables the freedrive mode.
        
        Args:
            enable (bool, optional): Enable or disable freedrive mode. Defaults to True.
            timeout (int, optional): Timeout for the freedrive mode in seconds. Defaults to 60.
        """
        pass

    def convertCameraToRobot(self, xC: float, yC: float, angle: float) -> tuple:
        """
        Converts camera coordinates to robot coordinates.
        
        Args:
            xC (float): X coordinate from the camera.
            yC (float): Y coordinate from the camera.
            angle (float): Angle in degrees.
        
        Returns:
            tuple: Converted robot coordinates and angle.
        """
        a = -0.006085198660343979
        b = 1.8581266655422601
        c = 1.9186277618485514
        d = -0.039582054309345534
        tx = 324.3707478254586
        ty = -708.2134371732161

        xR = a * xC + b * yC + tx
        yR = c * xC + d * yC + ty
                
        return xR, yR, 270 - angle

class UR5(Robot):
    """
    UR5 robot implementation.
    """
    
    def __init__(self, data: str):
        """
        Initializes the UR5 robot with the given data.
        
        Args:
            data (str): The data containing the type of the robot.
        """
        super().__init__(data)
        logging.basicConfig(level=logging.WARN)
        self._robot = urx('192.168.1.132')
        self._robot.set_payload(1.2, (0, 0, 0))
        self._robot.set_tcp((0, 0, 0.151, 0, 0, 0))
        self._robot.stopj()

    def move_to_coords(self, coords: list, velocity: float, acceleration: float, cnt_val: int = 0, linear: int = 0):
        """
        Moves the UR5 robot to the specified coordinates.
        
        Args:
            coords (list): The coordinates to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        velocity /= 100
        acceleration /= 100
        coords = [coords[0] / 1000, coords[1] / 1000, coords[2] / 1000, math.radians(coords[3]), math.radians(coords[4]), math.radians(coords[5])]
        self._robot.movel(tpose=coords, vel=velocity, acc=acceleration)

    def move_joints(self, joints: list, velocity: float, acceleration: float, cnt_val: int = 0, linear: int = 0):
        """
        Moves the joints of the UR5 robot to the specified positions.
        
        Args:
            joints (list): The joint positions to move the robot to.
            velocity (float): The velocity at which to move the robot.
            acceleration (float): The acceleration of the robot movement.
            cnt_val (int, optional): The continuous value for stopping. Defaults to 0.
            linear (int, optional): Whether the movement is linear or not. Defaults to 0.
        """
        velocity /= 100
        acceleration /= 100
        joints = [math.radians(joints[0]), math.radians(joints[1]), math.radians(joints[2]), math.radians(joints[3]), math.radians(joints[4]), math.radians(joints[5])]
        self._robot.movej(joints=joints, vel=velocity, acc=acceleration)

    def move_xs(self, command: str, pose_list: list, acceleration: float, velocity: float, radius: float):
        """
        Moves the robot based on a list of poses or joint positions.
        
        Args:
            command (str): The movement command ('movel' or 'movej').
            pose_list (list): List of poses or joint positions.
            acceleration (float): The acceleration of the robot movement.
            velocity (float): The velocity at which to move the robot.
            radius (float): The blending radius.
        """
        velocity /= 100
        acceleration /= 100
        radius /= 100
        newposes = []

        if command == 'movel':
            for coords in pose_list:
                coords = [coords[0] / 1000, coords[1] / 1000, coords[2] / 1000, math.radians(coords[3]), math.radians(coords[4]), math.radians(coords[5])]
                newposes.append(coords)
        elif command == 'movej':
            for joints in pose_list:
                joints = [math.radians(joints[0]), math.radians(joints[1]), math.radians(joints[2]), math.radians(joints[3]), math.radians(joints[4]), math.radians(joints[5])]
                newposes.append(joints)

        self._robot.movexs(command=command, pose_list=newposes, acc=acceleration, vel=velocity, radius=radius)

    def get_coords(self) -> list:
        """
        Returns the current coordinates of the UR5 robot.
        
        Returns:
            list: The current coordinates of the robot.
        """
        coords = self._robot.getl()
        return [coords[0] * 1000, coords[1] * 1000, coords[2] * 1000, math.degrees(coords[3]), math.degrees(coords[4]), math.degrees(coords[5])]

    def get_joints(self) -> list:
        """
        Returns the current joint positions of the UR5 robot.
        
        Returns:
            list: The current joint positions of the robot.
        """
        coords = self._robot.getj()
        return [math.degrees(coords[0]), math.degrees(coords[1]), math.degrees(coords[2]), math.degrees(coords[3]), math.degrees(coords[4]), math.degrees(coords[5])]

    def freedrive(self, enable: bool = True, timeout: int = 60):
        """
        Enables or disables the freedrive mode.
        
        Args:
            enable (bool, optional): Enable or disable freedrive mode. Defaults to True.
            timeout (int, optional): Timeout for the freedrive mode in seconds. Defaults to 60.
        """
        self._robot.set_freedrive(timeout if enable else False)

    def gripper_action(self, action: str):
        """
        Performs an action with the gripper.
        
        Args:
            action (str): The action to perform with the gripper.
        """
        self.gripper.gripper_action(action)

    def gripper_close(self):
        """
        Closes the gripper.
        """
        self._robot.set_digital_out(0, False)

    def gripper_open(self):
        """
        Opens the gripper.
        """
        self._robot.set_digital_out(0, True)

    def connect(self):
        """
        Connects to the UR5 robot.
        """
        pass

    def disconnect(self):
        """
        Disconnects from the UR5 robot.
        """
        self._robot.close()
        
    def convertCameraToRobot(self, xC: float, yC: float, angle: float) -> tuple:
        """
        Converts camera coordinates to robot coordinates.
        
        Args:
            xC (float): X coordinate from the camera.
            yC (float): Y coordinate from the camera.
            angle (float): Angle in degrees.
        
        Returns:
            tuple: Converted robot coordinates and angle.
        """
        a = -0.05324761290130534
        b = 1.9175380730856597
        c = 1.9067066688332064
        d = 0.025175112481834398
        tx = -1061.9398393492365
        ty = -797.2170478784193

        xR = a * xC + b * yC + tx
        yR = c * xC + d * yC + ty
            
        return xR, yR, 180 - angle