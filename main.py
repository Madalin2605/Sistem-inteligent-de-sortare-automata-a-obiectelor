import time
import numpy as np
import cv2
import torch
import sys

from Interfaces.robot_interface import Robot
from Interfaces.camera_interface import camera_interface as ci
from detection import detect, compute_positions


def play(result_list):
    for result in result_list:
        print("Inainte: ", result)
        center_depth = result[2]
        position = compute_positions(photo_position, result, center_depth, offset_x_cam, offset_y_cam, offset_z_cam)
        print("Dupa: ", position)
        robot.move_to_coords(position, acceleration=50, velocity=50)
        robot.gripper_open()
        time.sleep(1)
        robot.move_to_coords(photo_position, acceleration=50, velocity=50)
        robot.move_to_coords(drop_position, acceleration=50, velocity=50)
        robot.gripper_close()
        robot.move_to_coords(photo_position, acceleration=50, velocity=50)


if __name__ == '__main__':
    offset_x_cam = -27 - 77  # -27 -27 -27 #to add -27 #-90  # -27.5 #26.6  # -35.079 #27.5
    offset_y_cam = 60  # 10 +10 +10 #to add +10 #-40  # -65 #-11  # 22.573 #65s
    offset_z_cam = 90 - 30

    robot = Robot('UR5')
    robot.connect()
    print(robot.get_coords())

    photo_position = [-98.3790232447588, -999.1495292161882, 489.9517565797424, 0.22672869661694256, 179.93372315084773,
                      0.026041965345948076]
    drop_position = [820.5789947880054, -809.7703931849688, 50, 0.22833521042241903, 179.93389429866062,
                     0.02624508265789329]
    robot.move_to_coords(photo_position, acceleration=150, velocity=150)

    result_list = detect()

    play(result_list)


