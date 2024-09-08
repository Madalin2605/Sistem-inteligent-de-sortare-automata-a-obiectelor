import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt


def init_camera(resolution_width=640, resolution_height=480, fps=30):
    """
    Initializes the RealSense camera with the specified resolution and frame rate.

    Args:
        resolution_width (int, optional): The width of the resolution. Defaults to 640.
        resolution_height (int, optional): The height of the resolution. Defaults to 480.
        fps (int, optional): The frame rate. Defaults to 30.

    Returns:
        rs.profile: The profile object for the RealSense camera.
        rs.pipeline: The pipeline object for the RealSense camera.
        rs.config: The configuration object for the RealSense camera.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, fps)

    import yaml
    try:
        profile = pipeline.start(config)
    except:
        print("Camera is already streaming")

    with open('UI/web-hmi/src/assets/data_2.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    data['camera'] = ''

    if profile is None:
        data['camera'] = 'False'
        with open('UI/web-hmi/src/assets/data_2.yaml', 'w') as file:
            yaml.dump(data, file)
    else:
        data['camera'] = 'True'
        with open('UI/web-hmi/src/assets/data_2.yaml', 'w') as file:    
            yaml.dump(data, file)

    return profile, pipeline, config

import time  

def stream_camera(pipeline, config, show_frame=False, color_depth=False):
    """
    Streams the RealSense camera with the specified configuration and saves the frames.
    
    Args:
        pipeline (rs.pipeline): The pipeline object for the RealSense camera.
        config (rs.config): The configuration object for the RealSense camera.
        show_frame (bool, optional): Whether to display the frames using OpenCV. Defaults to False.
        color_depth (bool, optional): Whether to return and save both the color and depth frames. Defaults to False.
        
    Returns:
        np.ndarray: The color image.
        np.ndarray: The colorized depth image.
    """

    colorizer = rs.colorizer()
    
    pipeline.start(config)
    
    try:
        while True:
            # Wait for a new set of frames
            frames = pipeline.wait_for_frames()

            # Get the depth and color frames from frameset
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_data = np.asanyarray(color_frame.get_data())
            align = rs.align(rs.stream.color)
            frames = align.process(frames)
            aligned_depth_frame = frames.get_depth_frame()
            colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
            
            # Save images to the specified directory, overwriting the previous images
            cv2.imwrite('static/assets/rgb.png', color_data)
            cv2.imwrite('static/assets/depth.png', colorized_depth)

            # Display images using OpenCV if required
            if show_frame:
                cv2.imshow("Color Frame", color_data)
                cv2.imshow("Depth Frame", colorized_depth)
            
            # Pause for 1 second before capturing the next frame
            time.sleep(1)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        # Ensure that the pipeline stops when the loop ends
        pipeline.stop()
        cv2.destroyAllWindows()
        


def capture_RGB_image(pipeline, config):
    """
    Captures an RGB image from a RealSense camera.

    Args:
        pipeline (rs.pipeline): The pipeline object for the RealSense camera.
        config (rs.config): The configuration object for the RealSense camera.

    Returns:
        np.ndarray: The color image.
    """
    #profile = pipeline.start(config)
    for x in range(5):
        pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    # pipeline.stop()
    return color_image


def capture_aligned_images(pipeline, config, profile, hole_filling = False, show_frames=False):
    """
    Captures and aligns images from a RealSense camera.

    Args:
        pipeline (rs.pipeline): The pipeline object for the RealSense camera.
        config (rs.config): The configuration object for the RealSense camera.
        profile (rs.profile): The profile object for the Realsense camera.
        hole_filling (bool, optional): Whether to apply hole filling to the depth frame or not. Defaults to False. It's better used for close-up objects or even surfaces.

    Returns:
        np.ndarray: The aligned color image.
        rs.frame: The aligned depth frame.
        float: The depth scale of the camera.
        rs.intrinsics: The depth intrinsics of the camera.
        rs.intrinsics: The color intrinsics of the camera.
    """

    # try:
    #     profile = pipeline.start(config)
    # except:
    #     print("Camera is already streaming")
    for x in range(5):
        pipeline.wait_for_frames()

    # Get the depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_sensor.set_option(rs.option.emitter_enabled, 1)

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames().as_frameset()

    print(f'Frames este de tipul: {type(frames)}, size este de {frames.size()}')  
    colorizer = rs.colorizer()

    # Initialize post-processing filters
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 4)
    depth_to_disparity = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    disparity_to_depth = rs.disparity_transform(False)
    hole_filling_proc = rs.hole_filling_filter(2)

    # Apply post-processing filters to the aligned depth frame
    decimated_frames = decimation.process(frames).as_frameset()

    # Check if decimation was successful
    if not decimated_frames.get_depth_frame():
        raise RuntimeError("Depth frame is None after decimation")


    # Align depth and color frames
    align = rs.align(rs.stream.color) # if using rs.stream.color, use color_intrinsics to calculate the depth
    aligned_frames = align.process(decimated_frames)


    # Update color and depth frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()

    if show_frames:
        # Create subplots
        num_plots = 5 if hole_filling else 4
        fig, axs = plt.subplots(2, (num_plots + 1) // 2, figsize=(20, 10))
        axs = axs.flatten()

        # Show original aligned depth frame
        axs[0].imshow(np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data()))
        axs[0].set_title("Original Aligned Depth Frame")

    # Check if aligned depth frame is valid
    if not aligned_depth_frame:
        raise RuntimeError("Aligned depth frame is None")
    


    # Apply disparity transform to the depth frame
    filtered_depth_frame = depth_to_disparity.process(aligned_depth_frame)
    if show_frames:
        axs[1].imshow(np.asanyarray(colorizer.colorize(filtered_depth_frame).get_data()))
        axs[1].set_title("After Disparity Transform")

    # Check if disparity transform was successful
    if not filtered_depth_frame:
        raise RuntimeError("Depth frame is None after disparity transform")
    
    # Apply spatial filter to the depth frame
    filtered_depth_frame = spatial.process(filtered_depth_frame)

    if show_frames:
        axs[2].imshow(np.asanyarray(colorizer.colorize(filtered_depth_frame).get_data()))
        axs[2].set_title("After Spatial Filter")

    # Check if spatial filter was successful
    if not filtered_depth_frame:
        raise RuntimeError("Depth frame is None after spatial filter")

    # Apply disparity to depth transform to the depth frame
    filtered_depth_frame = disparity_to_depth.process(filtered_depth_frame).as_depth_frame()
    if show_frames:
        axs[3].imshow(np.asanyarray(colorizer.colorize(filtered_depth_frame).get_data()))
        axs[3].set_title("After Disparity to Depth Transform")

    # Check if disparity to depth transform was successful
    if not filtered_depth_frame:
        raise RuntimeError("Depth frame is None after disparity to depth transform")

    if hole_filling:
        filtered_depth_frame = hole_filling_proc.process(filtered_depth_frame).as_depth_frame()
        if show_frames:
            axs[4].imshow(np.asanyarray(colorizer.colorize(filtered_depth_frame).get_data()))
            axs[4].set_title("After Hole Filling")
    
    if show_frames:
        plt.show()

    # Get depth intrinsics for converting pixel coordinates to real-world coordinates
    depth_intrinsics = filtered_depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrinsics = aligned_color_frame.profile.as_video_stream_profile().intrinsics


    color_image = np.asanyarray(aligned_color_frame.get_data())

    return color_image, filtered_depth_frame, depth_scale, depth_intrinsics, color_intrinsics


def stop_camera(pipeline):
    """
    Stops the RealSense camera pipeline.

    Args:
        pipeline (rs.pipeline): The pipeline object for the RealSense camera.
    """
    pipeline.stop()


# def transform_coordinates(result_list, depth_intrinsics, depth_scale=None):
#     """
#     Pentru sortarea de jucarii (!!!!! DA IN mm !!!!!)
#     """

#     transformed_result = {}

#     for label, coordinates_list in result_list.items():
#         transformed_coordinates_list = []
#         print("Coordinates list:", coordinates_list)

#         for coordinates in coordinates_list:
#             print("Coordinates:", coordinates)
#             x, y, depth_value = coordinates
#             depth_point = pixel_to_mm_coordinates([x, y], depth_value, depth_intrinsics)

#             # Append the transformed coordinates to the list
#             transformed_coordinates_list.append(depth_point)

#         # Update the result dictionary with transformed coordinates
#         transformed_result[label] = transformed_coordinates_list

#     return transformed_result


def pixel_to_mm_coordinates(coords_list, depth_frame, depth_intrinsics):
    """
    Converts a list of pixel coordinates to real-world coordinates using depth information.

    Args:
        coords_list (list): A list of [x, y] pixel coordinates.
        depth_frame (rs.frame): The depth frame from the RealSense camera.
        depth_intrinsics (rs.intrinsics): The depth intrinsics of the camera.

    Returns:
        list: A list of real-world coordinates in millimeters corresponding to the input pixel coordinates.
    """

    depth_points_mm = []
    for coords in coords_list:
        depth_value = depth_frame.get_distance(coords[0], coords[1])
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, coords, depth_value)
        depth_point_mm = [point * 1000 for point in depth_point]
        depth_points_mm.append(depth_point_mm)
    return depth_points_mm

if __name__ == '__main__':
    profile, pipeline, config = init_camera(resolution_height=480, resolution_width=848, fps=30)
    color_image, depth_frame, depth_scale, depth_intrinsics, color_intrinsics = capture_aligned_images(pipeline, config, profile, hole_filling=True, show_frames=True)
    # clr_Frame, depth_frame = capture_aligned_images_filter_before_align(pipeline, config, profile, hole_filling=True)
    print(f'Distance to the center of the image {depth_frame.get_distance(424, 240)}')
    print(f'Real word coords: {pixel_to_mm_coordinates([[424, 240]], depth_frame, color_intrinsics)}')
    