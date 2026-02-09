import argparse
import cv2 as cv
import cv2.aruco
import numpy as np
import plyfile
import glob
import os
import shutil
import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
import OCDatasetLoader.OCArucoDetector as OCArucoDetector

from collections import namedtuple
from copy import deepcopy

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


def keyPressManager():
	print('keyPressManager.\nPress "c" to continue or "q" to abort.')
	while True:
		key = cv.waitKey(15)
		if key == ord('c'):
			print('Pressed "c". Continuing.')
			break
		elif key == ord('q'):
			print('Pressed "q". Aborting.')
			exit(0)


def drawMask(img, imgpts):

	imgpts = np.int32(imgpts).reshape(-1, 2)

	img = cv.drawContours(img, [imgpts[:4]], -1, (255, 255, 255), -3)

	img = cv.drawContours(img, np.hstack([[imgpts[4:6]],[imgpts[1::-1]]]), -1, (255, 255, 255), -3)

	img = cv.drawContours(img, np.hstack([[imgpts[5:7]],[imgpts[2:0:-1]]]), -1, (255, 255, 255), -3)

	img = cv.drawContours(img, np.hstack([[imgpts[6:8]],[imgpts[3:1:-1]]]), -1, (255, 255, 255), -3)

	img = cv.drawContours(img, np.hstack([[imgpts[3::-3]],[imgpts[4::3]]]), -1, (255, 255, 255), -3)

	# img = cv.drawContours(img, [imgpts[4:]], -1, (255, 255, 255), -3)

	return img


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
if __name__ == "__main__":

	# ---------------------------------------
	# --- Parse command line argument
	# ---------------------------------------
	ap = argparse.ArgumentParser()
	ap = OCDatasetLoader.addArguments(ap) # Dataset loader arguments
	args = vars(ap.parse_args())
	print(args)

	# ---------------------------------------
	# --- INITIALIZATION
	# ---------------------------------------

	dataset_loader = OCDatasetLoader.Loader(args)
	dataset_cameras = dataset_loader.loadDataset()

	aruco_detector = OCArucoDetector.ArucoDetector(args)

	dataset_arucos, dataset_cameras = aruco_detector.detect(dataset_cameras)
	print("\nDataset_cameras contains " + str(len(dataset_cameras.cameras)) + " cameras")

	# Extra
	dataset_arucos.distortion = np.array(dataset_cameras.cameras[0].rgb.camera_info.D)
	dataset_arucos.intrinsics = np.reshape(dataset_cameras.cameras[0].rgb.camera_info.K, (3, 3))
	dataset_arucos.world_T_aruco = {}
	dataset_arucos.aruco_T_world = {}

	for i, camera in enumerate(dataset_cameras.cameras):
		for key, aruco in camera.rgb.aruco_detections.items():

			# Add only if there is still not an estimate (made by another camera)
			if key not in dataset_arucos.world_T_aruco:
				dataset_arucos.world_T_aruco[key] = np.dot(camera.rgb.matrix, np.linalg.inv(camera.rgb.aruco_detections[key].aruco_T_camera))

			if key not in dataset_arucos.aruco_T_world:
				dataset_arucos.aruco_T_world[key] = np.dot(camera.rgb.matrix, camera.rgb.aruco_detections[key].aruco_T_camera)

	# ---------------------------------------
	# --- Inpainting
	# ---------------------------------------

	arucoBorder = 0.017
	arucoThickness = 0.006
	blurBorder = 0.035

	mask3DPoints = np.float32(
		[[-(dataset_arucos.markerSize / 2 + arucoBorder), dataset_arucos.markerSize / 2 + arucoBorder, 0.002],
		 [dataset_arucos.markerSize / 2 + arucoBorder, dataset_arucos.markerSize / 2 + arucoBorder, 0.002],
		 [dataset_arucos.markerSize / 2 + arucoBorder, -(dataset_arucos.markerSize / 2 + arucoBorder), 0.002],
		 [-(dataset_arucos.markerSize / 2 + arucoBorder), -(dataset_arucos.markerSize / 2 + arucoBorder), 0.002],
		 [-(dataset_arucos.markerSize / 2 + arucoBorder), dataset_arucos.markerSize / 2 + arucoBorder, -arucoThickness],
		 [dataset_arucos.markerSize / 2 + arucoBorder, dataset_arucos.markerSize / 2 + arucoBorder, -arucoThickness],
		 [dataset_arucos.markerSize / 2 + arucoBorder, -(dataset_arucos.markerSize / 2 + arucoBorder), -arucoThickness],
		 [-(dataset_arucos.markerSize / 2 + arucoBorder), -(dataset_arucos.markerSize / 2 + arucoBorder), -arucoThickness]])

	blurMask3DPoints = np.float32(
		[[-(dataset_arucos.markerSize / 2 + arucoBorder + blurBorder), dataset_arucos.markerSize / 2 + arucoBorder + blurBorder, 0.002],
		 [dataset_arucos.markerSize / 2 + arucoBorder + blurBorder, dataset_arucos.markerSize / 2 + arucoBorder + blurBorder, 0.002],
		 [dataset_arucos.markerSize / 2 + arucoBorder + blurBorder, -(dataset_arucos.markerSize / 2 + arucoBorder + blurBorder), 0.002],
		 [-(dataset_arucos.markerSize / 2 + arucoBorder + blurBorder), -(dataset_arucos.markerSize / 2 + arucoBorder + blurBorder), 0.002],
		 [-(dataset_arucos.markerSize / 2 + arucoBorder + blurBorder), dataset_arucos.markerSize / 2 + arucoBorder + blurBorder, -arucoThickness],
		 [dataset_arucos.markerSize / 2 + arucoBorder + blurBorder, dataset_arucos.markerSize / 2 + arucoBorder + blurBorder, -arucoThickness],
		 [dataset_arucos.markerSize / 2 + arucoBorder + blurBorder, -(dataset_arucos.markerSize / 2 + arucoBorder + blurBorder), -arucoThickness],
		 [-(dataset_arucos.markerSize / 2 + arucoBorder + blurBorder), -(dataset_arucos.markerSize / 2 + arucoBorder + blurBorder), -arucoThickness]])

	# Create dir to save coloured point clouds
	directory = 'ColouredClouds'
	if os.path.exists(directory):
		# Delete old folder
		shutil.rmtree(directory)
	os.makedirs(directory)

	for i, camera in enumerate(dataset_cameras.cameras):

		# Actually perform the inpainting. <False> only used for convenience when colouring with original texture (to compare results). 
		# TODO: add arg for this flag
		if True:
			print("Creating a mask for camera " + camera.name + "...")

			image = deepcopy(camera.rgb.image)
			height, width, channels = image.shape
			mask = np.zeros((height, width), dtype=np.uint8)
			crossmask = np.zeros((height, width), dtype=np.uint8)
			ghostMask = np.zeros((height, width), dtype=np.uint8)
			blurMask = np.zeros((height, width), dtype=np.uint8)
			world_T_camera = np.linalg.inv(camera.rgb.matrix)

			inpainted_arucos = []

			# For each ArUco detected in the image
			for key, aruco in camera.rgb.aruco_detections.items():

				# Project 3D points to image plane
				imgpts, jac = cv2.projectPoints(mask3DPoints, aruco.rodrigues, aruco.translation,
												dataset_arucos.intrinsics, dataset_arucos.distortion)

				maskpts, jac = cv2.projectPoints(blurMask3DPoints, aruco.rodrigues, aruco.translation,
												 dataset_arucos.intrinsics, dataset_arucos.distortion)

				mask = drawMask(mask, imgpts)

				blurMask = drawMask(blurMask, maskpts)

				#print("\t\t\t Added Aruco " + str(key) + " to the mask;")

				inpainted_arucos.extend([key])

			print('ArUcos in actual detection= ' + str(inpainted_arucos))

			# Show the masks over the original image
			redImg = np.zeros(image.shape, image.dtype)
			redImg[:, :] = (0, 0, 255)
			redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
			redMaskImage = cv.addWeighted(redMask, 0.5, image, 0.5, 0.0)

			# Cross-Inpainting - For ArUcos not in the image
			for key in dataset_arucos.arucos.iterkeys():

				if key in inpainted_arucos:
					continue

				aruco_T_world = dataset_arucos.arucos[key].matrix

				homogenous3DMask = np.ones(shape=(len(mask3DPoints), 4))

				homogenous3DMask[:, :-1] = mask3DPoints

				maskInNewCamera = np.zeros(shape=(len(mask3DPoints), 3))

				for k in range(len(mask3DPoints)):

					maskInNewCamera[k] = np.dot(world_T_camera, np.dot(aruco_T_world, homogenous3DMask[k]))[:3]

				# print("point in new camera = " + str(maskInNewCamera))

				# project 3D points to image plane
				ghostpts, jac = cv2.projectPoints(maskInNewCamera, (0,0,0), (0,0,0),
												dataset_arucos.intrinsics, dataset_arucos.distortion)

				mask = drawMask(mask, ghostpts)
				crossmask = drawMask(crossmask, ghostpts)

			blueImg = np.zeros(image.shape, image.dtype)
			blueImg[:, :] = (255, 100, 0)
			blueMask = cv2.bitwise_and(blueImg, blueImg, mask=crossmask)
			redMaskImage = cv.addWeighted(blueMask, 0.5, redMaskImage, 0.5, 0.0)

			# Show the mask images
			# cv.namedWindow('cam mask ' + str(i), cv.WINDOW_NORMAL)
			# cv.imshow('cam mask ' + str(i), ghostMask)

			# Apply inpainting algorithm
			inpaintedImage = cv.inpaint(image, mask, 5, cv.INPAINT_TELEA)
			# inpaintedImage = cv.inpaint(image, mask, 5, cv.INPAINT_NS)

			# Show the first inpaint
			# cv.namedWindow('First inpaint ' + str(i), cv.WINDOW_NORMAL)
			# cv.imshow('First inpaint ' + str(i), inpaintedImage)

			# Show masks over images
			cv.namedWindow('Red Mask image ' + str(i), cv.WINDOW_NORMAL)
			cv.imshow('Red Mask image ' + str(i), redMaskImage)

			# Apply first blur to smooth inpainting
			blurredImage = cv.medianBlur(inpaintedImage, 201)
			inpaintedImage[np.where(mask == 255)] = blurredImage[np.where(mask == 255)]

			# Show the blurred image
			# cv.namedWindow('Blurred image 1 ' + str(i), cv.WINDOW_NORMAL)
			# cv.imshow('Blurred image 1 ' + str(i), blurredImage)

			# Apply second blur to smooth out edges
			blurredImage = cv.medianBlur(inpaintedImage, 51)
			inpaintedImage[np.where(blurMask == 255)] = blurredImage[np.where(blurMask == 255)]

			# Show the blurred image
			# cv.namedWindow('Blurred image 2 ' + str(i), cv.WINDOW_NORMAL)
			# cv.imshow('Blurred image 2 ' + str(i), blurredImage)

			# Overall blur??
			inpaintedImage = cv.bilateralFilter(inpaintedImage, 9, 75, 75)
			# inpaintedImage = cv.blur(inpaintedImage, (5, 5))

			# Show the final image
			cv.namedWindow('Inpainted image ' + str(i), cv.WINDOW_NORMAL)
			cv.namedWindow('Inpainted image ' + str(i), cv.WINDOW_FULLSCREEN)
			cv.imshow('Inpainted image ' + str(i), inpaintedImage)
		else:
			# Just color in with the original images
			inpaintedImage = deepcopy(camera.rgb.image)
		###################################################################
		# Show print .ply file with color

		# Read vertices from point cloud
		ply_input_filename = args['path_to_images'] + '/' + camera.name.zfill(8) + '.ply'
		imgData = plyfile.PlyData.read(ply_input_filename)["vertex"]
		numVertex = len(imgData['x'])

		# create array of 3d points						   add 1 to make homogeneous
		xyz = np.c_[imgData['x'], imgData['y'], imgData['z'], np.ones(shape=(imgData['z'].size, 1))]

		pointsInOpenCV = np.zeros(shape=(len(xyz), 3))

		pointColour = np.zeros(shape=(len(xyz), 3))

		print("#################################################")
		print("Camera " + camera.name + "\n")

		# The local point clouds (.ply files) are stored in OpenGL coordinates.
		# This matrix puts the coordinate frames back in OpenCV fashion
		camera.depth.matrix[0, :] = [1, 0, 0, 0]
		camera.depth.matrix[1, :] = [0, 0, 1, 0]
		camera.depth.matrix[2, :] = [0, -1, 0, 0]
		camera.depth.matrix[3, :] = [0, 0, 0, 1]

		world_T_camera = np.linalg.inv(camera.rgb.matrix)

		for j in range(len(xyz)):
			pointsInOpenCV[j] = np.dot(world_T_camera, np.dot(camera.depth.matrix, xyz[j]))[:3]

		# print("Points transformed from OpenGl to OpenCV coords = ")
		# print(pointsInOpenCV)

		# project 3D points from ArUco to image plane
		pointsInImage, jac = cv2.projectPoints(pointsInOpenCV, (0, 0, 0), (0, 0, 0),
											   dataset_arucos.intrinsics, dataset_arucos.distortion)

		# print("Points projected to image = ")
		# print(pointsInImage)

		image = deepcopy(camera.rgb.image)

		# Figure out how many points project into the image
		nPointsWithColour = 0

		for j in range(len(pointsInImage)):
			row = int(round(pointsInImage[j][0][1]))
			col = int(round(pointsInImage[j][0][0]))

			# if it was projected within the image
			if 0 <= row < 1080 and 0 <= col < 1920:
				nPointsWithColour = nPointsWithColour + 1

		# Create the .ply file
		ply_output_filename = directory + '/' + camera.name.zfill(8) + '_with_colour.ply'
		file_object = open(ply_output_filename, "w")
		print(ply_output_filename)

		# write file header information
		file_object.write('ply' + '\n')
		file_object.write('format ascii 1.0' + '\n')
		file_object.write('comment ---' + '\n')
		file_object.write('element vertex ' + str(nPointsWithColour) + '\n')
		file_object.write('property float x' + '\n')
		file_object.write('property float y' + '\n')
		file_object.write('property float z' + '\n')
		file_object.write('property uchar red' + '\n')
		file_object.write('property uchar green' + '\n')
		file_object.write('property uchar blue' + '\n')
		file_object.write('element face 0' + '\n')
		file_object.write('property list uchar uint vertex_indices' + '\n')
		file_object.write('end_header' + '\n')

		# Actually get the colours for the points projected
		for j in range(len(pointsInImage)):
			row = int(round(pointsInImage[j][0][1]))
			col = int(round(pointsInImage[j][0][0]))

			# if it was projected within the image
			if 0 <= row < 1080 and 0 <= col < 1920:
				pointColour[j] = inpaintedImage[row, col]

				file_object.write(str(imgData['x'][j]) + ' ' + str(imgData['y'][j]) + ' ' + str(imgData['z'][j]) +
								  ' ' + str(int(pointColour[j][2])) + ' ' + str(int(pointColour[j][1])) +
								  ' ' + str(int(pointColour[j][0])) + '\n')
		file_object.close()
		print("Done!\n")

	keyPressManager()