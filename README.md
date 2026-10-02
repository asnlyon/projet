import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d


def get_object_position(img):
	"""Display a grayscale image and return two user-selected points."""
	plt.figure()
	plt.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)
	points = plt.ginput(2, timeout=-1)
	plt.close()
	return points


def create_gaussian_kernel(size=5, sigma=1.5):
	"""Create a normalized two-dimensional Gaussian kernel."""
	if size <= 0 or sigma <= 0:
		raise ValueError("size and sigma must be positive")

	coordinates = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
	x_grid, y_grid = np.meshgrid(coordinates, coordinates)
	kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
	return kernel / np.sum(kernel)


def process_and_sharpen(img_path):
	"""Deskew, crop, resize, and sharpen an image selected by the user."""
	image = Image.open(img_path).convert("L")
	fill_color = image.getpixel((0, 0))

	rotation_scale = 0.7
	scaled_size = tuple(
		max(1, int(round(dimension * rotation_scale))) for dimension in image.size
	)
	scaled_image = image.resize(scaled_size, Image.Resampling.LANCZOS)
	rotation_input = Image.new("L", image.size, color=fill_color)
	image_offset = (
		(image.width - scaled_image.width) // 2,
		(image.height - scaled_image.height) // 2,
	)
	rotation_input.paste(scaled_image, image_offset)
	tilted_image = rotation_input.rotate(45, expand=False, fillcolor=fill_color)
	point_1, point_2 = get_object_position(tilted_image)

	angle = np.arctan2(point_2[1] - point_1[1], point_2[0] - point_1[0])
	rotation_degrees = np.degrees(angle)
	deskewed_image = tilted_image.rotate(
		rotation_degrees,
		expand=False,
		fillcolor=tilted_image.getpixel((0, 0)),
	)

	image_center = np.array([image.width / 2, image.height / 2])
	rotation_matrix = np.array(
		[
			[np.cos(-angle), -np.sin(-angle)],
			[np.sin(-angle), np.cos(-angle)],
		]
	)
	centered_points = np.array([point_1, point_2]) - image_center
	rotated_points = centered_points @ rotation_matrix.T + image_center

	center_x, center_y = np.mean(rotated_points, axis=0)
	scale = np.linalg.norm(rotated_points[1] - rotated_points[0])
	crop_box = (
		center_x - scale,
		center_y - 0.5 * scale,
		center_x + scale,
		center_y + 0.5 * scale,
	)
	crop_box = tuple(int(round(coordinate)) for coordinate in crop_box)

	cropped_image = deskewed_image.crop(crop_box)
	resized_image = cropped_image.resize((800, 400), Image.Resampling.LANCZOS)

	image_array = np.asarray(resized_image, dtype=float)
	gaussian_kernel = create_gaussian_kernel(size=5, sigma=1.5)
	low_pass = convolve2d(image_array, gaussian_kernel, mode="same")
	high_pass = image_array - low_pass
	sharpened = np.clip(image_array + (3.0 * high_pass), 0, 255).astype(np.uint8)

	figure, axes = plt.subplots(1, 2)
	axes[0].imshow(image_array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
	axes[0].set_title("Avant")
	axes[0].axis("off")
	axes[1].imshow(sharpened, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
	axes[1].set_title("Après")
	axes[1].axis("off")
	figure.tight_layout()
	plt.show()

	return sharpened


if __name__ == "__main__":
	process_and_sharpen("plane1.jpg")
