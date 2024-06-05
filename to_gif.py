import imageio
import os

# Define the directory containing the images
dataset = '393'
depth_ratio = '1.00'
view = '02'
# image_dir = f'./exp/{dataset}_2dgs_{depth_ratio}/test-view/renders'
image_dir = f'./exp/zju_393_mono-none-smpl_nn-identity-sh-default/test-view/renders'

# Get a sorted list of file names in the directory
file_names = sorted(
    [f for f in os.listdir(image_dir) if f.startswith(f"render_c{view}_f") and f.endswith(".png")],
    key=lambda x: int(x.split('f000')[-1].split('.')[0])  # Extract the index and sort numerically
)

# Create a list to store the images
images = []

# Read and append each image to the list
for file_name in file_names:
    file_path = os.path.join(image_dir, file_name)
    images.append(imageio.imread(file_path))

# Define the output path for the GIF
output_path = f'{dataset}_c{view}_2dgs_new.gif'

# Save the images as a GIF
imageio.mimsave(output_path, images, duration=0.01)  # You can adjust the duration between frames

print(f"GIF saved at {output_path}")
