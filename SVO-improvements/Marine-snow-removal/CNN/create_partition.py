import os
import random
import shutil

original_folder = "E:\CNN\Artificial\images"
ground_truth_folder = "E:\CNN\Artificial\masks"
output_folder = "data"
split_ratio = (0.7, 0.15, 0.15)

if sum(split_ratio) > 1:
    raise ValueError("The sum of split ratios cannot be greater than 1.")


# Extract the numeric part from the original image filenames
original_files = os.listdir(original_folder)
original_numbers = [
    filename.split("_")[-1].split(".")[0] for filename in original_files
]

# Shuffle the list of filenames
random.seed(42)  # Add a seed for reproducibility
random.shuffle(original_numbers)

# Calculate the number of files for each partition
total_files = len(original_numbers)
train_size = int(split_ratio[0] * total_files)
val_size = int(split_ratio[1] * total_files)

# Split the list of numbers into training, validation, and testing sets
train_numbers = original_numbers[:train_size]
val_numbers = original_numbers[train_size : train_size + val_size]
test_numbers = original_numbers[train_size + val_size :]

# Create folders for training, validation, and testing sets in the output directory
for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, folder, "masks"), exist_ok=True)

# Copy corresponding files to the respective folders
for number in train_numbers:
    original_filename = f"image_{number}.png"
    ground_truth_filename = f"mask_{number}.png"

    shutil.copy(
        os.path.join(original_folder, original_filename),
        os.path.join(output_folder, "train", "images", original_filename),
    )
    shutil.copy(
        os.path.join(ground_truth_folder, ground_truth_filename),
        os.path.join(output_folder, "train", "masks", ground_truth_filename),
    )

for number in val_numbers:
    original_filename = f"image_{number}.png"
    ground_truth_filename = f"mask_{number}.png"

    shutil.copy(
        os.path.join(original_folder, original_filename),
        os.path.join(output_folder, "val", "images", original_filename),
    )
    shutil.copy(
        os.path.join(ground_truth_folder, ground_truth_filename),
        os.path.join(output_folder, "val", "masks", ground_truth_filename),
    )

for number in test_numbers:
    original_filename = f"image_{number}.png"
    ground_truth_filename = f"mask_{number}.png"

    shutil.copy(
        os.path.join(original_folder, original_filename),
        os.path.join(output_folder, "test", "images", original_filename),
    )
    shutil.copy(
        os.path.join(ground_truth_folder, ground_truth_filename),
        os.path.join(output_folder, "test", "masks", ground_truth_filename),
    )
