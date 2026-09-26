
# WEEK 2 LAB  IMAGE PROCESSING THROUGH A  4 STEPS PIPELINE


# STUDENT INFORMATION

# NAME                   ABDUL RAHIM
# ROLL NO                2K24/CSE/14






# ==============================================================================
# PROBLEM 1: THRESHOLDING
# ==============================================================================

# Step 1: Inputs and output
# Input: A 2D NumPy array ('image') with grayscale values in [0, 255] and an integer ('threshold').
# Output: A 2D NumPy array of dtype uint8 containing only values 0 and 255.

# Step 2: Rule
# If pixel >= threshold, output 255; otherwise output 0.

# Step 3: Small example
# Threshold = 128
# Input: [[20, 128, 200], [100, 150, 250]]
# Calculation:
# 20 < 128 -> 0   | 128 >= 128 -> 255 | 200 >= 128 -> 255
# 100 < 128 -> 0  | 150 >= 128 -> 255 | 250 >= 128 -> 255
# Output: [[0, 255, 255], [0, 255, 255]]

# Step 4: Python implementation

import numpy as np

def threshold_image(image: np.ndarray, threshold: int) -> np.ndarray:
    """Convert a grayscale image to black and white."""
    return np.where(image >= threshold, 255, 0).astype(np.uint8)

problem_1_input = np.array([[20, 128, 200], [100, 150, 250]], dtype=np.uint8)
problem_1_expected = np.array([[0, 255, 255], [0, 255, 255]], dtype=np.uint8)

np.testing.assert_array_equal(threshold_image(problem_1_input, 128), problem_1_expected)
print("Problem 1 passed")


# ==============================================================================
# PROBLEM 2: IMAGE MEMORY
# ==============================================================================

# Step 1: Inputs and output
# Inputs: Image width (int), height (int), and bits per pixel 'bpp' (int).
# Output: Memory used by the uncompressed image in bytes (int).

# Step 2: Rule
# bytes = (width * height * bpp) / 8

# Step 3: Small example
# Width = 1920, Height = 1080, bpp = 24
# Calculation: (1920 * 1080 * 24) / 8 = 6,220,800 bytes
# Output: 6220800

# Step 4: Python implementation
def display_memory_bytes(width: int, height: int, bpp: int) -> int:
    """Return image memory in bytes."""
    return int(width * height * bpp // 8)

expected_bytes = 1920 * 1080 * 24 // 8
assert display_memory_bytes(1920, 1080, 24) == expected_bytes
print("Problem 2 passed")


# ==============================================================================
# PROBLEM 3: AVERAGE 9 PIXELS
# ==============================================================================

# Step 1: Inputs and output
# Input: A 3x3 NumPy array ('neighborhood') containing 9 grayscale pixel values.
# Output: One rounded integer representing the average pixel value.

# Step 2: Rule
# average = round( sum(all 9 values) / 9 )

# Step 3: Small example
# Input: [[10, 20, 10], [30, 50, 30], [10, 20, 10]]
# Sum = 10 + 20 + 10 + 30 + 50 + 30 + 10 + 20 + 10 = 190
# Average = 190 / 9 = 21.111...
# Rounded Output: 21

# Step 4: Python implementation

import numpy as np

def mean_filter_3x3(neighborhood: np.ndarray) -> int:
    """Return the rounded average of 9 pixels."""
    return int(round(np.mean(neighborhood)))

problem_3_input = np.array([[10, 20, 10], [30, 50, 30], [10, 20, 10]], dtype=np.uint8)

assert mean_filter_3x3(problem_3_input) == 21
print("Problem 3 passed")



# ==============================================================================
# PROBLEM 4: CHANGE IMAGE CONTRAST
# ==============================================================================

# Step 1: Inputs and output
# Input: An array ('image'), old range boundaries ('in_min', 'in_max'), and target range boundaries ('out_min', 'out_max').
# Output: A NumPy array with values stretched and clipped to the new range [out_min, out_max].

# Step 2: Rule
# new_value = (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
# final_value = np.clip(new_value, out_min, out_max)

# Step 3: Small example
# Input: [50, 100, 150], in_min=50, in_max=150, out_min=0, out_max=255
# 50  -> (50 - 50)/(100) * 255 + 0 = 0.0
# 100 -> (100 - 50)/(100) * 255 + 0 = 127.5
# 150 -> (150 - 50)/(100) * 255 + 0 = 255.0
# Output: [0.0, 127.5, 255.0]


# Step 4: Python implementation

import numpy as np

def contrast_stretch(
    image: np.ndarray,
    in_min: float,
    in_max: float,
    out_min: float = 0.0,
    out_max: float = 255.0,
) -> np.ndarray:
    """Change image values from one range to another."""
    stretched = (image - in_min) / (in_max - in_min) * (
        out_max - out_min
    ) + out_min
    return np.clip(stretched, out_min, out_max)


problem_4_input = np.array([50, 100, 150], dtype=np.float32)
expected = np.array([0.0, 127.5, 255.0], dtype=np.float32)

np.testing.assert_allclose(
    contrast_stretch(problem_4_input, 50, 150), expected
)
print("Problem 4 passed")


# ==============================================================================
# PROBLEM 5: FIND AN EDGE WITH SOBEL
# ==============================================================================

# Step 1: Inputs and output
# Input: A 3x3 grayscale NumPy block ('block').
# Output: A tuple (gx, gy, strength) of floating-point values representing horizontal gradient, vertical gradient, and edge magnitude.

# Step 2: Rule
# gx = sum(block * Gx), gy = sum(block * Gy)
# strength = sqrt(gx^2 + gy^2)
# Gx kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
# Gy kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# Step 3: Small example
# Input: [[20, 20, 200], [20, 20, 200], [20, 20, 200]]
# gx = (-1*20 + 0*20 + 1*200) + (-2*20 + 0*20 + 2*200) + (-1*20 + 0*20 + 1*200) = 180 + 360 + 180 = 720.0
# gy = (-1*20 - 2*20 - 1*200) + (0) + (1*20 + 2*20 + 1*200) = -260 + 0 + 260 = 0.0
# strength = sqrt(720^2 + 0^2) = 720.0
# Output: (720.0, 0.0, 720.0)


# Step 4: Python implementation

import numpy as np

def sobel_response(block: np.ndarray) -> tuple[float, float, float]:
    """Return gx, gy, and edge strength."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = float(np.sum(block * Kx))
    gy = float(np.sum(block * Ky))
    strength = float(np.sqrt(gx**2 + gy**2))

    return gx, gy, strength


problem_5_input = np.array(
    [[20, 20, 200], [20, 20, 200], [20, 20, 200]], dtype=np.float32
)

gx, gy, strength = sobel_response(problem_5_input)
assert gx == 720.0 and gy == 0.0
print("Problem 5 passed")