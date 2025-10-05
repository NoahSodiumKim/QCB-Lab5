import cv2
import numpy as np

# --- 1. Define the Circularity Calculation ---
# A perfect circle has a circularity of 1.
# The formula is (4 * PI * Area) / (Perimeter^2)
def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Avoid division by zero
    if perimeter == 0:
        return 0
        
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    return circularity

# --- 2. Load and Pre-process the Image ---
# Replace 'petri_dish.jpg' with the path to your image
image = cv2.imread('group6_Noah.png')
if image is None:
    print("Error: Could not load image.")
    exit()

# Create a copy to draw on later
output_image = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve thresholding
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# --- 3. Segment the Image (Thresholding) ---
# This step is crucial and may require tuning.
# We use Otsu's thresholding which automatically finds an optimal
# threshold value. Since colonies are often darker than the background,
# we use THRESH_BINARY_INV to make them white.
otsu_threshold, image_result = cv2.threshold(
        blurred, 
        0, 
        255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
threshold_value = otsu_threshold*2 + 20

_, binary_image = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)


    # 3. Print the optimal threshold value that Otsu's method found
print(f"Otsu's method calculated the optimal threshold to be: {otsu_threshold}")

# Optional: Clean up the binary image with morphological operations
# kernel = np.ones((5,5),np.uint8)
# binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
# binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


# --- 4. Find Contours ---
# Find the outlines of the white objects in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- 5. Filter Contours by Area and Circularity ---
# Set your filtering parameters
min_area = 25        # Minimum pixel area to be considered a colony
max_area = 5000        # Maximum pixel area
circularity_threshold = 0.6 # How close to a perfect circle (1.0)

detected_colonies = 0

for c in contours:
    # Calculate area and circularity
    area = cv2.contourArea(c)
    circularity = calculate_circularity(c)

    # Check if the contour meets the criteria
    if min_area < area < max_area and circularity > circularity_threshold:
        detected_colonies += 1
        
        # Draw the contour on the output image
        cv2.drawContours(output_image, [c], -1, (0, 255, 0), 2) # Draw in green

        # Get the center of the contour to put text
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Put the circularity value next to the contour
            cv2.putText(output_image, f"{circularity:.2f}", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# --- 6. Display the Results ---
print(f"Detected {detected_colonies} colonies meeting the criteria.")
cv2.putText(output_image, f"Count: {detected_colonies}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the images
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Detected Colonies', output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()