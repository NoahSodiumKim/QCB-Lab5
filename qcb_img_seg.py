import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor

#loading model + processor
model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

#loading img
try:
    image_path = "group_4_sihao_jiang_cropedit.png"
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: '{image_path}' not found")
    exit()

original_image_np = np.array(image)

#processing + predicting
inputs = processor(image, return_tensors="pt").to(device)

print("finding masks")
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores.cpu()

final_masks = masks[0]
final_scores = scores[0]

#filtering out masks w low confidence
score_threshold = 0.8
confident_masks = final_masks[final_scores > score_threshold]
print(f"found {len(confident_masks)} confident masks (score > {score_threshold}).")

#creating colorized segmented image
def create_colorized_segmentation_image(original_image_np, masks, alpha=0.6):
    segmented_image = original_image_np.copy().astype(np.float32)

    #make masks red
    red_color = np.array([1.0, 0.0, 0.0])  

    for i, mask in enumerate(masks):
        #inverting mask cuz default highlights negative space
        mask_np = mask.cpu().numpy().astype(np.float32)
        inverted_mask = 1.0 - mask_np
        
        mask_expanded = np.expand_dims(inverted_mask, axis=-1)

        colored_overlay = mask_expanded * red_color * 255 

        #blending overlay w og image
        segmented_image = (segmented_image * (1 - alpha * mask_expanded)) + \
                          (colored_overlay * alpha * mask_expanded)

    segmented_image_uint8 = np.clip(segmented_image, 0, 255).astype(np.uint8)
    return Image.fromarray(segmented_image_uint8)


if len(confident_masks) > 0:
    colorized_output_image = create_colorized_segmentation_image(original_image_np, confident_masks)

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(colorized_output_image)
    axs[1].set_title(f"Segmented Colonies ({len(confident_masks)} found)")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    output_filename = "bacteria_colonies_segmented.png"
    colorized_output_image.save(output_filename)
    print(f"\nseg image saved as: {output_filename}")
else:
    print("No confident masks found :(")