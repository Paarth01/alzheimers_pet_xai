import os
from PIL import Image

def crop_assets():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    image_path = os.path.join(project_root, "outputs", "stage_comparison_gradcam.png")
    assets_dir = os.path.join(base_dir, "assets")
    
    os.makedirs(assets_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Source image not found at {image_path}")
        return False
        
    print(f"Loading stage comparison image from {image_path}...")
    img = Image.open(image_path)
    
    # Coordinates detected from image analysis:
    # Columns: Original, Heatmap, Overlay
    cols = [
        ("original", (176, 489)),
        ("heatmap", (525, 854)),
        ("overlay", (891, 1214))
    ]
    
    # Rows: CN, MCI, AD
    rows = [
        ("cn", (138, 392)),
        ("mci", (399, 658)),
        ("ad", (663, 929))
    ]
    
    for row_name, (y_start, y_end) in rows:
        for col_name, (x_start, x_end) in cols:
            crop_box = (x_start, y_start, x_end, y_end)
            cropped_img = img.crop(crop_box)
            
            output_filename = f"{row_name}_{col_name}.png"
            output_path = os.path.join(assets_dir, output_filename)
            
            # Save cropped image
            cropped_img.save(output_path, "PNG")
            print(f"Saved cropped asset: {output_filename} ({cropped_img.size})")
            
    print("Asset preparation completed successfully.")
    return True

if __name__ == "__main__":
    crop_assets()
