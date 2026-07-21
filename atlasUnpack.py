import sys
from PIL import Image
import os

def unpack_atlas(dataset_folder):
    atlas_path = os.path.join(dataset_folder, "packed.png")
    output_folder = os.path.join(dataset_folder, "FrontLD")
    tile_size = 256

    os.makedirs(output_folder, exist_ok=True)

    atlas = Image.open(atlas_path)
    width, height = atlas.size

    cols = width // tile_size
    rows = height // tile_size

    index = 0
    for row in range(rows):
        for col in range(cols):
            x = col * tile_size
            y = row * tile_size
            tile = atlas.crop((x, y, x + tile_size, y + tile_size))
            tile.save(os.path.join(output_folder, f"{index:04d}.png"))
            index += 1

    print(f"Done. Extracted {index} tiles to {output_folder}")

if __name__ == "__main__":
    dataset = sys.argv[1]
    unpack_atlas(dataset)
