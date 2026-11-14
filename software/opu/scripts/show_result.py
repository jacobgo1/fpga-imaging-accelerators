import numpy as np
from PIL import Image
from json import load as json_load


def read_labels(file_path: str, rows: int, cols: int) -> np.ndarray:
    with open(file_path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8).reshape((rows, cols))


def colour_labels(labels: np.ndarray) -> Image.Image:
    colour_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    colour_image[labels == 0] = (35, 100, 227)  # Sea: #2364e3
    colour_image[labels == 1] = (34, 139, 34)   # Land: #228b22
    colour_image[labels == 2] = (222, 222, 222) # Cloud: #dedede
    return Image.fromarray(colour_image)


def match_scaled_image(
    colour_image: Image.Image, scaled_image: Image.Image, flip: bool
) -> Image.Image:
    output_shape = scaled_image.size[::-1]
    colour_image = colour_image.resize(output_shape, resample=Image.NEAREST)
    colour_image = colour_image.rotate(90, expand=True)
    if flip:
        colour_image = colour_image.transpose(Image.FLIP_TOP_BOTTOM)
    return colour_image


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Show result of labels")
    parser.add_argument("file", type=str, help="Path to the binary file of labels")
    parser.add_argument("output", type=str, help="Path to output file")
    parser.add_argument("metadata", type=str, help="Path to the metadata JSON file")
    parser.add_argument(
        "--scaled",
        type=str,
        default=None,
        help="Path to the scaled image file, used to scale the labels correctly (optional)",
    )
    args = parser.parse_args()

    with open(args.metadata, "r") as f:
        metadata = json_load(f)

    labels = read_labels(args.file, metadata["lines"], metadata["samples"])
    colour_image = colour_labels(labels)

    if args.scaled:
        scaled_image = Image.open(args.scaled)
        colour_image = match_scaled_image(
            colour_image, scaled_image, metadata.get("flip_rgb", False)
        )

    colour_image.save(args.output)


if __name__ == "__main__":
    main()
