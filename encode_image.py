import base64
import argparse
import os

def image_to_base64(filepath):
    """Encodes an image file to a Base64 string."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Image file not found at '{filepath}'")

    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an image file to a Base64 string.")
    parser.add_argument("image_path", help="The path to the image file (e.g., images/my_image.png)")
    args = parser.parse_args()

    try:
        base64_string = image_to_base64(args.image_path)
        print(base64_string)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")