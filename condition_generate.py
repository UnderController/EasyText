import os
import json
from PIL import Image, ImageDraw, ImageFont
import random
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Select a font that supports the target language based on the language to be rendered.（font_path）")
    parser.add_argument('--font_number', type=int,default=5, help="Numeric parameter to select different font paths (use 5 for English): 1-ch 2-jan 3-ko 4-thai 5-Vietnamese,latin,Cyrillic,greek(including English, Italian, French, German, Russian, Greek, Vietnamese, etc.)")
    # input_directory 和 output_directory 参数
    parser.add_argument('--input_directory', type=str, default=r"sample\ch", help="Enter the folder path. The folder must contain both the text box position and content information for each image.")
    parser.add_argument('--output_directory', type=str, default=r"sample\ch_con", help="output path for condition")

    return parser.parse_args()

# 根据 font_number 选择 font_path
def get_font_path(font_number):
    font_paths = {
        1: "font/simhei.ttf",
        2: "font/Japanese/SourceHanSerif-Regular.otf",
        3: "font/Korean/SourceHanSerifK-Regular.otf",
        4: 'font/tahoma.ttf',
        5: "font/arial.ttf",
    }
    if font_number not in font_paths:
        raise ValueError(f"Invalid font_number: {font_number}. Please choose a valid number.")

    return font_paths[font_number]

def render_text_on_image(font_path,text_list, output_image_path, image_size=(64, 64), margin=3):
    """
    Renders text on a white image with dynamic height for multiple lines.
    :param text_list: List of texts to render
    :param output_image_path: Path to save the output image
    :param image_size: Size of the image and individual text box
    :param margin: Margin around the text within each box
    :return: List of text positions [(original_position, new_position)]
    """
    images = []
    text_positions = []
    current_y = 0
    for text in text_list:
        font_size = 55
        text_width = len(text) * image_size[0]
        text_height = image_size[1]
        line_image = Image.new("RGB", (text_width, text_height), "white")
        draw = ImageDraw.Draw(line_image)
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width_ = bbox[2] - bbox[0]
        text_height_ = bbox[3] - bbox[1]
        max_height = text_height_ - margin
        while text_height_ > max_height:
            font_size -= 2
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_height_ = bbox[3] - bbox[1]
            text_width_ = bbox[2] - bbox[0]
        while text_width_ > (text_width - margin):
            font_size -= 2
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width_ = bbox[2] - bbox[0]

        text_width = text_width_- text_width_%16 +32
        line_image = Image.new("RGB", (text_width, text_height), "white")
        draw = ImageDraw.Draw(line_image)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width_actual = text_bbox[2] - text_bbox[0]
        text_height_actual = text_bbox[3] - text_bbox[1]
        x = (text_width - text_width_actual) // 2
        y = (text_height - text_height_actual) // 2
        draw.text((x, y-text_bbox[1]), text, font=font, fill="black")

        top_left = (0, current_y)
        current_y = current_y +64
        bottom_right = (text_width - 1, current_y -1)
        text_positions.append([(top_left, bottom_right)])

        images.append(line_image)
    total_width = max([img.width for img in images])
    total_height = len(images) * image_size[1]

    final_image = Image.new("RGB", (total_width, total_height), "white")
    current_yy = 0
    for img in images:
        final_image.paste(img, (0, current_yy))
        current_yy += img.height
    final_image.save(output_image_path)


    return text_positions


def process_image_and_positions(font_path,text_position_path, output_dir, image_size=(64, 64), margin=5):
    """
    Processes a single image and its corresponding position file.
    :param text_position_path: Path to the position file
    :param output_dir: Directory to save the new images and updated position files
    :param image_size: Size of each text block in the image
    :param margin: Margin between text blocks
    """
    # Read the position data
    with open(text_position_path, "r", encoding="utf-8") as f:
        try:
            position_data = json.load(f)
            #print(text_position_path)
        except:
            print(text_position_path)

    text_list = []
    for item in position_data:
        text_list.append(item[1])  # Extract the text

    # Create the new image with rendered text
    output_image_path = os.path.join(output_dir, os.path.basename(text_position_path).replace('-position.txt', '-text-wb.png'))
    text_positions = render_text_on_image(font_path,text_list, output_image_path, image_size, margin)

    # Modify position data with the new positions
    new_position_data = []
    for old_pos, new_pos in zip(position_data, text_positions):
        new_position_data.append([old_pos[0], new_pos[0]])

    # Save the updated position data
    output_position_path = os.path.join(output_dir, os.path.basename(text_position_path))
    with open(output_position_path, "w", encoding="utf-8") as f:
        json.dump(new_position_data, f, ensure_ascii=False, indent=4)


def process_all_images_in_directory(input_dir, output_dir,font_path, image_size=(64, 64), margin=5):
    """
    Processes all images in the directory with corresponding position files.
    :param input_dir: Directory containing the images and position files
    :param output_dir: Directory to save the new images and updated position files
    :param image_size: Size of each text block in the image
    :param margin: Margin between text blocks
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith("-position.txt"):
            image_path = os.path.join(input_dir, filename)
            position_file_path = os.path.join(input_dir, filename)

            if os.path.exists(position_file_path):
                process_image_and_positions(font_path,position_file_path, output_dir, image_size, margin)


def main():
    args = parse_args()
    font_number = args.font_number
    font_path = get_font_path(font_number)
    input_directory = args.input_directory
    output_directory = args.output_directory
    process_all_images_in_directory(input_directory, output_directory,font_path)

if __name__ == '__main__':
    main()