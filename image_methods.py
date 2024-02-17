# Methods for doing stuff to images in response to output from face detection library.
# This is an external referenced in the server which just handles routes.

# for debugging to terminal with annotations
from icecream import ic

# open source
import face_recognition

# Python Image Library
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import random
import numpy as np

# BytesIO creates a file-like object from bytes
from io import BytesIO

# For handling images encoded as base64 strings.
import base64



# Define string hash generator called h()
def h(string_var: str) -> str:
    """
    String hash only.
    Consumes a full-length string to digest into a hash.
    Returns SHA256 hash function digest as hex string.
    """
    import hashlib
    m = hashlib.sha256()
    #ic(string_var.decode())
    #str_v = string_var.decode()
    str_v = bytes(string_var, 'utf-8')
    m.update(str_v)
    hex_str = m.hexdigest()
    ic(hex_str)
    return hex_str



# Iterative color palette. These are for when you want to cycle through colors.
# Color definitions are for iterating, such as Faces 1-10 get ROYGBIVROY for rainbow
blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)
cyan, magenta, yellow = (255, 255, 0), (255, 0, 255), (0, 255, 255)
orange, indigo, violet = (0, 127, 255), (255, 127, 127), (255, 127, 255)
black, white = (0, 0, 0), (255, 255, 255)
colors = (red, green, blue, cyan, magenta, yellow, black, white)
rainbow = (red, orange, yellow, green, blue, indigo, violet)
cold_colors = (blue, green, cyan)
warm_colors = (red, yellow, magenta)
primary_colors = (red, yellow, blue)
secondary_colors = (green, orange, violet)
rgb_colors = (red, green, blue)
bw = (black, white)


# Functions meant to be accessed externally to the module 


"""
A few functions below that return lists given image input. These need work.
This is stuff like, measuring the difference between two faces from feature vectors.
"""

def get_distance(imageA, imageB)->list:
    """
    Given two image file names, return a one-item list with the face distance.

    Args:
        imageA (filename str): first image to compare.
        imageB (filename str): second image to compare.

    Returns:
        list: distance between the two images.
    """
    first_image = face_recognition.load_image_file(imageA)
    second_image = face_recognition.load_image_file(imageB)
    
    first_encoding = face_recognition.face_encodings(first_image)[0]
    second_encoding = face_recognition.face_encodings(second_image)[0]
    
    known_encodings = []
    known_encodings.append(first_encoding)
    
    face_distances = face_recognition.face_distance(known_encodings, second_encoding)
    
    distance_values = []

    for i, face_distance in enumerate(face_distances):
        distance_values.append(i)
        print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
        print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
        print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    
    return distance_values
        


def distance_from_set(imageSet, imageA)->list:
    """
    Given two image file names, return a one-item list with the face distance.

    Args:
        imageA (filename str): first image to compare.
        imageB (filename str): second image to compare.

    Returns:
        list: distance between the two images.
    """
    encodings = []
    for img in imageSet:
        open_img = face_recognition.load_image_file(img)
        encoding = face_recognition.face_encodings(open_img)[0]
        encodings.append(encoding)
        
    test_image = face_recognition.face_encodings(imageA)[0]
    test_image_encoding = face_recognition.face_encodings(test_image)[0]
    
    face_distances = face_recognition.face_distance(encodings, test_image_encoding)
    
    distance_values = []

    for i, face_distance in enumerate(face_distances):
        distance_values.append(i)
        print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
        print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
        print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    
    return distance_values    



def draw_box(filename: str):
    img = face_recognition.load_image_file(filename)
    pil_image = Image.fromarray(img)
    
    known_face_names = []
    known_face_encodings = []
    
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    
    draw = ImageDraw.Draw(pil_image)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
        color = (0, 0, 255)
        draw.rectangle(((left, top), (right, bottom)), outline=color)
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=color, outline=color)
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    del draw
    
    return pil_image
        



"""
RAM-only versions of these to avoid writing file to memory.
BytesIO version does not commit to ROM.

Note: the uploading handler code in the server file normalizes uploads to jpeg.
"""

def str_hash_diff(str1, str2)->None:
    """
    Check to see if the hashes of 2 strings are different.
    String-encoded images, after edits, should have different hashes.
    Send the result to debug output using Icecream library.
    This is a debugging helper function so it does not need to return other than None.
    """
    from icecream import ic
    hash1, hash2 = h(str1), h(str2)
    if hash1 != hash2:
        ic("String modified.")
        ic("1st string hash: ", hash1)
        ic("2nd string hash: ", hash2)
    elif hash1 == hash2:
        ic("String NOT modified.")
        ic("1st string hash: ", hash1)
        ic("2nd string hash: ", hash2)
    else:
        ic("Failure to check string difference.")


def string_from_imgfile(filename: str)->str:
    """
    Open a local image using PIL and convert into base64 encoded JPEG.
    """
    from icecream import ic
    from io import BytesIO
    import base64
    # Python Image Library
    from PIL import Image

    # Open the image and set color mode
    img = Image.open(filename)
    img = img.convert('RGB')

    # Create buffer and put JPEG version of image into it
    buffered = BytesIO()
    # Optimize file size and enforce JPEG format
    img.save(buffered, format='JPEG', optimize=True, quality=95)

    # Encode into base64 string
    img_str = base64.b64encode(buffered.getvalue())

    return img_str


def string_from_bytes(bytes)->str:
    """
    Open a local image using PIL and convert into base64 encoded JPEG.
    """
    from icecream import ic
    from io import BytesIO
    import base64
    # Python Image Library
    from PIL import Image

    # Create buffer object instance using input JPEG bytes, such as from open()
    buffered = BytesIO(bytes)

    # Open the image and set color mode
    img = Image.open(buffered)
    img = img.convert('RGB')

    # Optimize file size and enforce JPEG format
    img.save(buffered, format='JPEG', optimize=True, quality=95)
    img_str = base64.b64encode(buffered.getvalue())

    # Debug to terminal
    ic("Hash of b64 image string read from file : ", h(img_str))
    return img_str


def imgfile_from_string(img_str: str, writefile=False):
    """
    Bytes-like object is returned from image string.
    Optional: write to DISK with a random file name.
    """
    import base64
    from io import BytesIO
    from icecream import ic
    decoded_image = base64.b64decode(img_str)
    buffered = BytesIO(decoded_image)
    if writefile:
        from secrets import token_urlsafe
        from random import randint
        from PIL import Image
        # Generate random filename and debug print it to console
        rand_filename = token_urlsafe(randint(8,16)) + ".jpg"
        ic("Random filename: ", rand_filename)
        # Put the image data from the buffer into the PIL Image object
        img = Image.open(buffered)
        # Save to same location as opened file using random name
        img.save(rand_filename, 'jpeg')
        # Return the buffer for consistency with behavior when not writing to disk
        return buffered
    else:
        # Return the buffer with the image bytes decoded from the string
        return buffered





def draw_box_str(image_string) -> str:
    """
    Consumes base64 encoded image. Handles it like a file.
    Use both imaging libraries to draw a rectangle.
    Returns the filename as a string after writing a file.
    """
    # Create a file-like object called file from an encoded image
    base64_decoded = base64.b64decode(image_string)
    file = BytesIO(base64_decoded)

    pillow_image = Image.open(file)

    # Enforce RGB mode and JPEG in the PIL version
    pillow_image = pillow_image.convert('RGB')
    #pillow_image.save(pillow_image, format='JPEG')

    # Object that draws on the image
    draw = ImageDraw.Draw(pillow_image)

    # Outline thickness determined by image area.
    img_height, img_width = pillow_image.size
    area = img_height * img_width
    if area < 1000000.0:
        # 1px outline default for area under 1000x1000
        w = 1
    else:
        # Scale proportionally
        w  = int(area / 1000000.0)

    # Iterate through list of detected faces.
    i = 1
    for f in faces:  
        # Find upper left and lower right coordinates of face boxes for drawing borders.
        height, width, x, y = f.box.height, f.box.width, f.box.x, f.box.y
        start_point = tuple(int(f) for f in (x, y))
        end_point = tuple(int(f) for f in (x + width, y + height))
        
        # Outline colors. This wraps the rainbow color list defined up top.
        n = i % len(rainbow)
        color = rainbow[n]

        # Override rainbow for default green box.
        #color = green

        # Draw box on image and return new image object.
        shape = [start_point, end_point]
        draw.rectangle(shape, outline=color, width=w)
        ic("Drew box for face number: ", i)
        i += 1

    # Create new file-like object and enforce JPEG encoding
    im_file = BytesIO()
    pillow_image.save(im_file, format="jpeg")
    # create im_bytes and set to im_file in binary format for base64 encoding
    im_bytes = im_file.getvalue()
    new_image_string = base64.b64encode(im_bytes)

    # Compare hashes of input and output. Debug to console with Icecream ic() call.
    str_hash_diff(image_string, new_image_string)

    return new_image_string



def write_text_str(image_string, text="Detected") -> str:
    """
    Write the given text onto an image for each face.
    Start with numbering the faces.
    """
    # Create a file-like object called file from an encoded image
    base64_decoded = base64.b64decode(image_string)
    file = BytesIO(base64_decoded)

    # let's also use the Python Image Library (PIL / Pillow) for more fonts and some other tricks
    pillow_image = Image.open(file)

    # Enforce RGB mode and JPEG in the PIL version
    pillow_image = pillow_image.convert('RGB')
    pillow_image.save(pillow_image, format='JPEG')

    # Object that draws on the image
    draw = ImageDraw.Draw(pillow_image)

    # Font_scale value should be 10 for a 1000x1000 scale image or smaller.
    img_height, img_width = pillow_image.size
    font_size = 10.0
    # But we still have to scale the font by image area.
    area = img_height * img_width
    if area < 1000000.0:
        ttf_font = ImageFont.truetype('Roboto-Regular.ttf', font_size)
    else:
        font_size = int((area / 1000000.0) * font_size)
        ttf_font = ImageFont.truetype('Roboto-Regular.ttf', font_size)

    # Write text for each face.
    i = 1
    for f in faces:
        # Get box coordinates of face. (x,y) is top left corner.   
        box_height, x, y = f.box.height, f.box.x, f.box.y

        bottom_left = tuple(int(f) for f in (x, y + box_height))

        # Make it rainbow.
        n = i % len(rainbow)
        color = rainbow[n]

        # Remove comment for default green.
        # color = green

        # Number the faces by detection order as default text behavior, after added text.
        text = text + "face " + str(i)
        draw.text(bottom_left, text, color, font=ttf_font)
        i += 1

    # Create new file-like object and enforce JPEG encoding
    im_file = BytesIO()
    pillow_image.save(im_file, format="jpeg")
    # create im_bytes and set to im_file in binary format for base64 encoding
    im_bytes = im_file.getvalue()
    new_image_string = base64.b64encode(im_bytes)

    # Compare hashes of input and output. Debug to console with Icecream ic() call.
    str_hash_diff(image_string, new_image_string)

    return new_image_string