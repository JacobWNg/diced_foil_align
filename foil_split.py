from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.mixture import GaussianMixture
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
import random


def load_image(folder, image_file): 
    image_path = os.path.join(folder, image_file)
    output_folder = os.path.dirname(folder)
    output_folder = os.path.join(output_folder, "results")
    output_folder0 = os.path.join(output_folder, os.path.splitext(os.path.basename(image_file))[0])
    if not os.path.isdir(output_folder0):
        os.mkdir(output_folder0)
    output_folder1 = os.path.join(output_folder0, "split")
    if not os.path.isdir(output_folder1):
        os.mkdir(output_folder1)
    output_folder2 = os.path.join(output_folder0, "channels")
    if not os.path.isdir(output_folder2):
        os.mkdir(output_folder2)
    # output_folder2 = os.path.join(output_folder0, "diced")
    # if not os.path.isdir(output_folder2):
    #     os.mkdir(output_folder2)
    # output_folder3 = os.path.join(output_folder2, "ECA") # Busbar for frontside
    # if not os.path.isdir(output_folder3):
    #     os.mkdir(output_folder3)
        
    img_color = cv2.imread(str(image_path), 1)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    plt.imsave(os.path.join(output_folder0, f"0 Original.jpg"), img_color[:,:,::-1])

    return img_color, img_gray, output_folder0


def image_histogram(img_color, output_folder, scale_percent):

    width = int(img_color.shape[1] * scale_percent / 100)
    height = int(img_color.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img_color, dim, interpolation=cv2.INTER_AREA)

    # Convert to RGB and HSV
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Split channels
    r, g, b = cv2.split(img_rgb)
    h, s, v = cv2.split(img_hsv)

    # Save each channel
    plt.imsave(f'{output_folder}/channels/image_original.png', img[:,:,::-1])
    plt.imsave(f'{output_folder}/channels/image_gray.png', img_gray, cmap='gray')
    plt.imsave(f'{output_folder}/channels/image_red.png', r, cmap='Reds')
    plt.imsave(f'{output_folder}/channels/image_green.png', g, cmap='Greens')
    plt.imsave(f'{output_folder}/channels/image_blue.png', b, cmap='Blues')
    plt.imsave(f'{output_folder}/channels/image_hue.png', h, cmap='hsv')
    plt.imsave(f'{output_folder}/channels/image_saturation.png', s, cmap='gray')
    plt.imsave(f'{output_folder}/channels/image_value.png', v, cmap='gray')

    # Plot and save RGB histograms together
    plt.figure()
    colors = ('r', 'g', 'b')
    channels = (r, g, b)
    for ch, color in zip(channels, colors):
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
        hist = cv2.calcHist([ch], [0], None, [254], [1, 255])
        hist = hist / hist.max()
        plt.plot(hist, color=color)
        plt.xlim([1, 255])
    plt.title('RGB Histograms')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_folder}/channels/histogram_rgb.jpg')
    plt.close()

    # Plot and save HSV histograms separately
    hsv_channels = {'hue': h, 'saturation': s, 'value': v}
    for name, channel in hsv_channels.items():
        plt.figure()
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = cv2.calcHist([channel], [0], None, [254], [1, 255])
        hist = hist / hist.max()
        plt.plot(hist, color='black')
        plt.title(f'{name.capitalize()} Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.savefig(f'{output_folder}/channels/histogram_{name}.jpg')
        plt.close()

    return


def get_feature_dimensions(img_color, image_details):
    h, w, _ = img_color.shape
    if image_details["side"] == "daf":
        # Assume Ring lighting
        # Use center ECA as feature for threshold and grid
        # Aspect = 8.5 / 3.116 = 2.7279 
        # Offset = 1532
        feature_aspect, feature_offset = 2.7279, 1532
    elif image_details["side"] == "cell":
        if image_details["imaging"] == "ring":
            # Use busbar as feature for threshold and grid 
            # Aspect = 9.925 / 0.3 = 33.0833
            # Offset = 3062 micron
            feature_aspect, feature_offset = 33.0833, 3062
        elif image_details["imaging"] == "coax":
            # Use epi as feature for threshold and grid 
            # Aspect = 10.225 / 6.535 = 1.5647
            # Offset = 100 micron
            feature_aspect, feature_offset = 1.5647, -150
    return feature_aspect, feature_offset
    # Busbar area > 30,000
    # Epi area < 2,000,000


def threshold_cells(img_color, image_details):
    # Convert to RGB and HSV
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    # Split channels
    r, g, b = cv2.split(img_rgb)
    h, s, v = cv2.split(img_hsv)

    if image_details["side"] == "daf":
        otsu, threshold = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif image_details["side"] == "cell":
        if image_details["imaging"] == "ring":
            otsu, threshold = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif image_details["imaging"] == "coax":
            # Use saturation image, inverted
            otsu, threshold = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            threshold = cv2.bitwise_not(threshold)
    kernel = np.ones((21, 21), np.uint8) # determine size based on fingers?
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    plt.imsave(os.path.join(output_folder, f"1 Threshold.jpg"), threshold, cmap='gray')
    return threshold


# def threshold_cells(img_color, image_details):
#     if image_details["side"] == "daf":
#         _, threshold = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)
#         kernel = np.ones((21, 21), np.uint8) 
#         # threshold = cv2.erode(threshold, kernel)
#         threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
#         threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
#     elif image_details["imaging"] == "coax": # cell
#         _, threshold = cv2.threshold(img_color[:,:,2], 200, 255, cv2.THRESH_BINARY_INV) 
#         # TODO Coax vs Ring lighting
#     else: # ring
#         _, threshold = cv2.threshold(img_gray, 210, 255, cv2.THRESH_TOZERO_INV)
#         _, threshold = cv2.threshold(threshold, 150, 255, cv2.THRESH_BINARY)
#         kernel = np.ones((9, 9), np.uint8) 
#         threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
#     # plt.imshow(threshold, cmap='gray'), plt.show()
#     plt.imsave(os.path.join(output_folder, f"1. Threshold.jpg"), threshold, cmap='gray')
#     return threshold


# def get_feature_dimensions(img_color, image_details):
#     h, w, _ = img_color.shape
#     if image_details["side"] == "cell":
#         if image_details["imaging"] == "ring":
#             # use busbar for contours
        
#             # diced image (4") 50x = 16,500 x 16,500
#             # busbar_width = 1615 # 50x
#             # busbar_height = 60 # 50x
#             h_busbar = h / 330
#             w_busbar = w / 10
#             a_busbar = h_busbar * w_busbar
#             contour_feature = h_busbar, w_busbar, a_busbar
#             feature_offset = 500

#         if image_details["imaging"] == "coax":
#             # use epi for contours

#             # wafer onboard (4") 30x = 20,000 x 20,000 (w x h)
#             # cell_width = 2000 # 30x
#             # cell_height = 1250 # 30x
#             # cell_area = 2560000 30x
#             h_epi = h / 16 # Maybe 15
#             w_epi = w / 10
#             a_epi = h_epi * w_epi
#             contour_feature = h_epi, w_epi, a_epi
#             feature_offset = 0
        

#         # diced image (4") 50x = 16,500 x 16,500
#         # busbar_width = 1700 # 50x
#         # busbar_height = 1150 # 50x
#         h_cell = h / 14 # TODO
#         w_cell = w / 9.5 # TODO
#         a_cell = h_cell * w_cell
#         cell_dimensions = h_cell, w_cell, a_cell

#         # wafer onboard (6") 30x = 15,500 x 15,500
#     else:
#         if image_details["imaging"] == "vhx":
#             # diced image (4.5") 30x = 11,500 x 11,500
#             # ECA_width = 840
#             # ECA_height = 300
#             # ECA_area = 248000
#             # w_cell = 1030
#             # h_cell = 690

#             # diced image (4.5") 30x = 21,000 x 21,000
#             # ECA_width = 1675
#             # ECA_height = 600
#             # ECA_area = 1115000
#             # w_cell = 2050
#             # h_cell = 1370

#             # use central ECA for contours
#             h_ECA = h / 38 # 35-38
#             w_ECA = w / 13.5 # 12-14
#             a_ECA = h_ECA * w_ECA
#             contour_feature = h_ECA, w_ECA, a_ECA
#             feature_offset = 55

#             h_cell = h / 16.5
#             w_cell = w / 11
#             a_cell = h_cell * w_cell
#             cell_dimensions = h_cell, w_cell, a_cell

#         elif image_details["imaging"] == "dslr":
#             # 6000 x 4000
#             # ECA_width = 370
#             # ECA_height = 130
#             # ECA_area = 50500
#             # w_cell = 450
#             # h_cell = 300

#             h_ECA = 130
#             w_ECA = 370
#             a_ECA = h_ECA * w_ECA
#             contour_feature = h_ECA, w_ECA, a_ECA
#             feature_offset = 22

#             h_cell = 300
#             w_cell = 450
#             a_cell = h_cell * w_cell
#             cell_dimensions = h_cell, w_cell, a_cell

#     return contour_feature, feature_offset, cell_dimensions


def filter_contours(img_color, threshold, contour_feature):
    img_output = img_color.copy()
    h, w, _ = img_color.shape
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours (based on threshold)")
    filtered_contours = [cnt for cnt in contours if 300000 < cv2.contourArea(cnt) < 3000000] # TODO Ok to keep hardcoded?
    print(f"Filtered to {len(filtered_contours)} contours (based on size)")
    cv2.drawContours(img_output, filtered_contours, -1, (0, 255, 0), 14)
    # plt.imshow(img_output[:,:,::-1]), plt.show()
    plt.imsave(os.path.join(output_folder, f"2 Contours.jpg"), img_output[:,:,::-1])
    return filtered_contours

def box_cells(img_color, filtered_contours, feature_aspect):
    img_output = img_color.copy()
    angles = []
    centers = []
    # print(f"Height: {contour_feature[0]}")
    # print(f"Width: {contour_feature[1]}")
    for i, cell in enumerate(filtered_contours):
        rect = cv2.minAreaRect(cell)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        aspect = max(rect[1])/min(rect[1])
        if abs((aspect - feature_aspect) / feature_aspect) < 0.1:
            angle = rect[2] - 90 if rect[2] > 45 else rect[2]
            angles.append(angle)
            center = (int(rect[0][0]), int(rect[0][1]))
            centers.append(center)
        else: 
            continue
    print(f"Found {len(angles)} cells (based on contour aspect ratio)")
    total_rotation = sum(angles) / len(angles)
    print(f"Angle = {total_rotation:.3f} degrees")
    # plt.figure(figsize=(15, 15))
    # plt.imshow(img_output[:,:,::-1]), plt.show()
    return total_rotation, centers


def correct_angle(img_color, angle, centers):
    # Get image dimensions
    (h, w) = img_color.shape[:2]
    center = (w // 2, h // 2)  # Rotation center

    # Step 1: Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Step 2: Rotate the image
    rotated_image = cv2.warpAffine(img_color, rotation_matrix, (w, h))

    rotated_centers = []
    for center in centers:
        (rect_center_x, rect_center_y) = center
        rect_center = np.array([[rect_center_x, rect_center_y, 1]])  # Convert to homogeneous coordinates
        rotated_center = np.dot(rotation_matrix, rect_center.T).T  # Apply rotation matrix
        new_center = (rotated_center[0][0], rotated_center[0][1])  # Update angle
        rotated_centers.append(new_center)
    centers = rotated_centers

    return rotated_image, centers

def overlay_cell_grid(rotated_image, centers):

    grid_image = rotated_image.copy()

    # Step 1: Extract all x-coordinates
    x_coords = np.array([x for x, y in centers])
    y_coords = np.array([y for x, y in centers])

    # Step 1: Reshape for clustering (since KMeans expects 2D input)
    x_coords_reshaped = x_coords.reshape(-1, 1)
    y_coords_reshaped = y_coords.reshape(-1, 1)

    # Step 1: Reshape for clustering (since KMeans expects 2D input)
    x_coords_reshaped = x_coords.reshape(-1, 1)
    y_coords_reshaped = y_coords.reshape(-1, 1)

    # Step 2: Apply K-Means clustering
    num_x_groups = 8
    num_x_groups = 12
    num_y_groups = 13
    num_y_groups = 20
    # num_y_groups = 6

    kmeans_x = KMeans(n_clusters=num_x_groups, random_state=42, n_init=10)
    kmeans_y = KMeans(n_clusters=num_y_groups, random_state=42, n_init=10)

    kmeans_x.fit(x_coords_reshaped)
    kmeans_y.fit(y_coords_reshaped)

    # Step 3: Get cluster labels
    labels_x = kmeans_x.labels_
    labels_y = kmeans_y.labels_

    # Step 4: Group coordinates by their assigned cluster
    grouped_x = {i: [] for i in range(num_x_groups)}
    grouped_y = {i: [] for i in range(num_y_groups)}

    for x, label in zip(x_coords, labels_x):
        grouped_x[label].append(x)

    for y, label in zip(y_coords, labels_y):
        grouped_y[label].append(y)

    # Step 5: Compute mean for each group
    means_x = {i: np.mean(grouped_x[i]) for i in grouped_x}
    means_y = {i: np.mean(grouped_y[i]) for i in grouped_y}

    # Step 6: Sort means by their values for proper order
    sorted_means_x = sorted(means_x.values())
    sorted_means_y = sorted(means_y.values())
    avg_x_spacing = []
    avg_y_spacing = []
    for i in range(0, len(sorted_means_x)-1):
        spacing = sorted_means_x[i+1] - sorted_means_x[i]
        if spacing > 100 and spacing < 3000: # TODO
            avg_x_spacing.append(spacing)
    if avg_x_spacing:
        avg_x_spacing = sum(avg_x_spacing) / len(avg_x_spacing) # = 10525 microns
        scale_x = 10525 / avg_x_spacing
    else:
        raise Exception('Issue determining x spacing')

    for i in range(0, len(sorted_means_y)-1):
        spacing = sorted_means_y[i+1] - sorted_means_y[i]
        if spacing > 100 and spacing < 2000: # TODO
            avg_y_spacing.append(spacing)
    if avg_y_spacing:
        avg_y_spacing = sum(avg_y_spacing) / len(avg_y_spacing) # = 7025 microns
        scale_y = 7025 / avg_y_spacing
    else:
        raise Exception('Issue determining y spacing')
    if abs(scale_x - scale_y) < 0.1:
        scale = (scale_x + scale_y) / 2
        print(f"Calculated scale = {round(scale, 5)} microns per pixel")
    else:
        print(f"Discrepancy in x and y scale values: {scale_x} vs {scale_y}")
        scale = 0

    for x in sorted_means_x:
        for y in sorted_means_y:
            # y += 500 # TODO
            cv2.circle(grid_image, [int(x),int(y)], 50, (0, 255, 0), -1)  # Draw a filled circle
    # plt.figure(figsize=(10, 10))
    # plt.imshow(grid_image[:,:,::-1]), plt.show()
    plt.imsave(os.path.join(output_folder, f"3 Grid.jpg"), grid_image[:,:,::-1])

    return sorted_means_x, sorted_means_y, scale


def singulate_cells(rotated_image, sorted_means_x, sorted_means_y, feature_offset, scale): 
    cells = []
    cell_x = 10.525*1000/scale
    cell_y = 7.025*1000/scale
    for y in sorted_means_y:
        y += int(feature_offset/scale)
        for x in sorted_means_x:
            singulated_cell = rotated_image[int(y-cell_y/1.8):int(y+cell_y/1.8), int(x-cell_x/1.8):int(x+cell_x/1.8)]
            cells.append(singulated_cell)
            # plt.imshow(singulated_cell[:,:,::-1]), plt.show()

    # i = random.choice([i for i in range(0,len(cells)) if i not in non_cells])
    # print(cell_IDs[i])
    # plt.imshow(cells[i][:,:,::-1]), plt.show()
    
    return cells


def save_images(cells, output_folder):

    non_cells = [1, 2, 3, 7, 8, 9, 10, 16, 17, 24, 73, 80, 81, 88, 89, 90, 95, 96] # edges or test structures
    non_cells = []
    cell_IDs = [
        "05D", "05E", "05F", "05G", "05H", "05I", "05J", "05K",
        "06D", "06E", "06F", "06G", "06H", "06I", "06J", "06K",
        "07D", "07E", "07F", "07G", "07H", "07I", "07J", "07K",
        "08D", "08E", "08F", "08G", "08H", "08I", "08J", "08K",
        "09D", "09E", "09F", "09G", "09H", "09I", "09J", "09K",
        "10D", "10E", "10F", "10G", "10H", "10I", "10J", "10K",
        "11D", "11E", "11F", "11G", "11H", "11I", "11J", "11K",
        "12D", "12E", "12F", "12G", "12H", "12I", "12J", "12K",
        "13D", "13E", "13F", "13G", "13H", "13I", "13J", "13K",
        "14D", "14E", "14F", "14G", "14H", "14I", "14J", "14K",
        "15D", "15E", "15F", "15G", "15H", "15I", "15J", "15K",
        "16D", "16E", "16F", "16G", "16H", "16I", "16J", "16K",
        "17D", "17E", "17F", "17G", "17H", "17I", "17J", "17K",
    ]
    cell_IDs = [
        "01B", "01C", "01D", "01E", "01F", "01G", "01H", "01I", "01J", "01K", "01L", "01M",
        "02B", "02C", "02D", "02E", "02F", "02G", "02H", "02I", "02J", "02K", "02L", "02M",
        "03B", "03C", "03D", "03E", "03F", "03G", "03H", "03I", "03J", "03K", "03L", "03M",
        "04B", "04C", "04D", "04E", "04F", "04G", "04H", "04I", "04J", "04K", "04L", "04M",
        "05B", "05C", "05D", "05E", "05F", "05G", "05H", "05I", "05J", "05K", "05L", "05M",
        "06B", "06C", "06D", "06E", "06F", "06G", "06H", "06I", "06J", "06K", "06L", "06M",
        "07B", "07C", "07D", "07E", "07F", "07G", "07H", "07I", "07J", "07K", "07L", "07M",
        "08B", "08C", "08D", "08E", "08F", "08G", "08H", "08I", "08J", "08K", "08L", "08M",
        "09B", "09C", "09D", "09E", "09F", "09G", "09H", "09I", "09J", "09K", "09L", "09M",
        "10B", "10C", "10D", "10E", "10F", "10G", "10H", "10I", "10J", "10K", "10L", "10M",
        "11B", "11C", "11D", "11E", "11F", "11G", "11H", "11I", "11J", "11K", "11L", "11M",
        "12B", "12C", "12D", "12E", "12F", "12G", "12H", "12I", "12J", "12K", "12L", "12M",
        "13B", "13C", "13D", "13E", "13F", "13G", "13H", "13I", "13J", "13K", "13L", "13M",
        "14B", "14C", "14D", "14E", "14F", "14G", "14H", "14I", "14J", "14K", "14L", "14M",
        "15B", "15C", "15D", "15E", "15F", "15G", "15H", "15I", "15J", "15K", "15L", "15M",
        "16B", "16C", "16D", "16E", "16F", "16G", "16H", "16I", "16J", "16K", "16L", "16M",
        "17B", "17C", "17D", "17E", "17F", "17G", "17H", "17I", "17J", "17K", "17L", "17M",
        "18B", "18C", "18D", "18E", "18F", "18G", "18H", "18I", "18J", "18K", "18L", "18M",
        "19B", "19C", "19D", "19E", "19F", "19G", "19H", "19I", "19J", "19K", "19L", "19M",
        "20B", "20C", "20D", "20E", "20F", "20G", "20H", "20I", "20J", "20K", "20L", "20M",
    ]

    for index, cell in enumerate(cells): 
        if index+1 in non_cells:    
            continue
        elif index > len(cell_IDs): 
            break        
        else:
            plt.imsave(os.path.join(output_folder, "split", f"{cell_IDs[index]}.jpg"), cell[:,:,::-1])
    print("Wafer split into cell images and saved")



image_folder = Path(r"C:\Users\Public\Python-scripts\diced_foil_align_JN\images")
# image_folder = Path(r"G:\Shared drives\TPV\Cell & Wafer\Characterization\Characterization Reports\D. RW Lots\2025\RW250110\1) Post Processing Data\VHX")
image_file = "RW240522-W10 (2_0767-2) Au bottom_50x_externalring.jpg"
image_file = "RW250110 W9, 30X.jpg"
side = "Cell" # Cell or DAF
imaging = "Coax" # Ring or Coax. Assumes VHX
metal = "Gold" # Gold or Nickel
diced = False # True or False

image_details = {
    "side": side.lower(),
    "imaging": imaging.lower(),
    "metal": metal.lower(),
    "diced": diced,
}

img_color, img_gray, output_folder = load_image(image_folder, image_file)
# image_histogram(img_color, output_folder, 10)

if len(os.listdir(os.path.join(output_folder, "split"))) == 0:
    feature_aspect, feature_offset = get_feature_dimensions(img_color, image_details)
    threshold = threshold_cells(img_color, image_details)
    filtered_contours = filter_contours(img_color, threshold, feature_aspect)
    angle, centers = box_cells(img_color, filtered_contours, feature_aspect)
    rotated_image, centers = correct_angle(img_color, angle, centers)
    sorted_means_x, sorted_means_y, scale = overlay_cell_grid(rotated_image, centers)
    cells = singulate_cells(rotated_image, sorted_means_x, sorted_means_y, feature_offset, scale)
    save_images(cells, output_folder)

if diced:
    print("Skipped to here")
    # scale = 6.17244 # top
    scale = 6.15735 # bottom 
    folder = os.path.join(output_folder, "split")
    output_folder = os.path.join(folder, "channels")
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            file_path = os.path.join(folder, filename)
            name = os.path.splitext(filename)[0]
            print(name)
            img = cv2.imread(file_path)
            image_histogram(img, folder, 100)
            break
    for filename in os.listdir(output_folder):
        if filename.endswith('.png'):
            file_path = os.path.join(output_folder, filename)
            name = os.path.splitext(filename)[0]
            img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # Check Otsu threshold output
            otsu, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            print(name, otsu)
            # plt.imshow(threshold, cmap="gray")
            plt.imsave(f'{output_folder}/threshold_{name}.jpg', threshold, cmap='gray')

    # TODO Try Canny edge detect
    # TODO Try Close before threshold (on Red channel)