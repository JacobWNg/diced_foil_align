from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import random
import csv


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
    output_folder2 = os.path.join(output_folder0, "diced")
    if not os.path.isdir(output_folder2):
        os.mkdir(output_folder2)
    output_folder3 = os.path.join(output_folder2, "ECA")
    if not os.path.isdir(output_folder3):
        os.mkdir(output_folder3)
        
    img_color = cv2.imread(str(image_path), 1)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    plt.imsave(os.path.join(output_folder0, f"0 Original.jpg"), img_color[:,:,::-1])

    if False:
        plt.subplot(1, 2, 1)
        plt.imshow(img_color[:,:,::-1])  
        plt.axis('off')
        plt.title("Image")
        plt.subplot(1, 2, 2)
        plt.imshow(img_gray, cmap='gray')
        plt.axis('off')
        plt.title("Grayscale")
        plt.show()

    return img_color, img_gray, output_folder0


def image_histogram(img_color, img_gray):
    # Determine if cell-side or DAF-side
    mask = np.all((img_color > 0) & (img_color < 255), axis=-1)
    pixels_color = img_color[mask]
    mask = (img_gray > 0) & (img_gray < 255)
    pixels_gray = img_gray[mask]

    # Plot histograms for each channel
    colors = ('red', 'green', 'blue')

    plt.figure(figsize=(12, 6))

    histogram = cv2.calcHist([pixels_gray], [0], None, [256], [0, 256])
    normalized_histogram = histogram / histogram.sum()
    plt.plot(normalized_histogram, color="black", label="grayscale")

    normalized_histograms = {}  # Dictionary to store normalized histograms

    for i, color in enumerate(colors):
        histogram, bin_edges = np.histogram(pixels_color[:, i], bins=256, range=(0, 256))
        # histogram = cv2.calcHist(pixels, [i], None, [256], [0, 256])
        normalized_histogram = histogram / histogram.sum()
        normalized_histograms[color] = normalized_histogram  # Store for later use
        plt.plot(normalized_histogram, color=color, label=color)

    if False:
        # Configure plot
        plt.title('Color Histogram')
        plt.xlabel('Pixel Intensity (0-255)')
        # plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(r"C:\Users\Public\Python-scripts\diced_foil_align\histograms\histogram.png", format='png', dpi=300)
        # plt.show()

    histogram, bin_edges = np.histogram(pixels_gray, bins=254, range=(1, 255))
    peak_index = np.argmax(histogram)
    peak_intensity = bin_edges[peak_index]

    if peak_index > 250: # for diced cell ring lighting
        side = "cell"
        imaging = "ring"
        metal = None
    elif  peak_index > 100: # for onboard cell coax lighting
        side = "cell"
        imaging = "coax"
        metal = None
    elif normalized_histograms['blue'][240] > 0.01:
        side = "DAF"
        imaging = "VHX"
        metal = "Gold"
    else:
        side = "DAF"
        imaging = "VHX"
        metal = "Nickel"

    image_details = {
        "side": side.lower(),
        "imaging": imaging.lower(),
        "metal": metal.lower(),
    }
    
    print(f"Image is {side}-side, {imaging} imaging conditions, {metal} backmetal")
    return image_details


def threshold_cells(img_color, image_details):
    if image_details["side"] == "daf":
        if image_details["imaging"] == "vhx":
            ret, threshold = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
        # if image_details["metal"] == "gold":
        #     ret, threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        # else: 
        #     ret, threshold = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY) # brighter cutoff for Ni
        elif image_details["imaging"] == "dslr":
            ret, threshold = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY)
        kernel = np.ones((21, 21), np.uint8) 
        # threshold = cv2.erode(threshold, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    elif image_details["imaging"] == "coax": # cell
        ret, threshold = cv2.threshold(img_color[:,:,2], 200, 255, cv2.THRESH_BINARY_INV) 
        # ret, threshold = cv2.threshold(img_color[:,:,2], 150, 255, cv2.THRESH_BINARY_INV) 
        # TODO Coax vs Ring lighting
        kernel = np.ones((9, 9), np.uint8) 
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    else: # ring
        ret, threshold = cv2.threshold(img_gray, 210, 255, cv2.THRESH_TOZERO_INV)
        ret, threshold = cv2.threshold(threshold, 150, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9, 9), np.uint8) 
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    # plt.imshow(threshold, cmap='gray'), plt.show()
    plt.imsave(os.path.join(output_folder, f"1 Threshold.jpg"), threshold, cmap='gray')
    return threshold

def get_feature_dimensions(img_color, image_details):
    h, w, _ = img_color.shape
    if image_details["side"] == "cell":
        if image_details["imaging"] == "ring":
            # use busbar for contours
        
            # diced image (4") 50x = 16,500 x 16,500
            # busbar_width = 1615 # 50x
            # busbar_height = 60 # 50x
            h_busbar = h / 330
            w_busbar = w / 10
            a_busbar = h_busbar * w_busbar
            contour_feature = h_busbar, w_busbar, a_busbar
            # feature_offset = 500
            feature_offset = 600

        if image_details["imaging"] == "coax":
            # use epi for contours

            # wafer onboard (4") 30x = 20,000 x 20,000 (w x h)
            # cell_width = 2000 # 30x
            # cell_height = 1250 # 30x
            # cell_area = 2560000 30x
            h_epi = h / 16 # Maybe 15
            w_epi = w / 10
            a_epi = h_epi * w_epi
            contour_feature = h_epi, w_epi, a_epi
            feature_offset = 0
        

        # diced image (4") 50x = 16,500 x 16,500
        # busbar_width = 1700 # 50x
        # busbar_height = 1150 # 50x
        h_cell = h / 14 # TODO
        w_cell = w / 9.5 # TODO
        a_cell = h_cell * w_cell
        cell_dimensions = h_cell, w_cell, a_cell

        # wafer onboard (6") 30x = 15,500 x 15,500
    else:
        if image_details["imaging"] == "vhx":
            # diced image (4.5") 30x = 11,500 x 11,500
            # ECA_width = 840
            # ECA_height = 300
            # ECA_area = 248000
            # w_cell = 1030
            # h_cell = 690

            # diced image (4.5") 30x = 21,000 x 21,000
            # ECA_width = 1675
            # ECA_height = 600
            # ECA_area = 1115000
            # w_cell = 2050
            # h_cell = 1370

            # use central ECA for contours
            h_ECA = h / 38 # 35-38
            w_ECA = w / 13.5 # 12-14
            a_ECA = h_ECA * w_ECA
            contour_feature = h_ECA, w_ECA, a_ECA
            feature_offset = 55

            h_cell = h / 16.5
            w_cell = w / 11
            a_cell = h_cell * w_cell
            cell_dimensions = h_cell, w_cell, a_cell

        elif image_details["imaging"] == "dslr":
            # 6000 x 4000
            # ECA_width = 370
            # ECA_height = 130
            # ECA_area = 50500
            # w_cell = 450
            # h_cell = 300

            h_ECA = 130
            w_ECA = 370
            a_ECA = h_ECA * w_ECA
            contour_feature = h_ECA, w_ECA, a_ECA
            feature_offset = 22

            h_cell = 300
            w_cell = 450
            a_cell = h_cell * w_cell
            cell_dimensions = h_cell, w_cell, a_cell

    return contour_feature, feature_offset, cell_dimensions


def filter_contours(img_color, threshold, contour_feature):
    img_output = img_color.copy()
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")
    filtered_contours = [cnt for cnt in contours if 0.8*contour_feature[2] < cv2.contourArea(cnt) < 1.2*contour_feature[2]]
    print(f"Filtered to {len(filtered_contours)} contours")
    cv2.drawContours(img_output, filtered_contours, -1, (0, 255, 0), 14)
    # plt.imshow(img_output[:,:,::-1]), plt.show()
    plt.imsave(os.path.join(output_folder, f"2 Contours.jpg"), img_output[:,:,::-1])
    return filtered_contours

def box_cells(img_color, filtered_contours, contour_feature):
    img_output = img_color.copy()
    angles = []
    centers = []
    # print(f"Height: {contour_feature[0]}")
    # print(f"Width: {contour_feature[1]}")
    for i, cell in enumerate(filtered_contours):
        rect = cv2.minAreaRect(cell)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        # print(f"Contour width: {rect[1][0]}")
        # print(f"Contour height: {rect[1][1]}")
        if (0.9*contour_feature[0] < rect[1][0] < 1.1*contour_feature[0]) & (0.9*contour_feature[1] < rect[1][1] < 1.1*contour_feature[1]):
            # Valid cell
            angle = rect[2] - 90 if rect[2] > 45 else rect[2]
            angles.append(angle)
            # cv2.drawContours(img_output, [box], 0, (0, 255, 0), 20)
            center = (int(rect[0][0]), int(rect[0][1]))  # rect[0] gives the center coordinates
            # cv2.circle(img_output, center, 25, (0, 255, 0), -1)  # Draw a filled circle
        elif (0.9*contour_feature[0] < rect[1][1] < 1.1*contour_feature[0]) & (0.9*contour_feature[1] < rect[1][0] < 1.1*contour_feature[1]):
            # Valid cell (rotated sideways)
            angle = rect[2] - 90 if rect[2] > 45 else rect[2]
            angles.append(angle)
            # cv2.drawContours(img_output, [box], 0, (0, 255, 0), 20)
            center = (int(rect[0][0]), int(rect[0][1]))  # rect[0] gives the center coordinates
            # cv2.circle(img_output, center, 25, (0, 255, 0), -1)  # Draw a filled circle
        else:
            continue
        centers.append(center)
    print(f"Found {len(angles)} cells")
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

def overlay_cell_grid(rotated_image, centers, image_details):

    grid_image = rotated_image.copy()

    # Step 1: Extract all x,y-coordinates
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
    num_y_groups = 12
    if image_details["imaging"] == "ring":
        num_y_groups = 13

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
    avg_x_spacing = sum(avg_x_spacing) / len(avg_x_spacing) # = 10525 microns
    scale_x = 10525 / avg_x_spacing

    for i in range(0, len(sorted_means_y)-1):
        spacing = sorted_means_y[i+1] - sorted_means_y[i]
        if spacing > 100 and spacing < 2000: # TODO
            avg_y_spacing.append(spacing)
    avg_y_spacing = sum(avg_y_spacing) / len(avg_y_spacing) # = 7025 microns
    scale_y = 7025 / avg_y_spacing
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

    return sorted_means_x,sorted_means_y, scale

def singulate_cells(rotated_image, cell_dimensions, sorted_means_x, sorted_means_y, feature_offset): 
    cells = []
    for y in sorted_means_y:
        y += feature_offset # TODO
        for x in sorted_means_x:
            singulated_cell = rotated_image[int(y-cell_dimensions[0]/1.95):int(y+cell_dimensions[0]/1.95), int(x-cell_dimensions[1]/1.95):int(x+cell_dimensions[1]/1.95)]
            # TODO Adjust offset
            cells.append(singulated_cell)
            # plt.imshow(singulated_cell[:,:,::-1]), plt.show()

    # i = random.choice([i for i in range(0,len(cells)) if i not in non_cells])
    # print(cell_IDs[i])
    # plt.imshow(cells[i][:,:,::-1]), plt.show()
    
    return cells


def save_images(cells, non_cells, cell_IDs, output_folder):
    for index, cell in enumerate(cells): 
        if index+1 in non_cells:    
            continue
        elif index > 8*12-1: 
            break        
        else:
            cv2.putText(cell, cell_IDs[index], (480,310), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            plt.imsave(os.path.join(output_folder, "split", f"{cell_IDs[index]}.jpg"), cell[:,:,::-1])
        if index == 18:
            b,g,r = cv2.split(cell)
            cell_hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(cell)
            name = ["blue", "green", "red", "hue", "saturation", "value"]
            maps = ["Blues", "Greens", "Reds", "hsv", "Greys", "Greys"]
            for i, component in enumerate([b,g,r,h,s,v]):
                plt.imsave(os.path.join(output_folder, "split", f"{cell_IDs[index]}_{name[i]}.jpg"), component, cmap=maps[i])
    print("Wafer split into cell images and saved")

def find_dicing_lines(cells, image_details, non_cells, cell_IDs, cell_dimensions, scale, output_folder):
    dimensions = []
    diced_cells = []
    cell_x = 10525
    cell_y = 7025
    for index, cell in enumerate(cells): 
        if index+1 in non_cells:    
            continue
        elif index > 100: 
            continue        
        else:
            # print(cell_IDs[index])
            output_image = cells[index].copy()
            b,g,r = cv2.split(cell)
            cell_hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            _,_,v = cv2.split(cell)

            # # Apply thresholding
            # if image_details["side"] == "cell":
            #     ret, threshold = cv2.threshold(b, 102, 255, cv2.THRESH_BINARY_INV)
            #     # plt.imshow(threshold, cmap='gray'), plt.show()
            #     kernel = np.ones((5, 5), np.uint8) 
            #     threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            #     kernel = np.ones((31, 31), np.uint8) 
            #     threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            #     inverse = cv2.bitwise_not(threshold)
            # if image_details["side"] == "daf":
            #     ret, threshold = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)
            #     # plt.imshow(threshold, cmap='gray'), plt.show()
            #     kernel = np.ones((21, 21), np.uint8) # TODO
            #     threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            #     # plt.imshow(threshold, cmap='gray'), plt.show()
            #     kernel = np.ones((11, 11), np.uint8) 
            #     threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            #     # plt.imshow(threshold, cmap='gray'), plt.show()
            #     inverse = cv2.bitwise_not(threshold)
            # # plt.imshow(inverse, cmap='gray'), plt.show()
            # # plt.imsave(os.path.join(output_folder, "diced", f"{cell_IDs[index]}_1_mask.jpg"), inverse, cmap="gray")

            # contours, _ = cv2.findContours(inverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # if len(contours) > 0:
            #     contour_cell = max(contours, key=cv2.contourArea)
            #     rect = cv2.minAreaRect(contour_cell)
            #     box = cv2.boxPoints(rect)
            #     box = np.intp(box)
            #     center, (width, height), angle = rect
            #     # if (max(width,height) < 1.05*cell_dimensions[1]) and (max(width,height) > 0.95*cell_dimensions[1]) and (min(width,height) < 1.05*cell_dimensions[0]) and (min(width,height) > 0.95*cell_dimensions[0]):
            #     diced = False
            #     if (max(width,height)*scale < 1.01*cell_x) and (max(width,height)*scale > 0.9*cell_x) and (min(width,height)*scale < 1.01*cell_y) and (min(width,height)*scale > 0.9*cell_y):
            #         diced = True
            #         cv2.drawContours(output_image, contour_cell, -1, (0,255,0), 3)
            #         # plt.imshow(output_image[:,:,::-1]), plt.show()
            #         # plt.imsave(os.path.join(output_folder, "diced", f"{cell_IDs[index]}_2_annotate.jpg"), output_image[:,:,::-1])
            #     else:
            #         print(f"{cell_IDs[index]}: Something wrong with thresholding or contour dimensions")
            #         inverse = cv2.cvtColor(inverse, cv2.COLOR_GRAY2BGR)
            #         blended = cv2.addWeighted(cell, 0.5, inverse, 0.5, 0)
            #         cv2.drawContours(blended, contour_cell, -1, (0,255,0), 2)
            #         plt.imsave(os.path.join(output_folder, "diced", f"{cell_IDs[index]}_mask.jpg"), blended[:,:,::-1])

            # if diced:
            #     # print(angle)
            #     if height > width:
            #         angle -= 90

            #     rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)

            #     rotated_image = cv2.warpAffine(cell, rot_matrix, (cell.shape[1], cell.shape[0]))

            #     margin = 10
            #     W = max(rect[1])
            #     H = min(rect[1])
            #     # print(f"Cell dimensions: Height={int(H*scale)} µm, Width={int(W*scale)} µm")
            #     dict = {"Cell" : cell_IDs[index], "Height" : int(H*scale), "Width" : int(W*scale)}
            #     dimensions.append(dict)
            #     # Xs = [i[0] for i in box]
            #     # Ys = [i[1] for i in box]
            #     # x1 = min(Xs)
            #     # x2 = max(Xs)
            #     # y1 = min(Ys)
            #     # y2 = max(Ys)

            #     cropped_rotated_image = cv2.getRectSubPix(rotated_image, (int(W)+margin, int(H)+margin), center)
            #     # plt.imshow(cropped_rotated_image[:,:,::-1]), plt.show()
            #     diced_cells.append([cell_IDs[index], cropped_rotated_image])
            #     plt.imsave(os.path.join(output_folder, "diced", f"{cell_IDs[index]}.jpg"), cropped_rotated_image[:,:,::-1])
    
            dict = {"Cell" : cell_IDs[index], "Height" : cell_y, "Width" : cell_x}
            dimensions.append(dict)
            diced_cells.append([cell_IDs[index], cell])
            plt.imsave(os.path.join(output_folder, "diced", f"{cell_IDs[index]}.jpg"), cell[:,:,::-1])

    fieldnames = dimensions[0].keys()
    output_file = os.path.join(output_folder, "diced", "dimensions.csv")
    # del open
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dimensions)

    # TODO
    return dimensions, diced_cells

def find_busbar():
    return


def find_ECA_shingle(cells, scale, dimensions):
    nominal_cell_height = int(7025 / scale) # in pixels
    nominal_ECA_height = int(300 / scale) # in pixels # TODO
    nominal_ECA_width = int(8800 / scale) # in pixels # TODO
    start_y = nominal_cell_height - 3*nominal_ECA_height
    for index, cell in enumerate(cells): 
        ID = cell[0]
        img = cell[1]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_hsv)
        drawing = img.copy()
        # plt.imshow(drawing), plt.show()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray[s > 80] = 0
        # plt.imshow(gray), plt.show()
        gray = gray[start_y:,:]
        ret, threshold = cv2.threshold(gray, 102, 255, cv2.THRESH_BINARY)

        kernel = np.ones((9, 9), np.uint8) 
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        # plt.imshow(threshold, cmap='gray'), plt.show()
        kernel = np.ones((9, 9), np.uint8) 
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        # plt.imshow(threshold, cmap='gray'), plt.show()
        # plt.imsave(os.path.join(output_folder, "diced", "ECA", f"{cell_IDs[index]}_threshold.jpg"), threshold, cmap="gray")

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_ECA = max(contours, key=cv2.contourArea)
        contour_ECA = contour_ECA + np.array([[[0, start_y]]])
        cv2.drawContours(drawing, contour_ECA, -1, (0, 255, 0), 1)
        rect = cv2.minAreaRect(contour_ECA)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        # box[:, 1] += start_y  # add to the y-coordinates
        center, (width, height), angle = rect
        if max(width, height) > 1.1*nominal_ECA_width:
            print(f"{ID} ECA strip detected too wide: {int(max(width, height)*scale)} microns")
            plt.imsave(os.path.join(output_folder, "diced", "ECA", f"{ID}.jpg"), threshold, cmap="gray")
            continue
        cv2.drawContours(drawing, [box], 0, (0, 0, 255), 1)
        # print(f"{cell_IDs[index]} ECA shingle height: {int(min(width, height)*scale)}")
        for entry in dimensions:
            if entry['Cell'] == ID:
                entry['ECA_height_max'] = int(min(width, height)*scale)
                entry['ECA_height_avg'] = int(cv2.contourArea(contour_ECA) / max(width, height) * scale)
                break
        # plt.imshow(drawing), plt.show()
        plt.imsave(os.path.join(output_folder, "diced", "ECA", f"{ID}.jpg"), drawing[:,:,::-1])

    fieldnames = dimensions[0].keys()
    output_file = os.path.join(output_folder, "diced", "ECA", "dimensions.csv")
    # del open
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dimensions)
    return


image_folder = Path(r"C:\Users\Public\Python-scripts\diced_foil_align\images")
files = ["Au DAF foil Germany 17, 30x ring.jpg"]
side = "DAF" # DAF or Cell
imaging = "VHX" # VHX or Coax or Ring
metal = "Gold" # Gold or Nickel

# scale = 1000/218.875 # um/px # 1 mm = 1000 um = 218.875 px

image_details = {
    "side": side.lower(),
    "imaging": imaging.lower(),
    "metal": metal.lower(),
}

non_cells = [1, 2, 3, 7, 8, 9, 10, 16, 17, 24, 73, 80, 81, 88, 89, 90, 95, 96] # edges or test structures
# ejected = [5, 6, 13, 14, 15, 21, 22, 23, 29, 30, 31, 32, 37, 38, 39, 40]
# non_cells += ejected
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
    "16D", "16E", "16F", "16G", "16H", "16I", "16J", "16K"
]

for image_file in files:
    test = os.path.join(image_folder, image_file)
    if os.path.isfile(test):
        img_color, img_gray, output_folder = load_image(image_folder, image_file)
        # image_details = image_histogram(img_color, img_gray)

        if True:
            contour_feature, feature_offset, cell_dimensions = get_feature_dimensions(img_color, image_details)
        if True:
            threshold = threshold_cells(img_color, image_details)
            filtered_contours = filter_contours(img_color, threshold, contour_feature)
            angle, centers = box_cells(img_color, filtered_contours, contour_feature)
            rotated_image, centers = correct_angle(img_color, angle, centers)
            sorted_means_x,sorted_means_y, scale = overlay_cell_grid(rotated_image, centers, image_details)
            cells = singulate_cells(rotated_image, cell_dimensions, sorted_means_x, sorted_means_y, feature_offset)
            save_images(cells, non_cells, cell_IDs, output_folder)
            dimensions, diced_cells = find_dicing_lines(cells, image_details, non_cells, cell_IDs, cell_dimensions, scale, output_folder)
            if image_details["side"] == "daf":
                find_ECA_shingle(diced_cells, scale, dimensions)    
    else:
        print(f"Skipping {image_file}")