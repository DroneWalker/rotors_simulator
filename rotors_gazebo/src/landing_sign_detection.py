import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn import linear_model
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import pdist, squareform


def plot_figure_single(switch, image_data, title):

    plt.figure(switch)
    plt.imshow(image_data, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    return

def plot_figure_multiple(switch, image_data_1, image_data_2, title_1, title_2):

    plt.figure(switch)
    plt.subplot(1, 2, 1), plt.imshow(image_data_1, cmap='gray')
    plt.title(title_1), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(image_data_2, cmap='gray')
    plt.title(title_2), plt.xticks([]), plt.yticks([])
    return

def linear_regression_vertical(data):
    X = []
    Y = []
    data_fit = []

    for point in data:
        x_i = point[0]
        y_i = point[1]
        X.append([x_i])
        Y.append(y_i)

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    Y_new = regr.predict(X)

    for i, point in enumerate(data):
        new_point = []
        new_point.append(int(point[0]))
        new_point.append(int(Y_new[i]))
        data_fit.append(new_point)

    return data_fit

def linear_regression_horizontal(data):
    X = []
    Y = []
    data_fit = []

    for point in data:
        x_i = point[1]
        y_i = point[0]
        X.append([x_i])
        Y.append(y_i)

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    Y_new = regr.predict(X)

    for i, point in enumerate(data):
        new_point = []
        new_point.append(int(Y_new[i]))
        new_point.append(int(point[1]))
        data_fit.append(new_point)

    return data_fit

def edge_line_drawer(gray, lines_vertical_fitted, lines_horizontal_fitted):

    data_edge_x = []
    data_edge_y = []
    edge_array_y = int(255) * np.ones(shape=(len(gray), len(gray[0])))
    edge_array_x = int(255) * np.ones(shape=(len(gray), len(gray[0])))
    all_bright = int(255) * np.ones(shape=(len(gray), len(gray[0])))
    edge_image_y = Image.new('L', (len(gray[0]), len(gray)))
    edge_image_x = Image.new('L', (len(gray[0]), len(gray)))

    # fill the vertical lines
    for line in lines_vertical_fitted:
        for dot in line:
            dot_x = dot[0]
            dot_y = dot[1]
            edge_array_y[dot_x, dot_y] = int(0.0)

    # fill the horizontal lines
    for line in lines_horizontal_fitted:
        for dot in line:
            dot_x = dot[0]
            dot_y = dot[1]
            edge_array_x[dot_x, dot_y] = int(0.0)

    for i in edge_array_y:
        data_edge_y = np.concatenate([data_edge_y, i])

    for i in edge_array_x:
        data_edge_x = np.concatenate([data_edge_x, i])

    edge_image_y.putdata(data_edge_y)
    edge_image_x.putdata(data_edge_x)
    # edge_image_y.save(output_edge_vertical_filename)
    # edge_image_x.save(output_edge_horizontal_filename)
    edge_image_y = np.asarray(edge_image_y)
    edge_image_x = np.asarray(edge_image_x)

    image_overlap = edge_image_x + edge_image_y
    edge_image_x_invt = all_bright - edge_image_x
    edge_image_y_invt = all_bright - edge_image_y
    image_overlap_invt = all_bright - image_overlap
    edge_image_total_invt = edge_image_x_invt + edge_image_y_invt - image_overlap_invt
    edge_image_total = all_bright - edge_image_total_invt

    return edge_image_x, edge_image_y, edge_image_total

def gradient_search(switch_1, switch_2, switch_3, switch_4, grad,
                    anchor_data, num_point_up_limit, num_point_low_limit, x_dim, y_dim):

    lines_vertical = []
    lines_horizontal = []
    points_vertical_line = []
    points_horizontal_line = []

    for anchor_point in anchor_data:

        new_vertical_line = []
        new_horizontal_line = []
        new_vertical_line.append(anchor_point)
        new_horizontal_line.append(anchor_point)
        # Location of the current anchor point
        i_pos = anchor_point[0]
        j_pos = anchor_point[1]
        # The gray level of the eight neighbors of current anchor point
        pixel_1_level = grad[i_pos - 1, j_pos - 1]
        pixel_1 = [i_pos - 1, j_pos - 1]

        pixel_2_level = grad[i_pos - 1, j_pos]
        pixel_2 = [i_pos - 1, j_pos]

        pixel_3_level = grad[i_pos - 1, j_pos + 1]
        pixel_3 = [i_pos - 1, j_pos + 1]

        pixel_4_level = grad[i_pos, j_pos - 1]
        pixel_4 = [i_pos, j_pos - 1]

        pixel_6_level = grad[i_pos, j_pos + 1]
        pixel_6 = [i_pos, j_pos + 1]

        pixel_7_level = grad[i_pos + 1, j_pos - 1]
        pixel_7 = [i_pos + 1, j_pos - 1]

        pixel_8_level = grad[i_pos + 1, j_pos]
        pixel_8 = [i_pos + 1, j_pos]

        pixel_9_level = grad[i_pos + 1, j_pos + 1]
        pixel_9 = [i_pos + 1, j_pos + 1]

        if switch_2 == 1:
            # Search downward (using pixel 7,8,9)
            downward_stop = int(pixel_7_level) + int(pixel_8_level) + int(pixel_9_level)
            while (not downward_stop == 0):
                count_9 = 0
                count_7 = 0
                if len(new_vertical_line) < num_point_up_limit:
                    if (pixel_7 in anchor_data) == True:
                        if (pixel_7 in points_vertical_line) is True or (pixel_7 in points_horizontal_line) is True:
                            break
                        else:
                            points_vertical_line.append(pixel_7)
                            new_vertical_line.append(pixel_7)
                            m = pixel_7[0]
                            n = pixel_7[1]
                    elif (pixel_8 in anchor_data) == True:
                        if (pixel_8 in points_vertical_line) is True or (pixel_8 in points_horizontal_line) is True:
                            break
                        else:
                            points_vertical_line.append(pixel_8)
                            new_vertical_line.append(pixel_8)
                            m = pixel_8[0]
                            n = pixel_8[1]
                    elif (pixel_9 in anchor_data) == True:
                        if (pixel_9 in points_vertical_line) is True or (pixel_9 in points_horizontal_line) is True:
                            break
                        else:
                            points_vertical_line.append(pixel_9)
                            new_vertical_line.append(pixel_9)
                            m = pixel_9[0]
                            n = pixel_9[1]
                    else:
                        if pixel_7_level > pixel_8_level and pixel_7_level > pixel_9_level:
                            if (pixel_7 in points_vertical_line) is True or (pixel_7 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_7)
                                new_vertical_line.append(pixel_7)
                                m = pixel_7[0]
                                n = pixel_7[1]
                        if pixel_8_level > pixel_7_level and pixel_8_level > pixel_9_level:
                            if (pixel_8 in points_vertical_line) is True or (pixel_8 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_8)
                                new_vertical_line.append(pixel_8)
                                m = pixel_8[0]
                                n = pixel_8[1]
                        if pixel_9_level > pixel_7_level and pixel_9_level > pixel_8_level:
                            if (pixel_9 in points_vertical_line) is True or (pixel_9 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_9)
                                new_vertical_line.append(pixel_9)
                                m = pixel_9[0]
                                n = pixel_9[1]
                        if pixel_7_level == pixel_8_level > pixel_9_level or pixel_8_level == pixel_9_level > pixel_7_level:
                            if (pixel_8 in points_vertical_line) is True or (pixel_8 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_8)
                                new_vertical_line.append(pixel_8)
                                m = pixel_8[0]
                                n = pixel_8[1]
                        if pixel_7_level == pixel_9_level and pixel_7_level > pixel_8_level:
                            if (pixel_7 in points_vertical_line) is True or (pixel_7 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_7)
                                new_vertical_line.append(pixel_7)
                                m = pixel_7[0]
                                n = pixel_7[1]
                        if pixel_7_level == pixel_8_level and pixel_8_level == pixel_9_level:
                            if (pixel_8 in points_vertical_line) is True or (pixel_8 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_8)
                                new_vertical_line.append(pixel_8)
                                m = pixel_8[0]
                                n = pixel_8[1]
                    if m + 1 < x_dim and n - 1 < y_dim:
                        pixel_7_level = grad[m + 1, n - 1]
                        pixel_7 = [m + 1, n - 1]
                    else:
                        break
                    if m + 1 < x_dim and n < y_dim:
                        pixel_8_level = grad[m + 1, n]
                        pixel_8 = [m + 1, n]
                    else:
                        break
                    if m + 1 < x_dim and n + 1 < y_dim:
                        pixel_9_level = grad[m + 1, n + 1]
                        pixel_9 = [m + 1, n + 1]
                    else:
                        break
                    downward_stop = int(pixel_7_level) + int(pixel_8_level) + int(pixel_9_level)
                else:
                    break
        # Return to the original initial anchor point's neighbors
        pixel_7_level = grad[i_pos + 1, j_pos - 1]
        pixel_7 = [i_pos + 1, j_pos - 1]
        pixel_9_level = grad[i_pos + 1, j_pos + 1]
        pixel_9 = [i_pos + 1, j_pos + 1]

        if switch_1 == 1:
            # Search upward (using pixel 1,2,3)
            upward_stop = int(pixel_1_level) + int(pixel_2_level) + int(pixel_3_level)
            while (not upward_stop == 0):
                if len(new_vertical_line) < num_point_up_limit:
                    if (pixel_1 in anchor_data) == True:
                        if (pixel_1 in points_vertical_line) is True or (pixel_1 in points_horizontal_line) is True:
                            break
                        else:
                            points_vertical_line.append(pixel_1)
                            new_vertical_line.insert(0, pixel_1)
                            m = pixel_1[0]
                            n = pixel_1[1]
                    elif (pixel_2 in anchor_data) == True:
                        if (pixel_2 in points_vertical_line) is True or (pixel_2 in points_horizontal_line) is True:
                            break
                        else:
                            points_vertical_line.append(pixel_2)
                            new_vertical_line.insert(0, pixel_2)
                            m = pixel_2[0]
                            n = pixel_2[1]
                    elif (pixel_3 in anchor_data) == True:
                        if (pixel_3 in points_vertical_line) is True or (pixel_3 in points_horizontal_line) is True:
                            break
                        else:
                            points_vertical_line.append(pixel_3)
                            new_vertical_line.insert(0, pixel_3)
                            m = pixel_3[0]
                            n = pixel_3[1]
                    else:
                        if pixel_1_level > pixel_2_level and pixel_1_level > pixel_3_level:
                            if (pixel_1 in points_vertical_line) is True or (pixel_1 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_1)
                                new_vertical_line.insert(0, pixel_1)
                                m = pixel_1[0]
                                n = pixel_1[1]
                        if pixel_2_level > pixel_1_level and pixel_2_level > pixel_3_level:
                            if (pixel_2 in points_vertical_line) is True or (pixel_2 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_2)
                                new_vertical_line.insert(0, pixel_2)
                                m = pixel_2[0]
                                n = pixel_2[1]
                        if pixel_3_level > pixel_1_level and pixel_3_level > pixel_2_level:
                            if (pixel_3 in points_vertical_line) is True or (pixel_3 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_3)
                                new_vertical_line.insert(0, pixel_3)
                                m = pixel_3[0]
                                n = pixel_3[1]
                        if pixel_1_level == pixel_2_level > pixel_3_level or pixel_2_level == pixel_3_level > pixel_1_level:
                            if (pixel_2 in points_vertical_line) is True or (pixel_2 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_2)
                                new_vertical_line.insert(0, pixel_2)
                                m = pixel_2[0]
                                n = pixel_2[1]
                        if pixel_1_level == pixel_3_level and pixel_3_level > pixel_2_level:
                            if (pixel_1 in points_vertical_line) is True or (pixel_1 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_1)
                                new_vertical_line.insert(0, pixel_1)
                                m = pixel_1[0]
                                n = pixel_1[1]
                        if pixel_1_level == pixel_2_level and pixel_2_level == pixel_3_level:
                            if (pixel_2 in points_vertical_line) is True or (pixel_2 in points_horizontal_line) is True:
                                break
                            else:
                                points_vertical_line.append(pixel_2)
                                new_vertical_line.insert(0, pixel_2)
                                m = pixel_2[0]
                                n = pixel_2[1]
                    if m - 1 < x_dim and n - 1 < y_dim:
                        pixel_1_level = grad[m - 1, n - 1]
                        pixel_1 = [m - 1, n - 1]
                    else:
                        break
                    if m - 1 < x_dim and n < y_dim:
                        pixel_2_level = grad[m - 1, n]
                        pixel_2 = [m - 1, n]
                    else:
                        break
                    if m - 1 < x_dim and n + 1 < y_dim:
                        pixel_3_level = grad[m - 1, n + 1]
                        pixel_3 = [m - 1, n + 1]
                    else:
                        break
                    upward_stop = int(pixel_1_level) + int(pixel_2_level) + int(pixel_3_level)
                else:
                    break
        # Return to the original initial anchor point's neighbors
        pixel_1_level = grad[i_pos - 1, j_pos - 1]
        pixel_1 = [i_pos - 1, j_pos - 1]
        pixel_3_level = grad[i_pos - 1, j_pos + 1]
        pixel_3 = [i_pos - 1, j_pos + 1]

        if switch_3 == 1:
            # Search to the right (using pixel 3,6,9)
            right_stop = int(pixel_3_level) + int(pixel_6_level) + int(pixel_9_level)
            point_count = 0
            while (not right_stop == 0):
                if len(new_horizontal_line) < num_point_up_limit:
                    if (pixel_3 in anchor_data) == True:
                        if (pixel_3 in points_vertical_line) is True or (pixel_3 in points_horizontal_line) is True:
                            break
                        else:
                            points_horizontal_line.append(pixel_3)
                            new_horizontal_line.append(pixel_3)
                            m = pixel_3[0]
                            n = pixel_3[1]
                    elif (pixel_6 in anchor_data) == True:
                        if (pixel_6 in points_vertical_line) is True or (pixel_6 in points_horizontal_line) is True:
                            break
                        else:
                            points_horizontal_line.append(pixel_6)
                            new_horizontal_line.append(pixel_6)
                            m = pixel_6[0]
                            n = pixel_6[1]
                    elif (pixel_9 in anchor_data) == True:
                        if (pixel_9 in points_vertical_line) is True or (pixel_9 in points_horizontal_line) is True:
                            break
                        else:
                            points_horizontal_line.append(pixel_9)
                            new_horizontal_line.append(pixel_9)
                            m = pixel_9[0]
                            n = pixel_9[1]
                    else:
                        if pixel_3_level > pixel_6_level and pixel_3_level > pixel_9_level:
                            if (pixel_3 in points_vertical_line) is True or (pixel_3 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_3)
                                new_horizontal_line.append(pixel_3)
                                m = pixel_3[0]
                                n = pixel_3[1]
                        if pixel_6_level > pixel_3_level and pixel_6_level > pixel_9_level:
                            if (pixel_6 in points_vertical_line) is True or (pixel_6 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_6)
                                new_horizontal_line.append(pixel_6)
                                m = pixel_6[0]
                                n = pixel_6[1]
                        if pixel_9_level > pixel_3_level and pixel_9_level > pixel_6_level:
                            if (pixel_9 in points_vertical_line) is True or (pixel_9 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_9)
                                new_horizontal_line.append(pixel_9)
                                m = pixel_9[0]
                                n = pixel_9[1]
                        if pixel_3_level == pixel_6_level > pixel_9_level or pixel_6_level == pixel_9_level > pixel_3_level:
                            if (pixel_6 in points_vertical_line) is True or (pixel_6 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_6)
                                new_horizontal_line.append(pixel_6)
                                m = pixel_6[0]
                                n = pixel_6[1]
                        if pixel_3_level == pixel_9_level and pixel_9_level > pixel_6_level:
                            if (pixel_3 in points_vertical_line) is True or (pixel_3 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_3)
                                new_horizontal_line.append(pixel_3)
                                m = pixel_3[0]
                                n = pixel_3[1]
                        if pixel_3_level == pixel_6_level and pixel_6_level == pixel_9_level:
                            if (pixel_6 in points_vertical_line) is True or (pixel_6 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_6)
                                new_horizontal_line.append(pixel_6)
                                m = pixel_6[0]
                                n = pixel_6[1]
                    if m - 1 < x_dim and n + 1 < y_dim:
                        pixel_3_level = grad[m - 1, n + 1]
                        pixel_3 = [m - 1, n + 1]
                    else:
                        break
                    if m < x_dim and n + 1 < y_dim:
                        pixel_6_level = grad[m, n + 1]
                        pixel_6 = [m, n + 1]
                    else:
                        break
                    if m + 1 < x_dim and n + 1 < y_dim:
                        pixel_9_level = grad[m + 1, n + 1]
                        pixel_9 = [m + 1, n + 1]
                    else:
                        break
                    right_stop = int(pixel_3_level) + int(pixel_6_level) + int(pixel_9_level)
                else:
                    break
        # Return to the original initial anchor point's neighbors
        pixel_3_level = grad[i_pos - 1, j_pos + 1]
        pixel_3 = [i_pos - 1, j_pos + 1]
        pixel_9_level = grad[i_pos + 1, j_pos + 1]
        pixel_9 = [i_pos + 1, j_pos + 1]

        if switch_4 == 1:
            # Search to the left (using pixel 1,4,7)
            left_stop = int(pixel_1_level) + int(pixel_4_level) + int(pixel_7_level)
            point_count = 0
            while (not left_stop == 0):
                if len(new_horizontal_line) < num_point_up_limit:
                    if (pixel_1 in anchor_data) == True:
                        if (pixel_1 in points_vertical_line) is True or (pixel_1 in points_horizontal_line) is True:
                            break
                        else:
                            points_horizontal_line.append(pixel_1)
                            new_horizontal_line.insert(0, pixel_1)
                            m = pixel_1[0]
                            n = pixel_1[1]
                    elif (pixel_4 in anchor_data) == True:
                        if (pixel_4 in points_vertical_line) is True or (pixel_4 in points_horizontal_line) is True:
                            break
                        else:
                            points_horizontal_line.append(pixel_4)
                            new_horizontal_line.insert(0, pixel_4)
                            m = pixel_4[0]
                            n = pixel_4[1]
                    elif (pixel_7 in anchor_data) == True:
                        if (pixel_7 in points_vertical_line) is True or (pixel_7 in points_horizontal_line) is True:
                            break
                        else:
                            points_horizontal_line.append(pixel_7)
                            new_horizontal_line.insert(0, pixel_7)
                            m = pixel_7[0]
                            n = pixel_7[1]
                    else:
                        if pixel_1_level > pixel_4_level and pixel_1_level > pixel_7_level:
                            if (pixel_1 in points_vertical_line) is True or (pixel_1 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_1)
                                new_horizontal_line.insert(0, pixel_1)
                                m = pixel_1[0]
                                n = pixel_1[1]
                        if pixel_4_level > pixel_1_level and pixel_4_level > pixel_7_level:
                            if (pixel_4 in points_vertical_line) is True or (pixel_4 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_4)
                                new_horizontal_line.insert(0, pixel_4)
                                m = pixel_4[0]
                                n = pixel_4[1]
                        if pixel_7_level > pixel_1_level and pixel_7_level > pixel_4_level:
                            if (pixel_7 in points_vertical_line) is True or (pixel_7 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_7)
                                new_horizontal_line.insert(0, pixel_7)
                                m = pixel_7[0]
                                n = pixel_7[1]
                        if pixel_1_level == pixel_4_level > pixel_7_level or pixel_4_level == pixel_7_level > pixel_1_level:
                            if (pixel_4 in points_vertical_line) is True or (pixel_4 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_4)
                                new_horizontal_line.insert(0, pixel_4)
                                m = pixel_4[0]
                                n = pixel_4[1]
                        if pixel_1_level == pixel_7_level and pixel_7_level > pixel_4_level:
                            if (pixel_1 in points_vertical_line) is True or (pixel_1 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_1)
                                new_horizontal_line.insert(0, pixel_1)
                                m = pixel_1[0]
                                n = pixel_1[1]
                        if pixel_1_level == pixel_4_level and pixel_4_level == pixel_7_level:
                            if (pixel_4 in points_vertical_line) is True or (pixel_4 in points_horizontal_line) is True:
                                break
                            else:
                                points_horizontal_line.append(pixel_4)
                                new_horizontal_line.insert(0, pixel_4)
                                m = pixel_4[0]
                                n = pixel_4[1]
                    if m - 1 < x_dim and n - 1 < y_dim:
                        pixel_1_level = grad[m - 1, n - 1]
                        pixel_1 = [m - 1, n - 1]
                    else:
                        break
                    if m < x_dim and n - 1 < y_dim:
                        pixel_4_level = grad[m, n - 1]
                        pixel_4 = [m, n - 1]
                    else:
                        break
                    if m + 1 < x_dim and n - 1 < y_dim:
                        pixel_7_level = grad[m + 1, n - 1]
                        pixel_7 = [m + 1, n - 1]
                    else:
                        break
                    left_stop = int(pixel_1_level) + int(pixel_4_level) + int(pixel_7_level)
                else:
                    break
        # Return to the original initial anchor point's neighbors
        pixel_1_level = grad[i_pos - 1, j_pos - 1]
        pixel_1 = [i_pos - 1, j_pos - 1]
        pixel_7_level = grad[i_pos + 1, j_pos - 1]
        pixel_7 = [i_pos + 1, j_pos - 1]

        if len(new_vertical_line) >= num_point_low_limit:
            lines_vertical.append(new_vertical_line)
        if len(new_horizontal_line) >= num_point_low_limit:
            lines_horizontal.append(new_horizontal_line)

    return lines_vertical,lines_horizontal

def feature_point_finding(vertical_input, horizontal_input):

    feature_points = []

    for line in vertical_input:

        mid_point = []
        x_start = line[0][0]
        y_start = line[0][1]

        x_end = line[-1][0]
        y_end = line[-1][1]

        x_mid = int((x_start + x_end) * 0.5)
        y_mid = int((y_start + y_end) * 0.5)
        mid_point.append(x_mid)
        mid_point.append(y_mid)
        feature_points.append(mid_point)

    for line in horizontal_input:

        mid_point = []
        x_start = line[0][0]
        y_start = line[0][1]

        x_end = line[-1][0]
        y_end = line[-1][1]

        x_mid = int((x_start + x_end) * 0.5)
        y_mid = int((y_start + y_end) * 0.5)
        mid_point.append(x_mid)
        mid_point.append(y_mid)
        feature_points.append(mid_point)

    return feature_points

def process_DBSCAN(data):

    X = StandardScaler().fit_transform(data)

    db = DBSCAN(eps = 3, min_samples = 3).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # If the ground truth labels are not known, evaluation must be performed using the model itself.
    # The Silhouette Coefficient  is an example of such an evaluation, where a higher Silhouette Coefficient
    # score relates to a model with better defined clusters. The Silhouette Coefficient is defined for each
    # sample and is composed of two scores:
    # a: The mean distance between a sample and all other points in the same class.
    # b: The mean distance between a sample and all other points in the next nearest cluster.
    # The Silhouette Coefficient s for a single sample is then given as:
    # s = (b - a) / max(a, b)

    # silhouette_coefficient = metrics.silhouette_score(X, labels)

    clustered_output = []
    noise_output = []
    unique_labels = set(labels)
    for k in unique_labels:

        clustered_data = []
        noise = []
        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        clustered_data.append(xy)
        clustered_output.append(clustered_data)

        xy = X[class_member_mask & ~core_samples_mask]
        noise.append(xy)
        noise_output.append(noise)

    return labels, n_clusters_


def major_point_filter(feature_points, major_group):

    major_feature_points = []

    for i in major_group:
        major_feature_points.append(feature_points[i])

    return major_feature_points


def feature_point_statistic(input_points):

    # initialize the parameters
    max_value = 0.0
    min_value = 0.0
    mean_value = 0.0
    median_value = 0.0

    # total number of input point
    num_point = len(input_points)

    # make point list into point numpy array
    input_points = np.asarray(input_points)

    # calculating the distances between points to points
    if input_points != []:
        dist_condensed = pdist(input_points)
        # dist = squareform(dist_condensed)

        # calculating the statistic of distance result
        max_value = np.amax(dist_condensed)
        min_value = np.amin(dist_condensed)
        mean_value = np.mean(dist_condensed)
        median_value = np.median(dist_condensed)
    else:
        max_value = 0.0
        min_value = 0.0
        mean_value = 0.0
        median_value = 0.0

    return max_value, min_value, mean_value, median_value, num_point

def landing_sign_detection(img):
    # ==================================================================================================================== #
    # Output/input file names
    # ==================================================================================================================== #
    start_time = time.time()
    # output_DBSCAN_result = 'DBSCAN_result.png'
    # text_for_image = "Altitude = 0.8 meters"
    # converting the input image from color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pic_gray_array = []
    pic_grad_x_array = []
    pic_grad_y_array = []

    bin_array = []
    marked_gray_level = []
    new_pic_gray_array = []

    anchor_array_x = []
    anchor_array_y = []
    anchor_array_total = []

    anchor_position = []
    anchor_level = []

    pic_anchor_array_x = []
    pic_anchor_array_y = []

    lines_vertical_fitted = []
    lines_horizontal_fitted = []

    detial_ratio_hrz = 5
    detial_ratio_vrt = 1
    count = 0
    clean_pad = 2
    num_point_up_limit = 8
    num_point_low_limit = 8
    max_ratio = 0.45

    switch_upward_search = 0
    switch_downward_search = 1
    switch_right_search = 1
    switch_left_search = 1

    switch_plot = 0
    switch_figure_2 = 0
    switch_figure_3 = 0
    switch_figure_4 = 0
    switch_figure_5 = 0
    switch_figure_6 = 0
    switch_figure_7 = 0
    switch_figure_8 = 0
    switch_figure_9 = 0
    switch_figure_10 = 0
    switch_figure_11 = 0
    switch_figure_12 = 0

    # ==================================================================================================================== #
    # Step 1: Gray Level Filter
    # ==================================================================================================================== #
    for index_1 in gray:
        pic_gray_array = np.concatenate([pic_gray_array,index_1])

    for index_2 in range(0,256):
        bin_array.append(index_2)

    total_pixel_num = len(pic_gray_array)
    # valid_peak_threshold = 0.01 * total_pixel_num
    valid_peak_threshold = 500

    plt.figure(1)
    plt.subplot(2, 1, 1)
    n_1, bins_1, patches_1 = plt.hist(x=pic_gray_array, bins=bin_array, color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Gray Levels')
    plt.xlim(0,255)
    plt.yscale('log')
    plt.ylabel('Frequency')
    plt.ylim(1, 1E+6)
    plt.title('Distribution of Gray Levels')

    for index_3 in n_1:
        if index_3 > valid_peak_threshold and count < 254:
            marked_gray_level.append(count)
        if index_3 > valid_peak_threshold and count == 254:
            marked_gray_level.append(count + 1)
        count = count + 1

    ln = len(marked_gray_level)

    for index_4 in pic_gray_array:
        if index_4 in marked_gray_level:
            new_pic_gray_array.append(index_4)


    plt.subplot(2, 1, 2)
    n_2, bins_2, patches_2 = plt.hist(x=new_pic_gray_array, bins=bin_array, color='#b03a2e',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Gray Levels')
    plt.xlim(0,255)
    plt.yscale('log')
    plt.ylabel('Frequency')
    plt.ylim(1, 1E+6)
    plt.title('Distribution of Gray Levels (valid peak selected)')

    if len(marked_gray_level) > 1:
        threshold = 0.5 * (marked_gray_level[-2] + marked_gray_level[-1])
    else:
        threshold = marked_gray_level[0]

    thresh_array = np.zeros((len(gray) * len(gray[0]), ), dtype=int)
    thresh = Image.new('L', (len(gray[0]), len(gray)))

    for index_5, val_1 in enumerate(pic_gray_array):
        if val_1 > threshold:
            thresh_array[index_5] = int(val_1)
        else:
            thresh_array[index_5] = int(0.0)

    thresh.putdata(thresh_array)
    # thresh.save(output_filename)
    thresh = np.asarray(thresh)

    # ==================================================================================================================== #
    # Step 2: Sobel Operator - Gradient Calculation
    # ==================================================================================================================== #
    sobel_64f_x = cv2.Sobel(thresh, cv2.CV_64F,1,0,ksize=1)
    sobel_64f_y = cv2.Sobel(thresh, cv2.CV_64F,0,1,ksize=1)

    abs_sobel64f_x = np.absolute(sobel_64f_x)
    abs_sobel64f_y = np.absolute(sobel_64f_y)

    sobel_8u_x = np.uint8(abs_sobel64f_x) / 10.0
    sobel_8u_y = np.uint8(abs_sobel64f_y) / 10.0

    # cv2.imwrite(output_sobelx_filename, sobel_8u_x)
    # cv2.imwrite(output_sobely_filename, sobel_8u_y)

    for index_6 in sobel_8u_x:
        pic_grad_x_array = np.concatenate([pic_grad_x_array,index_6])

    for index_7 in sobel_8u_y:
        pic_grad_y_array = np.concatenate([pic_grad_y_array,index_7])

    pic_grad_array = np.sqrt(pic_grad_x_array ** 2 + pic_grad_y_array ** 2)

    grad_threshold = int(max((1 - max_ratio) * pic_grad_array))

    for index_8 in range(len(pic_grad_array)):

        pic_grad = pic_grad_array[index_8]

        if pic_grad < grad_threshold:
            pic_grad_x_array[index_8] = int(0.0)
            pic_grad_y_array[index_8] = int(0.0)

        else:
            pic_grad_x_array[index_8] = int(255)
            pic_grad_y_array[index_8] = int(255)

    pic_grad_total_array = 0.5 * (pic_grad_x_array + pic_grad_y_array)

    grad = Image.new('L', (len(gray[0]), len(gray)))
    edge_img_x = Image.new('L', (len(gray[0]), len(gray)))
    edge_img_y = Image.new('L', (len(gray[0]), len(gray)))
    edge_img_total = Image.new('L', (len(gray[0]), len(gray)))

    grad.putdata(pic_grad_array)
    edge_img_x.putdata(pic_grad_x_array)
    edge_img_y.putdata(pic_grad_y_array)
    edge_img_total.putdata(pic_grad_total_array)

    # grad.save(output_sobel_grad_filename)
    # edge_img_x.save(output_filename_edge_img_x)
    # edge_img_y.save(output_filename_edge_img_y)
    # edge_img_total.save(output_filename_edge_img_total)

    grad = np.asarray(grad)
    edge_img_x = np.asarray(edge_img_x)
    edge_img_y = np.asarray(edge_img_y)
    edge_img_total = np.asarray(edge_img_total)

    # ==================================================================================================================== #
    # Step 3: Select Anchor Points
    # ==================================================================================================================== #
    # Row raster scan
    row_zeros = np.zeros((len(gray[0]), ), dtype = int)

    for index_9, row in enumerate(edge_img_x):
        if index_9 % detial_ratio_hrz == 0:
            anchor_array_x.append(row)
        else:
            anchor_array_x.append(row_zeros)

    # Column raster scan
    for index_10, row in enumerate(edge_img_y):
        row_new = []
        for index_10, col in enumerate(row):

            if index_10 % detial_ratio_vrt == 0:
                row_new.append(col)
            else:
                row_new.append(0.0)

        anchor_array_y.append(row_new)

    for index_11 in anchor_array_x:
        pic_anchor_array_x = np.concatenate([pic_anchor_array_x,index_11])
    for index_12 in anchor_array_y:
        pic_anchor_array_y = np.concatenate([pic_anchor_array_y,index_12])

    pic_anchor_array_total = pic_anchor_array_x + pic_anchor_array_y

    anchor_x = Image.new('L', (len(gray[0]), len(gray)))
    anchor_y = Image.new('L', (len(gray[0]), len(gray)))
    anchor_total = Image.new('L', (len(gray[0]), len(gray)))

    anchor_x.putdata(pic_anchor_array_x)
    # anchor_x.save(output_anchor_x_filename)
    anchor_x = np.asarray(anchor_x)
    anchor_y.putdata(pic_anchor_array_y)
    # anchor_y.save(output_anchor_y_filename)
    anchor_y = np.asarray(anchor_y)
    anchor_total.putdata(pic_anchor_array_total)
    # anchor_total.save(output_anchor_total_filename)
    anchor_total = np.asarray(anchor_total)

    # ==================================================================================================================== #
    # Step 4: Trimming the Clustered Anchor Points
    # ==================================================================================================================== #
    # Create a filter to clean the pixel around a given anchor point with 'clean_pad' pixel range
    dim_filter = 2* clean_pad + 1
    anchor_filter = np.zeros(shape = (dim_filter,dim_filter))
    anchor_filter[clean_pad,clean_pad] = anchor_filter[clean_pad,clean_pad] + 1

    anchor_total_extend = np.zeros(shape = ((anchor_total.shape[0] + 2*clean_pad), (anchor_total.shape[1] + 2*clean_pad)))

    x_dim = anchor_total.shape[0]
    y_dim = anchor_total.shape[1]

    anchor_total_extend[clean_pad:x_dim+clean_pad,clean_pad:y_dim+clean_pad] = anchor_total_extend[clean_pad:x_dim+clean_pad,clean_pad:y_dim+clean_pad] + anchor_total

    for i in range(anchor_total.shape[0]):
        for j in range(anchor_total.shape[1]):
            m = i + clean_pad
            n = j + clean_pad
            element = anchor_total_extend[m,n]
            if element != 0:
                anchor_total_extend[m-clean_pad:m+clean_pad+1, n-clean_pad:n+clean_pad+1] = anchor_total_extend[m-clean_pad:m+clean_pad+1, n-clean_pad:n+clean_pad+1]*anchor_filter

    anchor_total = anchor_total_extend[clean_pad:x_dim+clean_pad,clean_pad:y_dim+clean_pad]

    # ==================================================================================================================== #
    # Step 5: Recoding the Location of Anchor Points
    # ==================================================================================================================== #
    anchor_count = 0
    for i, row in enumerate(anchor_total):
        for j, col in enumerate(row):
            level = anchor_total[i,j]
            new_position = []
            if level != 0:
                anchor_count = anchor_count + 1
                new_position.append(i)
                new_position.append(j)
                anchor_position.append(new_position)

    anchor_position_save = np.asarray(anchor_position)

    # ==================================================================================================================== #
    # Step 6: Linking the Anchor Points
    # ==================================================================================================================== #

    switch_1 = switch_upward_search
    switch_2 = switch_downward_search
    switch_3 = switch_right_search
    switch_4 = switch_left_search

    lines_vertical, lines_horizontal = gradient_search(switch_1, switch_2, switch_3, switch_4,
                                                       grad, anchor_position, num_point_up_limit, num_point_low_limit,
                                                       x_dim, y_dim)

    # ==================================================================================================================== #
    # Step 7: line fitting
    # ==================================================================================================================== #

    for line in lines_vertical:
        line_input_v = linear_regression_vertical(line)
        lines_vertical_fitted.append(line_input_v)

    for line in lines_horizontal:
        line_input_h = linear_regression_horizontal(line)
        lines_horizontal_fitted.append(line_input_h)

    # ==================================================================================================================== #
    # Step 8: Creating the edge image
    # ==================================================================================================================== #

    edge_image_x, edge_image_y, edge_image_total = edge_line_drawer(gray, lines_vertical_fitted, lines_horizontal_fitted)

    # ==================================================================================================================== #
    # Step 8: Locating the feature points
    # ==================================================================================================================== #

    feature_points = feature_point_finding(lines_vertical_fitted, lines_horizontal_fitted)
    feature_points_array = np.asarray(feature_points)

    if feature_points != []:
        for point in feature_points:
            row_pos = point[1]
            col_pos = point[0]
            cv2.circle(img,(row_pos, col_pos), 3, (0,0,255), 2)

        print("number of feature point: %d" % (len(feature_points)))

    else:
        print("Current lacation of drone unable to find any feature point.")

    # ==================================================================================================================== #
    # Step 9: DBSCAN
    # ==================================================================================================================== #

    labels, num_clusters = process_DBSCAN(feature_points)

    cluster_total = []
    cluster_count = []

    x_min = []
    x_max = []
    y_min = []
    y_max = []

    major_group_x = []
    major_group_y = []
    major_group_x_avg = -1
    major_group_y_avg = -1

    for k in range(num_clusters):
        count_tag = 0
        cluster_k = []
        for i, tag in enumerate(labels):
            if tag == k:
                count_tag = count_tag + 1
                cluster_k.append(i)
        cluster_total.append(cluster_k)
        cluster_count.append(count_tag)

    if cluster_count != []:
        major_group_index = cluster_count.index(max(cluster_count))
        major_group = cluster_total[major_group_index]
        del cluster_total[major_group_index]
        other_group = []
        for i in cluster_total:
            other_group += i
    else:
        major_group = []
        other_group = []

    for point in feature_points:
        row_pos = point[1]
        col_pos = point[0]
        cv2.circle(img,(row_pos, col_pos), 3, (255,200,0), 2)

    if major_group != []:
        for i in major_group:
            row_pos = feature_points[i][1]
            col_pos = feature_points[i][0]
            major_group_x.append(col_pos)
            major_group_y.append(row_pos)
            cv2.circle(img,(row_pos, col_pos), 3, (255,0,0), 2)

        for i in other_group:
            row_pos = feature_points[i][1]
            col_pos = feature_points[i][0]
            cv2.circle(img,(row_pos, col_pos), 3, (0,0,255), 2)

        major_group_x_avg = np.average(major_group_x)
        major_group_y_avg = np.average(major_group_y)

        x_min = np.amin(major_group_x_avg)
        x_max = np.amin(major_group_x_avg)
        y_min = np.amin(major_group_y_avg)
        y_max = np.amin(major_group_y_avg)

        if major_group_x_avg > 0 and major_group_y_avg > 0:
            major_center = [int(major_group_x_avg), int(major_group_y_avg)]

            cv2.circle(img,(major_center[1], major_center[0]), 3, (0,255,0), -1)

            print("Center of the H landing sign: x = %f, y = %f" % (major_center[1], major_center[0]))

        else:

            print("No landing mark center can be found in current drone location")

    major_feature_points = major_point_filter(feature_points, major_group)
    f_max, f_min, f_mean, f_median, f_num_point = feature_point_statistic(feature_points)
    mj_max, mj_min, mj_mean, mj_median, mj_num_point = feature_point_statistic(major_feature_points)

    # ==================================================================================================================== #
    # Figures and outputs
    # ==================================================================================================================== #
    if switch_plot == 1:

        if switch_figure_2 == 1:
            switch = 2
            title_1 = 'original input image'
            title_2 = 'filtered image'
            plot_figure_multiple(switch, img, thresh, title_1, title_2)

        if switch_figure_3 == 1:
            switch = 3
            title_1 = 'sobel processed in y direction'
            title_2 = 'sobel processed in x direction'
            plot_figure_multiple(switch, sobel_8u_x, sobel_8u_y, title_1, title_2)

        if switch_figure_4 == 1:
            switch = 4
            title_1 = 'edge points in y direction'
            title_2 = 'edge points in x direction'
            plot_figure_multiple(switch, edge_img_x, edge_img_y, title_1, title_2)

        if switch_figure_5 == 1:
            switch = 5
            title_1 = 'anchor points in y direction'
            title_2 = 'anchor points in x direction'
            plot_figure_multiple(switch, anchor_x, anchor_y, title_1, title_2)

        if switch_figure_6 == 1:
            switch = 6
            title = 'combined gradient'
            plot_figure_single(switch, grad, title)

        if switch_figure_7 == 1:
            switch = 7
            title = 'total edge points'
            plot_figure_single(switch, edge_img_total, title)

        if switch_figure_8 == 1:
            switch = 8
            title = 'total anchor points'
            plot_figure_single(switch, anchor_total, title)

        if switch_figure_9 == 1:
            switch = 9
            title_1 = 'horizontal edges'
            title_2 = 'vertical edges'
            plot_figure_multiple(switch, edge_image_x, edge_image_y, title_1, title_2)

        if switch_figure_10 == 1:
            switch = 10
            title = 'final edges'
            plot_figure_single(switch, edge_image_total, title)

        if switch_figure_11 == 1:
            switch = 11
            title = 'feature points for the original input image'
            plot_figure_single(switch, img, title)

        if switch_figure_12 == 1:
            switch = 12
            title = 'DBSCAN result for the original input image'
            plot_figure_single(switch, img, title)

        print("--- %.2f seconds ---" % (time.time() - start_time))
        plt.show()

    print('statistic of distance between all the feature point:')
    print(f_max, f_min, f_mean, f_median, f_num_point)
    print()
    print('statistic of distance between all the major feature point:')
    print(mj_max, mj_min, mj_mean, mj_median, mj_num_point)
    print()
    print('samples in other group:')
    print(other_group)
    print()

    # Showing the run time of this program
    print("--- %.2f seconds ---" % (time.time() - start_time))
    print()
    # ==================================================================================================================== #

    if major_group_x_avg > 0 and major_group_y_avg > 0:
        major_center = [int(major_group_y_avg), int(major_group_x_avg)]
        major_center = np.asarray(major_center)
    else:
        major_center = []

    return major_center, major_group, feature_points, f_min, f_num_point, mj_min, mj_num_point, other_group, x_min, x_max, y_min, y_max

