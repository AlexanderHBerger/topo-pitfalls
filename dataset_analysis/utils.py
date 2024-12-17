import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt

def connectivity_susceptibility(data, pad = False):
    """
    Calculate the connectivity susceptibility for the given dataset.
    """
    num_data = len(data)
    metric_dict = {}

    one_visualized = False

    for i in range(num_data):
        label = data[i]["seg"][1].numpy().astype(np.uint8)

        # add a padding background padding
        if pad:
            label = np.pad(label, 1, mode="constant", constant_values=0)

        # labeling all the foreground components
        lab1 = skimage.measure.label(label, connectivity=1) 
        lab2 = skimage.measure.label(label, connectivity=2) 

        # labeling all the background components
        lab1_BG = skimage.measure.label(1-label, connectivity=1) #
        lab2_BG = skimage.measure.label(1-label, connectivity=2) 

        # labeling of all the components
        lab1_BG[1-label == 1] += np.max(lab1)
        lab2_BG[1-label == 1] += np.max(lab2)

        # combinging 8 connected components with 4 connected background components and vice versa
        lab1_combined = lab1 + lab2_BG 
        lab2_combined = lab2 + lab1_BG

        if not one_visualized:
            plt.imshow(lab1_combined)
            plt.axis("off")
            plt.show()
            plt.imshow(lab2_combined)
            plt.axis("off")
            plt.show()
            one_visualized = True

        # number of foreground connected components
        cc_4_FG = np.max(lab1)
        cc_8_FG = np.max(lab2)

        # calculate the number of background connected components
        cc_4_BG = np.max(lab1_BG)
        cc_8_BG = np.max(lab2_BG)

        # calculate the ARE and VOI difference
        are, are_prec, are_rec = skimage.metrics.adapted_rand_error(lab1, lab2)
        v_i = skimage.metrics.variation_of_information(lab1_combined, lab2_combined, ignore_labels = 0)

        try:
            metric_dict["B0_error"].append(abs(cc_4_FG-cc_8_FG))
            metric_dict["B1_error"].append(abs(cc_4_BG-cc_8_BG))
            metric_dict["ARE"].append(are)
            metric_dict["VOI_0"].append(v_i[0])
            metric_dict["VOI_1"].append(v_i[1])
        except KeyError:
            metric_dict["B0_error"] = [abs(cc_4_FG-cc_8_FG)]
            metric_dict["B1_error"] = [abs(cc_4_BG-cc_8_BG)]
            metric_dict["ARE"] = [are]
            metric_dict["VOI_0"] = [v_i[0]]
            metric_dict["VOI_1"] = [v_i[1]]

    return metric_dict


# util functions to display images and segmentations
def print_avg_errors(metric_dict):
    for key, value in metric_dict.items():
        print(f"{key}: {sum(value)/len(value)}")

def print_num_cc(num_cc_dict):
    for key, value in num_cc_dict.items():
        print(f"{key}: {sum(value)}")


def single_num_cc_sample(label, connectivity, min_size = 0):
    """
    Calculate the number of connected components for the given label.
    """
    labeled = skimage.measure.label(label, connectivity=connectivity)
    labeled = skimage.morphology.remove_small_objects(labeled, min_size=min_size, connectivity=connectivity)
    num_labels = len(np.unique(labeled)) - 1
    return num_labels


def all_num_cc_sample(label, min_size = 0):
    """
    Calculate the number of FG and BG connected components for the given label with 4 and 8 connectivity.
    """

    # num_cc_FG_8
    num_cc_FG_8 = single_num_cc_sample(label, connectivity=2, min_size=min_size)
    # num_cc_FG_4
    num_cc_FG_4 = single_num_cc_sample(label, connectivity=1, min_size=min_size)
    # num_cc_BG_8
    num_cc_BG_8 = single_num_cc_sample(1-label, connectivity=2, min_size=min_size)
    # num_cc_BG_4
    num_cc_BG_4 = single_num_cc_sample(1-label, connectivity=1, min_size=min_size)

    return num_cc_FG_8, num_cc_FG_4, num_cc_BG_8, num_cc_BG_4


def all_num_cc(dataset, min_size = 0):
    """
    Create a dictionary for foreground and background connected components with 8-connectivity and 4 connectivity.
    """
    cc_dict = {"BG_8": [], "BG_4": [], "FG_8": [], "FG_4": []}

    for i in range(len(dataset)):

        # dtype=np.uint8 is required for cv2.connectedComponents
        label = dataset[i]["seg"][1].numpy().astype(np.uint8)

        # get the number of connected components for the current sample
        num_cc_FG_8, num_cc_FG_4, num_cc_BG_8, num_cc_BG_4 = all_num_cc_sample(label, min_size=min_size)

        # append the number of connected components to the dictionary
        cc_dict["BG_8"].append(num_cc_BG_8)
        cc_dict["BG_4"].append(num_cc_BG_4)
        cc_dict["FG_8"].append(num_cc_FG_8)
        cc_dict["FG_4"].append(num_cc_FG_4)

    return cc_dict