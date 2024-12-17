# %%
from datasets.drive import DriveFullData
from datasets.cremi import CremiFullData
from datasets.roads import RoadsFullData

import matplotlib.pyplot as plt
from skimage.morphology import remove_small_holes
import numpy as np
import skimage

# set the data paths

drive_data_path = "data/drive"
drive_test_path = "data/drive_test"

drive_data = DriveFullData(drive_data_path, drive_test_path)

# %%
# plote the top left corner of the image
plt.imshow(drive_data[0]["img"][:, 300:400, 320:420].permute(1, 2, 0))
plt.axis("off")
# and the corresponding segmentation
#plt.savefig("drive_img.png", bbox_inches="tight")
plt.show()
plt.imshow(drive_data[0]["seg"][1, 300:400, 320:420], cmap="gray", alpha=1)
plt.axis("off")
# tighten the layout
#plt.savefig("drive_seg.png", bbox_inches="tight")
plt.show()

# %%
# do connected component labeling on the displayed part of the segmentation
seg = drive_data[0]["seg"][1, 300:400, 320:420].numpy()
seg = seg.astype(np.uint8)
labels = skimage.measure.label(seg, connectivity=1)
# plot the connected components, give the background a white color
plt.imshow(labels, cmap="tab20")
plt.imshow(seg, cmap="gray", alpha=1-seg)
plt.axis("off")
#plt.savefig("drive_cc_4con.png", bbox_inches="tight")
plt.show()
# %%
# repreat the connected component labeling with 4-connectivity
labels = skimage.measure.label(seg, connectivity=2)
# plot the connected components, give the background a white color
plt.imshow(labels, cmap="tab20")
plt.imshow(seg, cmap="gray", alpha=1-seg)
# remove the axis
plt.axis("off")
#plt.savefig("drive_cc_8con.png", bbox_inches="tight")
plt.show()

# %%

cremi_data_path = "data/cremi_original"
cremi_test_path = "data/cremi_original_test"

cremi_data = CremiFullData(cremi_data_path, cremi_test_path)

sample_idx = 27
# %%
# get the dimensions of the image
print(cremi_data[sample_idx]["img"].shape)

# plote the top left corner of the image
plt.imshow(cremi_data[sample_idx]["img"][:, 200:500, 100:400].permute(1, 2, 0))
plt.axis("off")
# and the corresponding segmentation
plt.savefig("gen_figures/cremi_img.png", bbox_inches="tight")
plt.show()
plt.imshow(cremi_data[sample_idx]["seg"][1,  200:500, 100:400], cmap="gray", alpha=1)
plt.axis("off")
# tighten the layout
plt.savefig("gen_figures/cremi_seg.png", bbox_inches="tight")
plt.show()


# %%
tab20b = plt.get_cmap('tab20b')  
tab20 = plt.get_cmap('tab20')  

# cobmione the two colormaps
colors = np.vstack((tab20b.colors, tab20.colors))
# create a new colormap
combined_cmap = plt.cm.colors.ListedColormap(colors)


# %%
# do connected component labeling on the displayed part of the segmentation
seg = cremi_data[sample_idx]["seg"][1, 200:500, 100:400].numpy()
seg = seg.astype(np.uint8)
labels_4 = skimage.measure.label(1-seg, connectivity=1)
# plot the connected components, give the background a white color
plt.imshow(labels_4, cmap=combined_cmap)
plt.imshow(seg, cmap="gray", alpha=seg)
plt.axis("off")
plt.savefig("gen_figures/cremi_cc_4con.png", bbox_inches="tight")
plt.show()
# %%
# repreat the connected component labeling with 4-connectivity
labels_8 = skimage.measure.label(1-seg, connectivity=2)
# plot the connected components, give the background a white color
plt.imshow(labels_8, cmap="tab20b")
plt.imshow(seg, cmap="gray", alpha=seg)
# remove the axis
plt.axis("off")
plt.savefig("gen_figures/cremi_cc_8con.png", bbox_inches="tight")
plt.show()
# %%


from skimage.measure import regionprops

def relabel_exact_matches(labels_4, labels_8):
    """
    Relabel components in labels_4 and labels_8 to share the same label
    only if their components match exactly.
    """
    # Initialize relabeled arrays
    relabeled_4 = np.zeros_like(labels_4)
    relabeled_8 = np.zeros_like(labels_8)
    
    new_label = 1  # Start assigning new labels from 1
    
    # Iterate through components in labels_4
    for region_4 in regionprops(labels_4):
        label_4 = region_4.label
        mask_4 = labels_4 == label_4  # Binary mask for this label in labels_4
        
        # Check if this mask matches exactly any component in labels_8
        overlapping_labels = np.unique(labels_8[mask_4])
        for label_8 in overlapping_labels:
            if label_8 == 0:  # Skip background
                continue
            
            mask_8 = labels_8 == label_8  # Binary mask for this label in labels_8
            
            # Check for exact match
            if np.array_equal(mask_4, mask_8):
                # Assign the same new label to both
                relabeled_4[mask_4] = new_label
                relabeled_8[mask_8] = new_label
                new_label += 1
            else:
                relabeled_4[mask_4] = new_label
                new_label += 1

    # assign new labels to all the component that have the identical label 1 in relabeled_8
    for region_8 in regionprops(labels_8):
        label_8 = region_8.label
        mask_8 = labels_8 == label_8

        # check if all the components in the mask have value 1
        if not np.all(relabeled_8[mask_8] == 1):
            continue

        # Check if this mask matches exactly any component in labels_4
        overlapping_labels = np.unique(labels_4[mask_8])

        for label_4 in overlapping_labels:
            if label_4 == 0:
                continue

            mask_4 = labels_4 == label_4

            if np.array_equal(mask_4, mask_8):
                continue
            else:
                relabeled_8[mask_8] = new_label
                new_label += 1

    
    
    return relabeled_4, relabeled_8


# %%
relabeled_4, relabeled_8 = relabel_exact_matches(labels_4, labels_8)
plt.imshow(relabeled_4, cmap="tab20b")
plt.imshow(seg, cmap="gray", alpha=seg)
plt.axis("off")
plt.savefig("gen_figures/cremi_cc_4con_relabel.png", bbox_inches="tight", dpi=300)
plt.show()

plt.imshow(relabeled_8, cmap="tab20b")
plt.imshow(seg, cmap="gray", alpha=seg)
plt.axis("off")
plt.savefig("gen_figures/cremi_cc_8con_relabel.png", bbox_inches="tight", dpi=300)
plt.show()
# %%


# %%
