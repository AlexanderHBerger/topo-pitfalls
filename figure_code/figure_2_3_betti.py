# import the necessary packages
# %%
from datasets.drive import DriveFullData
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_holes

# set the data paths

drive_data_path = "data/drive"
drive_test_path = "data/drive_test"
drive_data = DriveFullData(drive_data_path, drive_test_path)

# %%

# plote the top left corner of the image
plt.imshow(drive_data[0]["img"][:, 100:200, 100:200].permute(1, 2, 0))
plt.axis("off")
plt.savefig("gen_figures/drive_seg_raw.png", bbox_inches="tight")
# and the corresponding segmentation
plt.show()

# %%

# remove holes with size less than 4 pixels
seg_b1 = drive_data[0]["seg"][1].numpy().copy()
seg_b1 = remove_small_holes(seg_b1, 4)

# use a new segmentation where the capillaries are disconnected
seg_b0 = drive_data[0]["seg"][1].numpy().copy()

# slice through the segmentation
seg_b0[119, 100:140] = 0
seg_b0[139:140, 100:140] = 0


# save figures for all the segs

plt.imshow(drive_data[0]["seg"][1, 100:200, 100:200], cmap="gray")
plt.axis("off")
plt.savefig("gen_figures/drive_seg_be.png", bbox_inches="tight")


plt.imshow(seg_b1[100:200, 100:200], cmap="gray")
plt.axis("off")
plt.savefig("gen_figures/drive_p1_be.png", bbox_inches="tight")

plt.imshow(seg_b0[100:200, 100:200], cmap="gray")
plt.axis("off")
plt.savefig("gen_figures/drive_p2_be.png", bbox_inches="tight")

