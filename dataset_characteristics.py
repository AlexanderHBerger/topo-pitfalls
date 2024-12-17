# %%
from dataset_analysis import utils


from datasets.drive import DriveFullData
from datasets.roads import RoadsFullData
from datasets.cremi import CremiFullData
#from datasets.cracks import Cracks




# load the data for the 3 used datasets
cremi_train_path = "data/cremi_original"
cremi_test_path = "data/cremi_original_test"

drive_train_path = "data/drive"
drive_test_path = "data/drive_test"

roads_train_path = "data/roads"
roads_test_path = "data/roads_test"

cremi_data = CremiFullData(cremi_train_path, cremi_test_path)
drive_data = DriveFullData(drive_train_path, drive_test_path)
roads_data = RoadsFullData(roads_train_path, roads_test_path)

# %%
# get the number of connected components for the CREMI dataset
num_cc_cremi = utils.all_num_cc(cremi_data, min_size=0)
print("CREMI dataset")
utils.print_num_cc(num_cc_cremi)

#num_cc_cremi_rm1 = utils.all_num_cc(cremi_data, min_size=2)
#print("CREMI dataset Remove 1")
#print_num_cc(num_cc_cremi_rm1)
#num_cc_cremi_rm2 = utils.all_num_cc(cremi_data, min_size=3)
#print("CREMI dataset Remove 2")
#print_num_cc(num_cc_cremi_rm2)
#num_cc_cremi_rm5 = utils.all_num_cc(cremi_data, min_size=6)
#print("CREMI dataset Remove 5")
#print_num_cc(num_cc_cremi_rm5)

cremi_suscept = utils.connectivity_susceptibility(cremi_data)
utils.print_avg_errors(cremi_suscept)

# %%
# utils.all the number of connected components for the DRIVE dataset
num_cc_drive = utils.all_num_cc(drive_data, min_size=0)
print("drive dataset")
utils.print_num_cc(num_cc_drive)

#num_cc_drive_rm1 = utils.all_num_cc(drive_data, min_size=2)
#print("drive dataset Remove 1")
#print_num_cc(num_cc_drive_rm1)
#num_cc_drive_rm2 = utils.all_num_cc(drive_data, min_size=3)
#print("drive dataset Remove 2")
#print_num_cc(num_cc_drive_rm2)
#num_cc_drive_rm5 = utils.all_num_cc(drive_data, min_size=6)
#print("drive dataset Remove 5")
#print_num_cc(num_cc_drive_rm5)

drive_suscept = utils.connectivity_susceptibility(drive_data)
utils.print_avg_errors(drive_suscept)


# %%
# get the number of connected components for the ROADS dataset
num_cc_roads = utils.all_num_cc(roads_data, min_size=0)
print("roads dataset")
utils.print_num_cc(num_cc_roads)

#num_cc_roads_rm1 = utils.all_num_cc(roads_data, min_size=2)
#print("roads dataset Remove 1")
#print_num_cc(num_cc_roads_rm1)
#num_cc_roads_rm2 = utils.all_num_cc(roads_data, min_size=3)
#print("roads dataset Remove 2")
#print_num_cc(num_cc_roads_rm2)
#num_cc_roads_rm5 = utils.all_num_cc(roads_data, min_size=6)
#print("roads dataset Remove 5")
#print_num_cc(num_cc_roads_rm5)

roads_suscept = utils.connectivity_susceptibility(roads_data)
utils.print_avg_errors(roads_suscept)






# %%
