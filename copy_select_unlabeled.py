import os

base_path = "~/unlabeled_notes"
usb = "/media/rbain/GH3"

offsets = [900, 2900, 3900, 6500, 8500, 9500, 14000, 15250]

for i_off in offsets:
    for i in range(1000):
        for j in range(5):
            path_suffix = "frame_" + str(i + i_off) + "_" + str(j) + ".jpg"
            copy_path = os.path.join(usb, path_suffix)
            image_path = os.path.join(base_path, path_suffix)
            os.system("cp " + str(image_path) + " " + str(copy_path))