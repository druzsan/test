import os
import cv2
import numpy as np

test_dirpath = "data/small_dataset/cropped/test"
test_pairs_filepath = "data/small_dataset/cropped/testPairs"
test_res_filepath = "data/small_dataset/cropped/res3"

imagepaths = []
with open(test_pairs_filepath, "r") as test_pairs_file:
    for line in test_pairs_file.readlines():
        imagepaths.append([os.path.join(test_dirpath, imagename) for imagename in line.split()[:2]])

scores = []
with open(test_res_filepath, "r") as test_res_file:
    for line in test_res_file.readlines():
        scores.append(float(line))
scores = np.array(scores)

if len(imagepaths) != len(scores):
    print("The number of similarity scores doesn\'t match the number of image pairs. Exit")
    exit(0)

for i in range(len(imagepaths)):
    merged_image = np.concatenate([cv2.imread(imagepaths[i][0]), cv2.imread(imagepaths[i][1])], axis=1)
    score = scores[i]
    cv2.imshow(str(score), merged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()