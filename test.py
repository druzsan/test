import cv2
# from .models.vgg16 import VGG16 as net
# from .models.imagenet_utils import preprocess_input, decode_predictions

image = cv2.imread("./data/Aaron_Peirsol_0001.jpg")
cv2.imshow("image", image)
# cv2.waitKey(0)