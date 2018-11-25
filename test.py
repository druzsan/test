import cv2
from models.vgg16 import VGG16 as net
from models.imagenet_utils import preprocess_input, decode_predictions

model = net()

image = cv2.imread("./data/Aaron_Peirsol_0001.jpg")
# cv2.imshow("image", image)
image = cv2.resize(image, (224, 224))
# cv2.imshow("resized image", image)
# cv2.waitKey(0)

input = preprocess_input(image)
output = model.predict(input)
prediction = decode_predictions(output)
print(prediction)