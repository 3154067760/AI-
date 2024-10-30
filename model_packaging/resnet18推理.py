import cv2
import numpy as np
import onnxruntime
from PIL import Image
from torchvision import models, transforms
import torch
import onnx

t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

def resnet2onnx(img_path):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    input = pre_process(img_path)
    input = torch.tensor(input, dtype=torch.float32)
    out = model(input)
    torch.onnx.export(model, x, 'models/resnet18.onnx', input_names=['input'],
                      output_names=['output'])

def pre_process(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(224, 224))
    img = img / 255
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32)
    return img

def loadOnnx(img_path):
    session = onnxruntime.InferenceSession('./models/resnet18.onnx',
                                           providers=[
                                               'CPUExecutionProvider',
                                               'CUDAExecutionProvider',
                                           ])
    input_x = pre_process(img_path)
    session_out = session.run(None, {'input': input_x})
    # 假设输出是分类结果，可以进行预测
    output = session_out[0]
    predicted_class = np.argmax(output)
    return predicted_class

if __name__ == '__main__':
    # img_path = './imgs/img.png'
    # predicted_class = loadOnnx(img_path)
    # print(f"预测的类别为：{predicted_class}")
# 补充:image = cv.dnn.blobFromImage(bgr, 1 / 255.0, (640, 640), swapRB=True,crop=False)
# cv.dnn.blobFromImage 函数将输入图像转换为适合深度学习网络的格式。参数的作用如下：
# bgr ：输入图像，通常是 BGR 格式的图像（OpenCV 的默认格式）。
# 1 / 255.0 ：将图像的像素值归一化到 [0, 1] 范围内（因为图像的像素值通常在 [0, 255] 范围
# 内）。
# (640, 640) ：指定输出 blob 的尺寸。在这里，将图像调整为 640x640 像素。
    import cv2 as cv
    import numpy as np

    # 读取一张图像（这里假设是一张彩色图像）
    image = cv.imread('imgs/img.png')
    blob = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(224, 224), mean=(0, 0, 0), swapRB=True, crop=False)# 1 是由函数内部的默认设置
    print("原始图像形状:", image.shape)
    print("Blob数据形状:", blob.shape)
    #原始图像形状: (190, 263, 3)
    #Blob数据形状: (1, 3, 224, 224)