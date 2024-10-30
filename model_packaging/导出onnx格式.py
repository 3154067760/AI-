from torchvision import models
import torch
def resnet2onnx():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # 此
    # 处使用的是自带的权重
    model.eval()
    # 输入输出参数
    input_names = ['input']
    output_names = ['output']
    # 虚拟输入
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    '''
    model - torch.nn.Module，要导出的模型；
    args - 参数，模型的输入，
    path - 模型onnx导出的位置
    verbose - 是否打印模型打包的输出信息 默认Flase
    input_names - 输入节点名称
    output_names - 输出节点名称
    '''
    torch.onnx.export(model, x, 'models/resnet18.onnx', input_names=input_names,
    output_names=output_names, verbose=True)
if __name__ == "__main__":
    resnet2onnx()