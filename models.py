import segmentation_models_pytorch
print(f'SMP version: {segmentation_models_pytorch.__version__}\n')

import constant

manet = segmentation_models_pytorch.MAnet(
    encoder_name='timm-mobilenetv3_large_100',
    in_channels=constant.NUMBER_OF_INPUT_CHANNELS,
    classes=constant.NUMBER_OF_CLASSES,
    activation=None
)

deeplab_pp = segmentation_models_pytorch.DeepLabV3Plus(
    encoder_name='timm-mobilenetv3_large_100',
    in_channels=constant.NUMBER_OF_INPUT_CHANNELS,
    classes=constant.NUMBER_OF_CLASSES,
    activation=None
)

unet_pp = segmentation_models_pytorch.UnetPlusPlus(
    encoder_name='timm-mobilenetv3_large_100',
    in_channels=constant.NUMBER_OF_INPUT_CHANNELS,
    classes=constant.NUMBER_OF_CLASSES,
    activation=None
)

fpn = segmentation_models_pytorch.FPN(
    encoder_name='timm-mobilenetv3_large_100',
    in_channels=constant.NUMBER_OF_INPUT_CHANNELS,
    classes=constant.NUMBER_OF_CLASSES,
    activation=None
)

link_net = segmentation_models_pytorch.Linknet(
    encoder_name='timm-mobilenetv3_large_100',
    in_channels=constant.NUMBER_OF_INPUT_CHANNELS,
    classes=constant.NUMBER_OF_CLASSES,
    activation=None
)

psp = segmentation_models_pytorch.PSPNet(
    encoder_name='timm-mobilenetv3_large_100',
    in_channels=constant.NUMBER_OF_INPUT_CHANNELS,
    classes=constant.NUMBER_OF_CLASSES,
    activation=None
)

model_list = [
    'manet',
    'deeplab_pp',
    'unet_pp',
    'fpn',
    'link_net',
    'psp'
]

model_dict = {
    model_list[0]: manet,
    model_list[1]: deeplab_pp,
    model_list[2]: unet_pp,
    model_list[3]: fpn,
    model_list[4]: link_net,
    model_list[5]: psp
}
