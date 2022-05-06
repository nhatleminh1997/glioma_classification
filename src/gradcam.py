"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch

from misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    # def forward_pass_on_convolutions(self, x):
    #     """
    #         Does a forward pass on convolutions, hooks the function at given layer
    #     """
    #     conv_output = None
    #     for module_pos, module in self.model.features._modules.items():
    #         x = module(x)  # Forward
    #         if int(module_pos) == self.target_layer:
    #             x.register_hook(self.save_gradient)
    #             conv_output = x  # Save the convolution output on that layer
    #     return conv_output, x
    #
    # def forward_pass(self, x):
    #     """
    #         Does a full forward pass on the model
    #     """
    #     # Forward pass on the convolutions
    #     conv_output, x = self.forward_pass_on_convolutions(x)
    #     x = x.view(x.size(0), -1)  # Flatten
    #     # Forward pass on the classifier
    #     x = self.model.classifier(x)
    #     return conv_output, x
    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        if str(type(self.model)).__contains__('resnet'):
            for layer_name, module in self.model._modules.items():
                outer_layer_name = layer_name
                if len(
                        module._modules) > 0:  # Resnet "layer#" outerlayers have inner layers.  The "layer#" aren't operational
                    for bottleneck_num, btl_module in module._modules.items():  # Immediate inner layer is "BottleNeck". Again, not operational.  Here, layer_name = 'layer1' module = Sequential
                        bottleneck_number = bottleneck_num  # here bottleneck_num = '0' btl_module is BottleNeck
                        for inner_bottle_layer_name, inner_btl_module in btl_module._modules.items():  # These should be actual layers.  Here, inner_bottle_layer_name = 'conv1'. module=Conv2d
                            if inner_bottle_layer_name == 'downsample':  # These aren't real layers
                                continue
                            x = inner_btl_module(x)  # Forward
                            # print('*******\n{0}\n*******'.format(x.data))
                            print(
                                '{0}:\t{1}:\t{2}'.format(outer_layer_name, bottleneck_number, inner_bottle_layer_name))
                            print(x.data.sum())
                            if inner_bottle_layer_name == self.target_layer and outer_layer_name == 'layer4' and bottleneck_number == '2':
                                x.register_hook(self.save_gradient)
                                conv_output = x  # Save the convolution output on that layer
                                return conv_output, x  # Returning here should skip avgPool and fc as we desire
                else:  # No inner layers, outer layer is what we want
                    # print('*******\n{0}\n*******'.format(x.data))
                    x = module(x)  # Forward
                    print('{0}:\t{1}:\t{2}'.format(outer_layer_name, "NO_BOTTLENECK", layer_name))
                    print(x.data.sum())
                    if layer_name == self.target_layer:
                        x.register_hook(self.save_gradient)
                        conv_output = x  # Save the convolution output on that layer\
                        # if layer_name == 'avgpool': # Let's see if this makes a difference
                        return conv_output, x
            return conv_output, x

        else:  # AlexNet/VGG models
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                # if int(module_pos) == self.target_layer:
                if int(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
            return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)

        # Forward pass on the classifier
        if str(type(self.model)).__contains__('resnet'):
            # We need to replace with the exact layers as we don't have the .classifier and .features
            # x = x.view(x.size(0), -1)  # Flatten
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.model.fc(x)
            print('max_val: {0} index: {1}'.format(np.max(x.data.cpu().numpy()[0]), np.argmax(x.data.cpu().numpy()[0])))
        else:
            x = x.view(x.size(0), -1)  # Flatten
            x = self.model.classifier(x)
        return conv_output, x

    def generate_cam(self, input_image, target_index=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_index is None:
            target_index = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_index] = 1
        # Zero grads
        if str(type(self.model)).__contains__('resnet'):  # Resnet
            for layer_name, module in self.model._modules.items():  # For the inner models
                outer_layer_name = layer_name
                if len(
                        module._modules) > 0:  # Resnet "layer#" outerlayers have inner layers.  The "layer#" aren't operational
                    for bottleneck_num, btl_module in module._modules.items():  # Immediate inner layer is "BottleNeck". Again, not operational.  Here, layer_name = 'layer1' module = Sequential
                        bottleneck_number = bottleneck_num  # here bottleneck_num = '0' btl_module is BottleNeck
                        for inner_bottle_layer_name, inner_btl_module in btl_module._modules.items():  # These should be actual layers.  Here, inner_bottle_layer_name = 'conv1'. module=Conv2d
                            if inner_bottle_layer_name == 'downsample':  # These aren't real layers
                                continue
                            inner_btl_module.zero_grad()
                else:
                    module.zero_grad()  # for the outer modules
        else:  # AlexNet/VGG
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        self.model.conv1.zero_grad()
        self.model.bn1.zero_grad()
        self.model.relu.zero_grad()
        self.model.maxpool.zero_grad()
        self.model.layer1.zero_grad()
        self.model.layer2.zero_grad()
        self.model.layer3.zero_grad()
        self.model.layer4.zero_grad()
        self.model.avgpool.zero_grad()
        self.model.fc.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam


if __name__ == '__main__':
    # Get params
    target_example = 0  # recurrent
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer='fc')
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, 1)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
