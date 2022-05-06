# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
from PIL import ImageFont, ImageDraw, Image
import requests
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
from config import parse_opts
import pdb
import factory.model_factory as model_factory
import nibabel as nib
import os
import matplotlib.pyplot as plt
from skimage.transform import rotate
import SimpleITK as sitk
config = parse_opts()

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 3
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    config = parse_opts()
    net, _ = model_factory.get_model(config)
    state_dict = torch.load('D:\\PyTorchConv2D\\checkpoints\\densenet\\save_best_084.pth')['state_dict']
    net.load_state_dict(state_dict)
    finalconv_name = 'features'

elif model_id == 4:
    config = parse_opts()
    net, _ = model_factory.get_model(config)
    state_dict = torch.load('D:\\PyTorchConv2D\\checkpoints\\save_best_087.pth')['state_dict']
    net.load_state_dict(state_dict)
    finalconv_name = 'layer4'


net.eval()


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
def fill_small_holes (im_th):
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out.astype(np.bool)
def load_mri_dataset(running_instance):
    array_3d = np.zeros([5, 256, 256]).astype(np.float32)
    sequences = ['apt.nii', 'T1.nii', 'T1c.nii', 'T2.nii', 'Flair.nii']
    for sequence in sequences:
        if sequence == 'apt.nii':
            count = 0
        elif sequence == 'T1.nii':
            count = 1
        elif sequence == 'T1c.nii':
            count = 2
        elif sequence == 'T2.nii':
            count = 3
        elif sequence == 'Flair.nii':
            count = 4
        img_path = running_instance + sequence
        img = nib.load(img_path)
        data = img.get_fdata().astype(np.float32)
        data = data * 2 - 1
        if len(data.shape) > 2:
            data = np.squeeze(data)
        array_3d[count, :, :] = data
    im_as_ten = torch.from_numpy(array_3d).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)

    return im_as_var, array_3d


root = 'D:\\PyTorchConv2D\\saliency_example_0818_densenet'
label_folder = 5

classes = { 0 : 'negative',
            1 : 'postive'}

example_list = [(os.path.join(root,str(label_folder),o),label_folder) for o in os.listdir(os.path.join(root, str(label_folder))) if o.endswith('T2.nii')]

for i in range(len(example_list)):
    # hook the feature extractor
    features_blobs = []
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    example_index = i
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('\\')+1:img_path.rfind('_')]
    img_path_pattern =  img_path[0:img_path.rfind('\\')+1] + file_name_to_export + '_'
    # Read image
    img_variable, img_array = load_mri_dataset(img_path_pattern)
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(2):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    height, width, _ = 256, 256, 3


    # render the CAM and output
    print('output '+ file_name_to_export+ '_CAM.jpg' + 'for the top1 prediction: %s'%classes[idx[0]])

    T1c = ((img_array[2,:,:] + 1)/2 * 255).astype(np.uint8)
    # save original cam
    cam_or = cv2.resize(CAMs[0], (width, height))
    #give up out brain region
    cam_or_mask = fill_small_holes((T1c > 0).astype(np.uint8))
    cam_or_masked = np.multiply(cam_or, cam_or_mask)
    index = height*width-np.sum(cam_or_mask)
    cam_or = np.rot90(cam_or_masked,3)
    cam_or = np.fliplr(cam_or)
    itk_cam = sitk.GetImageFromArray(cam_or)
    save_path = os.path.join(root, str(label_folder), file_name_to_export + '_CAM_or_masked.nii')
    sitk.WriteImage(itk_cam, save_path)

    img = cv2.cvtColor(T1c, cv2.COLOR_GRAY2RGB)

    #find 95 percentle mask
    array = CAMs[0].flatten()
    sorted_array = np.sort(array)
    mask = (CAMs[0] >= sorted_array[int(sorted_array.shape[0]*0.90)]).astype(np.uint8)
    # resize mask to out shape
    mask = cv2.resize(mask, (width, height))
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    # apply mask to heatmap
    heatmap_masked = np.zeros((height,width,3))
    for i in range(heatmap.shape[2]):
        heatmap_masked[:,:,i] = np.multiply(heatmap[:,:,i],mask)

    result = heatmap_masked * 0.5 + img * 0.5
    # match the direction
    data = rotate(result, 270, resize=False)
    result = np.fliplr(data).astype(np.uint8)
    save_path = os.path.join(root, str(label_folder),file_name_to_export+'_CAM_masked.jpg')



    cv2_im_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)

    # Draw the text
    draw.text((0, 0), '{:.5f} -> {}'.format(probs[0], classes[idx[0]]))
    # Save the image
    pil_im.save(save_path)

    # save original heatmap
    result = heatmap * 0.5 + img * 0.5
    # match the direction
    data = rotate(result, 270, resize=False)
    result = np.fliplr(data).astype(np.uint8)
    save_path = os.path.join(root, str(label_folder), file_name_to_export + '_CAM.jpg')

    cv2_im_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)

    # Draw the text
    draw.text((0, 0), '{:.5f} -> {}'.format(probs[0], classes[idx[0]]))
    # Save the image
    pil_im.save(save_path)


