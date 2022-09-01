import torch
import random
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Resize,ColorJitter
from scipy import io
import cv2
import torchvision.transforms.functional as F
import numbers
# import phaseWaugment

def scale_image(scale, img, target, img_padding=0, target_padding=0):
    """
    resize image with unchanged aspect ratio using padding
    """
    in_w, in_h = img.shape[1], img.shape[0]

    new_w = int(in_w * scale)  # （new_w, new_h）是高宽比不变的新的尺度
    new_h = int(in_h * scale)

    resized_image = np.array(Image.fromarray(img.astype(np.uint8)).resize((new_w, new_h)))
    resized_target = np.array(Image.fromarray(target.astype(np.uint8)).resize((new_w, new_h)))

    # print(np.where(resized_target==255))

    if(scale<1):
        # 定义画布canvas，然后把有效区域resized_image填充进去
        img_canvas = np.full((in_w, in_h), img_padding, np.uint8)  # 定义画布
        target_canvas = np.full((in_w, in_h), target_padding, np.uint8)  # 定义画布
        # 画布_h - 高宽不变_new_h
        start_h = (in_w - new_h) // 2  # 开始插入的高度位置, 也是上下需要padding的大小
        start_w = (in_h - new_w) // 2  # 开始插入的宽度位置
        img_canvas[start_h:start_h + new_h, start_w:start_w + new_w] = resized_image
        target_canvas[start_h:start_h + new_h, start_w:start_w + new_w] = resized_target

        return img_canvas, target_canvas
    else:
        offset_y = np.random.randint(new_h - in_h + 1)
        offset_x = np.random.randint(new_w - in_w + 1)
        cropped_image = resized_image[offset_y:offset_y + in_h, offset_x:offset_x + in_w]
        cropped_target = resized_target[offset_y:offset_y + in_h, offset_x:offset_x + in_w]
        # print(np.where(cropped_image==1))
        # print(np.where(cropped_image==255))
        return cropped_image.astype(np.uint8), cropped_target.astype(np.uint8)

def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64, 128])
    cmap[1,:] = np.array([244, 35, 232])
    # cmap[2,:] = np.array([70,  70,  70])
    # cmap[3,:] = np.array([102, 102, 156])
    # cmap[4,:] = np.array([190, 153, 153])
    # cmap[5,:] = np.array([153, 153, 153])

    # cmap[6,:] = np.array([ 250,170, 30])
    # cmap[7,:] = np.array([ 220,220,  0])
    # cmap[8,:] = np.array([ 107,142, 35])
    # cmap[9,:] = np.array([ 152,251,152])
    # cmap[10,:] = np.array([ 70,130,180])

    # cmap[11,:] = np.array([ 220, 20, 60])
    # cmap[12,:] = np.array([ 255,  0,  0])
    # cmap[13,:] = np.array([ 0,  0,142])
    # cmap[14,:] = np.array([  0,  0, 70])
    # cmap[15,:] = np.array([  0, 60,100])
    # cmap[16, :] = np.array([0, 80, 100])

    #cmap[17, :] = np.array([0, 0,  0])
    #cmap[16, :] = np.array([0, 80, 100])
    #cmap[17, :] = np.array([0, 0, 230])
    #cmap[18, :] = np.array([119, 11, 32])
    #cmap[19, :] = np.array([110, 11, 30])

    #cmap[20, :] = np.array([124, 252, 0])
    #cmap[21, :] = np.array([218, 165, 32])
    #cmap[22, :] = np.array([210, 105, 30])
    #cmap[23, :] = np.array([178, 34, 34])
    #cmap[24, :] = np.array([169, 169, 169])
    #cmap[25, :] = np.array([128, 0, 0])
    #cmap[26, :] = np.array([192, 14, 235])
    #cmap[27, :] = np.array([184, 134, 11])
    #cmap[28, :] = np.array([255, 215, 0])
    #cmap[29, :] = np.array([128, 128, 0])

    #cmap[30, :] = np.array([0, 0, 0])

    #cmap[16,:] = np.array([  0, 80,100])
    #cmap[17,:] = np.array([  0,  0,230])
    #cmap[18,:] = np.array([ 119, 11, 32])
    #cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)



class ToLabelFloat:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).unsqueeze(0).type(torch.FloatTensor)


class Colorize:

    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        #print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input, target):
        for t in self.transforms:
            input, target = t(input, target)
        return input, target


class Standard(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input, target):
        for tensor in input:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return input, target


class MyTotensor(object):

    def __call__(self, input, target):
        input = ToLabelFloat()(input)
        target = ToLabelFloat()(target)
        return input, target


class phaseW_augment(object):

    def __call__(self, input, target):
        input = np.array(input)
        if(input.ndim==3):
            input[:,:,0] = phaseWaugment.phaseW_augment(input[:,:,0],fringe_rotation='h')
        else:
            input = phaseWaugment.phaseW_augment(input,fringe_rotation='h')
        input = Image.fromarray(np.uint8(input))
        return input,target

class Normalize(object):

    def __call__(self, input, target):
        input[:,:,0] = torch.clamp(input[:,:,0],0,1)
        input[:,:,1] = torch.clamp(input[:,:,1]*100/89,0,1)
        input[:,:,2] = torch.clamp(input[:,:,2]*100/98,0,1)
        return input, target

class RandomScaleCrop(object):
    def __call__(self, input, target):
        
        scale  = np.random.uniform(1, 1.2, 1)
        input = np.array(input).astype(np.uint8)

        target = np.array(target).astype(np.uint8)
        # 3 channels
        # target = target
        # print(np.where(target==2))
        # print(np.where(target==254))
        # io.savemat('target.mat', {'array':  target})
        cropped_input, cropped_target =  scale_image(scale, input, target, img_padding=0, target_padding=0)
        # print(np.where(cropped_target==1))
        # print(np.where(cropped_target==254))
        # # print(np.where(cropped_input==1))
        # io.savemat('cropped_target.mat', {'array':  cropped_target})
        cropped_input = Image.fromarray(cropped_input) 
        cropped_target = Image.fromarray(cropped_target) 
        return cropped_input, cropped_target


class ToTensor_and_Resize_EncodeTarget(object):
    def __init__(self, enc):
        self.enc=enc
        pass
    def __call__(self, input, target):
   
        input = ToTensor()(input)
        if (self.enc):
            target = Resize([128,128], Image.NEAREST)(target)
        # class =2 
        target = ToLabel()(target)    #将数组转为张量
        target = Relabel(254, 2)(target) #将标签中的255转为1
        target = Relabel(127, 1)(target) #将标签中的255转为1
        # target = Relabel(1, 0)(target) #将标签中的255转为1
        # target = Relabel(2, 1)(target) #将标签中的255转为1
        return input, target

class ToTensor_and_Resize_EncodeTarget_shift(object):
    def __init__(self, enc):
        self.enc=enc
        pass
    def __call__(self, input, target):
        input = ToTensor()(input)
        if (self.enc):
            target = Resize([32,32], Image.NEAREST)(target)
        # class =2 
     
        target = ToLabel()(target)    #将数组转为张量
        # target = target/127
        target = Relabel(254, 2)(target) #将标签中的255转为1
        target = Relabel(127, 1)(target) #将标签中的255转为1
        # target = Relabel(1, 0)(target) #将标签中的255转为1
        # target = Relabel(2, 1)(target) #将标签中的255转为1
        return input, target

class ToTensor_and_Resize_EncodeTarget_shift_channels(object):
    def __init__(self, enc):
        self.enc=enc
        pass
    def __call__(self, input, target):
        input = ToTensor()(input)
        if (self.enc):
            target = Resize([64,64], Image.NEAREST)(target)
        # class =2 
     
        target = ToLabel()(target)    #将数组转为张量
        # target = target/127
        target = Relabel(254, 2)(target) #将标签中的255转为1
        target = Relabel(127, 1)(target) #将标签中的255转为1
        # target = Relabel(1, 0)(target) #将标签中的255转为1
        # target = Relabel(2, 1)(target) #将标签中的255转为1
        return input, target


class ToTensor_and_Resize_EncodeTarget_c2(object):
    def __init__(self, enc):
        self.enc=enc
        pass
    def __call__(self, input, target):
        input = ToTensor()(input)

        if (self.enc):
            target = Resize([128,128], Image.NEAREST)(target)
        # class =2 
     
        target = ToLabel()(target)    #将数组转为张量
        target = Relabel(255, 1)(target) #将标签中的255转为1
        return input, target


# random rotation 90 180 270
class RandomRotation(object):
    def __call__(self, input, target):
        p = random.random()
        if(p<1./3):
            input = input.transpose(Image.ROTATE_90)
            target = target.transpose(Image.ROTATE_90)
        elif(p>2./3):
            input = input.transpose(Image.ROTATE_270)
            target = target.transpose(Image.ROTATE_270)
        else:
            input = input.transpose(Image.ROTATE_180)
            target = target.transpose(Image.ROTATE_180)
        return input, target

class RandomAngleRotation(object):
    def __call__(self, input, target):
        # 随机旋转
        angel = random.randint(0,359)
        input = input.rotate(angel,fillcolor=0)
        target = target.rotate(angel,fillcolor=0)
        return input, target


class MaskErode(object):
    def __call__(self, input, target):
        # target.show()
        target = np.array(target).astype(np.uint8)
        # print(np.where(target==255))
        # # *255
        target = cv2.cvtColor(target, 1)
        # b.设置卷积核5*5
        kernel = np.ones((3,3),np.uint8)
        # kernel2 = np.ones((3,3),np.uint8)
        # c.图像的腐蚀，默认迭代次数
        erosion = cv2.erode(target,kernel)
        # erosion2 = cv2.erode(target,kernel2)
        # 效果展示
        # cv2.imshow('origin',src)
        ## 腐蚀后
        # cv2.imshow('after erosion',erosion)
        target = Image.fromarray(erosion,mode="RGB").convert('L')
        # target2 = Image.fromarray(erosion2,mode="RGB").convert('L')
        # target.show()
        # input.show()
        # target2.show()
        return input, target


class RandomHorizontalFlip(object):
    def __call__(self, input, target):
        p = random.random()
        if(p<0.5):
            input = input.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        return input, target

class RandomVerticalFlip(object):
    def __call__(self, input, target):
        p = random.random()
        if(p<0.5):
            input = input.transpose(Image.FLIP_TOP_BOTTOM)
            target = target.transpose(Image.FLIP_TOP_BOTTOM)

        return input, target


# class PmgColorJitter(object):
#     def __call__(self, input, target):
#         [

#         return input, target
class PmgColorJitter(torch.nn.Module):
    # a = ColorJitter()
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        phaseW = np.array(img)[:,:,0]
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)
        img=np.array(img)
        img[:,:,0]=phaseW
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img, target

