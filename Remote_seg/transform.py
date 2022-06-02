#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image,ImageOps
import PIL.ImageEnhance as ImageEnhance
import random


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        dp = im_lb['dp']
        lb = im_lb['lb']
#        deep = im_lb['dp']
        assert im.size == lb.size
        assert im.size == dp.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im,dp=dp, lb=lb)
        if w < W or h < H:
            padh =  H - h if h < H else 0
            padw = W - w if w < W else 0
            im = ImageOps.expand(im, border=(int(padw/2), int(padh/2), padw-int(padw/2), padh-int(padh/2)), fill=0)
            lb = ImageOps.expand(lb, border=(int(padw/2), int(padh/2), padw-int(padw/2), padh-int(padh/2)), fill=255)
            dp = ImageOps.expand(dp, border=(int(padw/2), int(padh/2), padw-int(padw/2), padh-int(padh/2)), fill=0)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                dp = dp.crop(crop),
                lb = lb.crop(crop)
                    )


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            dp = im_lb['dp']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        dp = dp.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )

class VeticalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            dp = im_lb['dp']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_TOP_BOTTOM),
                        dp = dp.transpose(Image.FLIP_TOP_BOTTOM),
                        lb = lb.transpose(Image.FLIP_TOP_BOTTOM),
                    )
        
class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        dp = im_lb['dp']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    dp = dp.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                )
class fixScale(object):
    def __init__(self, scales=0.75, *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        dp = im_lb['dp']
        lb = im_lb['lb']
        W, H = im.size
        scale = self.scales
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    dp = dp.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                )
        
class outScale(object):
    def __init__(self, scales=0.125, *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = self.scales
        w, h = int(W * scale), int(H * scale)
        return dict(im = im,
                    lb = lb.resize((w, h), Image.NEAREST),
                )
        
class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        dp = im_lb['dp']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im = im,
                    dp = dp,
                    lb = lb,
                )


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb




if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
