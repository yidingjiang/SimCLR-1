# batch crop based on PyTorch
import torch
import torch.nn.functional as F
import numpy as np

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), mode='bilinear'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.mode = mode
        self.scale = scale
        self.ratio = ratio
    
    def get_params(self, img, scale, ratio, shape=None):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (tensor): Image to be cropped. (B, C, H, W)
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        if img is not None:
            B, _, width, height = img.shape
        else:
            assert shape is not None, "no images are being passed. Please pass some shape."
            B = shape[0]
            height = shape[-2]
            width = shape[-1]

        area = height * width

        top_left    = []
        top_right   = []
        h_vals      = []
        w_vals      = []

        itr = 0

        while len(top_left) < B:
            target_area = np.random.uniform(low=scale[0], high=scale[1], size=B) * area
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            aspect_ratio = np.exp(np.random.uniform(low=log_ratio[0], high=log_ratio[1], size=B))

            w = np.round(np.sqrt(target_area * aspect_ratio)).astype(int)
            h = np.round(np.sqrt(target_area / aspect_ratio)).astype(int)

            w_idx = w[(w < width) & (h < height)]
            h_idx = h[(w < width) & (h < height)]

            for i in range(len(w_idx)): 
                top_left.append(np.random.randint(0, height - h_idx[i]))
                top_right.append(np.random.randint(0, width - w_idx[i]))

            h_vals.append(h_idx)
            w_vals.append(w_idx)
            
            itr +=1
        # print("Total iterations ", itr)
        top_left = np.array(top_left)
        top_right = np.array(top_right)
        h_vals = np.concatenate(h_vals)
        w_vals = np.concatenate(w_vals)
        return top_left[:B], top_right[:B], h_vals[:B], w_vals[:B]

    def generate_parameters(self, shape):
        i, j, h, w = self.get_params(img=None, scale=self.scale, ratio=self.ratio, shape=shape)
        return {"top_x": i, "top_y": j, "height": h, "width": w}

    def __call__(self, img):
        """
        Args:
            img (tensor): batch of image tensors to be cropped and resized.

        Returns:
            image tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = crop_and_resize(img, i, j, h, w, self.size, self.mode)
        return img

def crop_and_resize(images, top_left, top_right, height, width, size, interpolation='bilinear'):
    # image: (B, C, H, W)
    B = images.shape[0]
    resized = []
    for i in range(B):
        cropped = images[i, :, top_left[i] : top_left[i] + height[i], top_right[i] : top_right[i] + width[i]]
        #import pdb; pdb.set_trace()
        resized.append(F.interpolate(cropped.unsqueeze(0), size=size, mode=interpolation))
    resized = torch.cat(resized)
    return resized