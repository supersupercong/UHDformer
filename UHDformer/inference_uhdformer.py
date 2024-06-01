import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.femasr_arch import FeMaSRNet
from basicsr.utils.download_util import load_file_from_url

import torch

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr

from comput_psnr_ssim import calculate_ssim as ssim_gray
from comput_psnr_ssim import calculate_psnr as psnr_gray

# def ssim_gray(imgA, imgB, gray_scale=True):
#     if gray_scale:
#         score, diff = ssim(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True,
#                            multichannel=False)
#     # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
#     else:
#         score, diff = ssim(imgA, imgB, full=True, multichannel=True)
#     return score
#
#
# def psnr_gray(imgA, imgB, gray_scale=True):
#     if gray_scale:
#         psnr_val = psnr(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY))
#         return psnr_val
#     else:
#         psnr_val = psnr(imgA, imgB)
#         return psnr_val


pretrain_model_url = {
    'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
    'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
}

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def equalize_hist_color(img):
    # 使用 cv2.split() 分割 BGR 图像
    channels = cv2.split(img)
    eq_channels = []
    # 将 cv2.equalizeHist() 函数应用于每个通道
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    # 使用 cv2.merge() 合并所有结果通道
    eq_image = cv2.merge(eq_channels)
    return eq_image

    # def get_residue_structure_mean(self, tensor, r_dim=1):
    #     max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    #     min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    #     res_channel = (max_channel[0] - min_channel[0])
    #     mean = torch.mean(tensor, dim=r_dim, keepdim=True)
    #
    #     device = mean.device
    #     res_channel = res_channel / torch.max(mean, torch.full(size=mean.size(), fill_value=0.000001).to(device))
    #     return res_channel

def get_residue_structure_mean(tensor, r_dim=1):
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = (max_channel[0] - min_channel[0])
    mean = torch.mean(tensor, dim=r_dim, keepdim=True)
    device = mean.device
    res_channel = res_channel / torch.max(mean, torch.full(size=mean.size(), fill_value=0.000001).to(device))
    return res_channel
import torch.nn.functional as F
def check_image_size(x,window_size=128):
    _, _, h, w = x.size()
    mod_pad_h = (window_size  - h % (window_size)) % (
                window_size )
    mod_pad_w = (window_size  - w % (window_size)) % (
                window_size)
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
    return x

def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', type=str, default='/data_8T1/wangcong/dataset/Rain13K/rain13ktest/Rain100H/input',
    #                     help='Input image or folder')
    # parser.add_argument('-g', '--gt', type=str, default='/data_8T1/wangcong/dataset/Rain13K/rain13ktest/Rain100H/target',
    #                     help='groundtruth image')
    # parser.add_argument('-i', '--input', type=str,
    #                     default='/data_8T1/wangcong/dataset/real-world-images/real-input',
    #                     help='Input image or folder')
    # parser.add_argument('-g', '--gt', type=str,
    #                     default='/data_8T1/wangcong/dataset/real-world-images/real-input',
    #                     help='groundtruth image')
    parser.add_argument('-i', '--input', type=str,
                        default='/data_8T1/wangcong/dataset/haze_dataset/4KID/test/input',
                        help='Input image or folder')
    parser.add_argument('-g', '--gt', type=str,
                        default='/data_8T1/wangcong/dataset/haze_dataset/4KID/test/gt',
                        help='groundtruth image')
    # parser.add_argument('-i', '--input', type=str,
    #                     default='/data_8T1/wangcong/dataset/LOLdataset/eval15/low',
    #                     help='Input image or folder')
    # parser.add_argument('-g', '--gt', type=str,
    #                     default='/data_8T1/wangcong/dataset/LOLdataset/eval15/high',
    #                     help='groundtruth image')
    # parser.add_argument('-w_vqgan', '--weight_vqgan', type=str,
    #                     default='/data_8T1/wangcong/net_g_260000.pth',
    #                     help='path for model weights')
    parser.add_argument('-w', '--weight', type=str,
                        default='./experiments/014_FeMaSR_LQ_stage/models/net_g_600000.pth',
                        help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results/UHD', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600,
                        help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.weight is None:
    #     weight_path_vqgan = load_file_from_url(pretrain_model_url[f'x{args.out_scale}'])
    # else:
    #     weight_path_vqgan = args.weight_vqgan
    enhance_weight_path = args.weight
    # print('weight_path', weight_path_vqgan)
    # set up the model
    # VQGAN = FeMaSRNet(codebook_params=[[16, 1024, 256], [32, 1024, 128], [64, 1024, 64], [128, 1024, 32]], LQ_stage=False, scale_factor=args.out_scale).to(device)
    # VQGAN.load_state_dict(torch.load(weight_path_vqgan)['params'], strict=False)
    # VQGAN.eval()

    EnhanceNet = FeMaSRNet(number_block=5,
                           unit_num=3,
                 num_heads=8,
                 match_factor=4,
                 ffn_expansion_factor=4,
                 scale_factor=8,
                bias=True,
                LayerNorm_type='WithBias',
                           attention_matching=True,
                           ffn_matching=True,
                           ffn_restormer=False,
                           ).to(device)
    EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=False)
    EnhanceNet.eval()
    print_network(EnhanceNet)
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    ssim_all = 0
    psnr_all = 0
    lpips_all = 0
    num_img = 0
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        gt_path = args.gt
        file_name = path.split('/')[-1]

        gt_img = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED)
        print('image name', path)
        # print(gt_img)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)
        b, c, h, w = img_tensor.size()
        print('b, c, h, w = img_tensor.size()', img_tensor.size())
        img_tensor = check_image_size(img_tensor)
        # self.gt_rec, feature_degradation, restoration
        # with torch.no_grad():
        #     _, feature_degradation = VQGAN.VQGAN(img_tensor)

        with torch.no_grad():
            import time
            t0 = time.time()
            output = EnhanceNet.restoration_network(img_tensor)
            t1 = time.time()
            print('time:', t1-t0)
        output = output
        # output = sr_model.test(img_tensor, rain = img_tensor-output)
        # else:
        #     output = sr_model.test_tile(img_tensor)
        # output_img = output['out_final']

        # [2, 1, 0]
        # output_first = tensor2img(output_first)
        output = output[:, :, :h, :w]
        output_img = tensor2img(output)
        gray = True
        # ssim = ssim_gray(output_img, gt_img, gray_scale=gray)
        # psnr = psnr_gray(output_img, gt_img, gray_scale=gray)
        ssim = ssim_gray(output_img, gt_img)
        psnr = psnr_gray(output_img, gt_img)
        lpips_value = lpips(2 * torch.clip(img2tensor(output_img).unsqueeze(0) / 255.0, 0, 1) - 1,
                            2 * img2tensor(gt_img).unsqueeze(0) / 255.0 - 1).data.cpu().numpy()
        ssim_all += ssim
        psnr_all += psnr
        lpips_all += lpips_value
        num_img += 1
        print('num_img', num_img)
        print('ssim', ssim)
        print('psnr', psnr)
        print('lpips_value', lpips_value)
        save_path = os.path.join(args.output, f'{img_name}')
        # save_path_first = os.path.join(args.output + 'first/', f'{img_name}')
        imwrite(output_img, save_path)

        pbar.update(1)
    pbar.close()
    print('avg_ssim:%f' % (ssim_all / num_img))
    print('avg_psnr:%f' % (psnr_all / num_img))
    print('avg_lpips:%f' % (lpips_all / num_img))


if __name__ == '__main__':
    main()
