# UHDformer 

This is the official PyTorch codes for the paper  
[Correlation Matching Transformation Transformers for UHD Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/28341)  
[Cong Wang](https://scholar.google.com/citations?user=0DrHHRwAAAAJ&hl=zh-CN), [Jinshan Pan](https://jspan.github.io/), [Wei Wang](http://yipengqin.github.io/), [Gang Fu](https://scholar.google.com/citations?hl=zh-CN&user=2k1Hcd4AAAAJ), [Siyuan Liang](https://scholar.google.com/citations?hl=zh-CN&user=MLE3GekAAAAJ), Mengzhu Wang, [Xiao-Ming Wu](https://www4.comp.polyu.edu.hk/~csxmwu/), [Jun Liu](https://scholar.google.com/citations?hl=zh-CN&user=Q5Ild8UAAAAJ)

## Overall of UHDformer
![framework_img](imgs/overall.png)

## Dual-path Correlation Matching Transformation
![DualCMT](imgs/dual.png)

## Main Results

### Low-light Image Enhancement on UHD-LL
<img src="imgs/low-light.png" width="50%">

### Image Dehazing on UHD-Haze
<img src="imgs/dehazing.png" width="50%">

### Image Deblurring on UHD-Blur
<img src="imgs/deblurring.png" width="50%">

## Dependencies and Installation

- Ubuntu >= 18.04
- CUDA >= 11.0
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/supersupercong/UHDformer.git
cd UHDformer 

# create new anaconda env
conda create -n uhdformer python=3.8
source activate uhdformer 

# install python dependencies
pip3 install -r requirements.txt
python setup.py develop
```

## Datasets Download

[UHD-LL](https://drive.google.com/drive/folders/1yJTf874-rrBfgxlmElkGoOYxmu7jZMh4?usp=sharing), [UHD-Haze](https://drive.google.com/drive/folders/1EAHC8UM3HwrI2O-AHFDXpoRRCsyXXUTz?usp=sharing), [UHD-Blur](https://drive.google.com/drive/folders/18kYF-Apj_KBXc6prO--xvN6zoMZ7r_8j?usp=sharing)

## Pre-trained Model

[UHD-LL](https://drive.google.com/drive/folders/1XwIWOjOepUA-kXoaeOlzMDijVK4LMj-1?usp=sharing), UHD-Haze, UHD-Blur

### Train

```
bash train.sh
```

### Test

```
bash test.sh
```


## Citation
```
@inproceedings{wang2024uhdformer,
      author={Cong Wang and Jinshan Pan and Wei Wang and Gang Fu and Siyuan Liang and Mengzhu Wang and Xiao-Ming Wu and Jun Liu},
      title={Correlation Matching Transformation Transformers for UHD Image Restoration}, 
      year={2024},
      Journal = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Acknowledgement

This project is based on [FeMaSR](https://github.com/chaofengc/FeMaSR).
