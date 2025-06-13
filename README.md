<div align="center">
<h1>EasyText: Controllable Diffusion Transformer for Multilingual Text Rendering</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2505.24417-red)](https://arxiv.org/abs/2505.24417)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/songyiren725/EasyText)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/lllrrnn/EasyText)
<img src='./assets/high_qually_sample_1.jpg' width='100%' />
</div>

---

## 👥 Authors
>Runnan Lu<sup>1</sup>, Yuxuan Zhang<sup>2</sup>, Jiaming Liu<sup>3</sup>,Haofan Wang<sup>4</sup>  and Yiren Song<sup>1</sup>.
> 
> <sup>1</sup> National University of Singapore
> <sup>2</sup> The Chinese University of Hong Kong
> <sup>3</sup> Tiamat AI
> <sup>4</sup> Liblib AI
> 

---

## 🚀 Quick Start
### Configuration
#### 1. **Environment setup**
```bash
cd EasyText
conda create -n EasyText python=3.11.10
conda activate EasyText
```
#### 2. **Requirements installation**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt
```

### 3. Inference
We provided the integration of diffusers pipeline with our model. Simply run the inference script:

```
python inference_1024.py
```






