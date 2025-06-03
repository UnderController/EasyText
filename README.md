# EasyTest

> **EasyText: Controllable
Diffusion Transformer for Multilingual Text
Rendering**
> <br>
> Runnan Lu, Yuxuan Zhang, Jiaming Liu,Haofan Wang  and Yiren Song.
> <br>
> National University of Singapore, The Chinese University of Hong Kong, Tiamat AI and Liblib AI.
> <br>


<br>

<img src='./assets/high_qually_sample_1.jpg' width='100%' />


## Quick Start
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






