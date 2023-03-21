![Asset 5](https://user-images.githubusercontent.com/13435425/222425743-213279f9-d0a1-413c-a16a-2c88b512f827.png)

### :rocket: Diffusion models are making the headlines as a new generation of powerful generative models.

However, many of the ongoing research considers solutions that are quite often **quite specific** and **require large computational resources for training**.

:beginner: **DiffusionFastForward** offers a general template for diffusion models for images that can be a starting point for understanding and researching diffusion-based generative models.
* :zap: **PyTorch Lightning** to enable easy training!
* :money_with_wings: You can run all experiments online on Google colab - **no need for own GPU machine**!
* :mag_right: Examples for **both low-resolution and high-resolution** data!
* :tent: Examples of **latent diffusion**!
* :art: Examples of **image translation** with diffusion!

The code structure is simple, so that you can easily customize it to your own applications.

> :construction: Disclaimer: This repository does not provide any weights to the models. The purpose of this software is to be able to train **new** weights on a previously unexplored type of data.

## Contents

There are three elements integrated into this project:
* :computer: **Code**
* :bulb: **Notes** (in `notes` directory)
* :tv: **Video Course** (to be released on YouTube)

---
### :computer: [**Code**](https://github.com/mikonvergence/DiffusionFastForward/tree/master/src)
This repository offers a starting point for training diffusion models on new types of data. It can serve as a baseline that can hopefully be developed into more robust solutions based on the specific features of the performed generative task.

It includes notebooks that can be run stand-alone:
1. [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikonvergence/DiffusionFastForward/blob/master/01-Diffusion-Sandbox-colab.ipynb) [01-Diffusion-Sandbox](https://github.com/mikonvergence/DiffusionFastForward/blob/master/01-Diffusion-Sandbox.ipynb)  - visualizations of the diffusion process 
2. [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikonvergence/DiffusionFastForward/blob/master/02-Pixel-Diffusion-colab.ipynb) [02-Pixel-Diffusion](https://github.com/mikonvergence/DiffusionFastForward/blob/master/02-Pixel-Diffusion.ipynb) - basic diffusion suitable for low-resolution data
3. [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikonvergence/DiffusionFastForward/blob/master/03-Conditional-Pixel-Diffusion-colab.ipynb) [03-Conditional-Pixel-Diffusion](https://github.com/mikonvergence/DiffusionFastForward/blob/master/03-Conditional-Pixel-Diffusion.ipynb) - image translation with diffusion for low-resolution data
4. [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikonvergence/DiffusionFastForward/blob/master/04-Latent-Diffusion-colab.ipynb) [04-Latent-Diffusion](https://github.com/mikonvergence/DiffusionFastForward/blob/master/04-Latent-Diffusion.ipynb) - latent diffusion suitable for high-resolution data
5. [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikonvergence/DiffusionFastForward/blob/master/05-Conditional-Latent-Diffusion-colab.ipynb) [05-Conditional-Latent-Diffusion](https://github.com/mikonvergence/DiffusionFastForward/blob/master/05-Conditional-Latent-Diffusion.ipynb) - image translation with latent diffusion

### Dependencies
Assuming `torch` and `torchvision` is installed:
```bash
pip install pytorch-lightning==1.9.3 diffusers einops
```

---
### :bulb: [**Notes**](https://github.com/mikonvergence/DiffusionFastForward/tree/master/notes)
Short summary notes are released as part of this repository and they overlap semantically with the notebooks!
1. [01-Diffusion-Theory](https://github.com/mikonvergence/DiffusionFastForward/blob/master/notes/01-Diffusion-Theory.md) - visualizations of the diffusion process
2. [02-Pixel-Diffusion](https://github.com/mikonvergence/DiffusionFastForward/blob/master/notes/02-Pixel-Diffusion.md) - basic diffusion suitable for low-resolution data
3. [03-Conditional-Pixel-Diffusion](https://github.com/mikonvergence/DiffusionFastForward/blob/master/notes/03-Conditional-Pixel-Diffusion.md) - image translation with diffusion for low-resolution data
4. [04-Latent-Diffusion](https://github.com/mikonvergence/DiffusionFastForward/blob/master/notes/04-Latent-Diffusion.md) - latent diffusion suitable for high-resolution data
5. [05-Conditional-Latent-Diffusion](https://github.com/mikonvergence/DiffusionFastForward/blob/master/notes/05-Conditional-Latent-Diffusion.md) - image translation with latent diffusion

---
### :tv: [**Video Course**](https://youtube.com/playlist?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN) (released on YouTube)
The course is released on YouTube and provides an extension to this repository. Some additional topics are covered, such as seminal papers and on-going research work.

[<img width="1596" alt="Screenshot 2023-03-01 at 19 46 20" src="https://user-images.githubusercontent.com/13435425/222248673-bfcce06c-0f5b-421b-92b2-b4ed130c0dfb.png">](https://youtube.com/playlist?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)

The current plan for the video course (links added upon publishing):
* :tv: [#00 Introduction](https://youtu.be/sXBPyKycOX8?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)
* :tv: [#01 Basics: Denoising Diffusion Process](https://youtu.be/7d73Y30viC8?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)
* :tv: [#02 Basics: Denoising Diffusion of Images](https://youtu.be/Whykds7oWQI?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)
* :tv: [#03 Practical: Unconditional Diffusion in Low-Resolution](https://youtu.be/eVYrHiwqBbE?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)
* :soon: [#04 Extra: Summary of Seminal Works](/)
* :tv: [#05 Basics: Conditional Diffusion](https://youtu.be/KuKeM_NbdSM?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)
* :tv: [#06 Practical: Condition Diffusion in Low-Resolution](https://youtu.be/C-HjC-Kh4M0?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)
* :tv: [#07 Basics: High-Resolution Diffusion](https://youtu.be/QaasB5PYMYs?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)
* :tv: [#08 Practical: High-Resolution Diffusion](https://youtu.be/SPOIBtYEIxw?list=PL5RHjmn-MVHDMcqx-SI53mB7sFOqPK6gN)
* :soon: [#09 Extra: Diffusion Applications](/)
* :soon: [#10 Extra: Further Insight into Diffusion](/)

---

### :moneybag: Training Cost

Most examples are one of two types of models, trainable within a day:

**PixelDiffusion (Good for small images :baby:)**
Appropriate for LR data. Direct diffusion in pixel space.
| **Image Resolution** | 64x64 |
|---|---|
| **Training Time** | ~10 hrs |
| **Memory Usage**  | ~4 GB |

![out-pixel-conditional-1](https://user-images.githubusercontent.com/13435425/216780421-3613e80f-312b-4fed-8e24-9be22902a40f.png)

![out-pixel-conditional-2](https://user-images.githubusercontent.com/13435425/216780427-5f737606-6b34-4477-88d4-187b863df03a.png)

![out-pixel-conditional-3](https://user-images.githubusercontent.com/13435425/216780431-e3cf6a78-edc4-4ec6-9d3f-8a38aa01d5e0.png)

**LatentDiffusion (Good for large images :whale2:)**
Useful for HR data. Latent diffusion in compressed space.
| **Image Resolution** | 256x256 |
|---|---|
| **Training Time** | ~20 hrs |
| **Memory Usage**  | ~5 GB |

![out-latent-conditional-1](https://user-images.githubusercontent.com/13435425/216780511-1f2d60ec-2d7e-4b5c-b851-c330687e55c3.png)

![out-latent-conditional-2](https://user-images.githubusercontent.com/13435425/216780522-cc20ad37-4bf9-4a1d-af59-75c697029553.png)

![out-latent-conditional-3](https://user-images.githubusercontent.com/13435425/216780528-e155b541-4dab-453d-9958-72e6350ae659.png)

---

### Other Software Resources
There are many great projects focused on diffusion generative models. However, most of them involve somewhat complex frameworks that are not always suitable for learning and preliminary experimentation.

* 🤗 [diffusers](https://github.com/huggingface/diffusers)
* lucidrains [PyTorch DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch)
* OpenAI [guided-diffusion](https://github.com/openai/guided-diffusion)
* OpenAI [improved-diffusion](https://github.com/openai/improved-diffusion)
* CompVis [latent-diffusion](https://github.com/CompVis/latent-diffusion)
* Meta [DiT](https://github.com/facebookresearch/DiT)
* MONAI [GenerativeModels for Medical Imaging](https://github.com/Project-MONAI/GenerativeModels)

### Other Educational Resources
Some excellent materials have already been published on the topic! Huge respect to all of the creators :pray: - check them out if their work has helped you!

:coffee: **Blog Posts**
* [Score-based Perspective](https://yang-song.net/blog/2021/score/) by [Yang Song](https://yang-song.net)
* [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) by [Lilian Weng](https://twitter.com/lilianweng/)
* [Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) by [Niels Rogge](https://twitter.com/NielsRogge) and [Kashif Rasul](https://twitter.com/krasul)
* [Diffusion as a kind of VAE](https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html) by [Angus Turner](https://angusturner.github.io/about/)

:crystal_ball: **Explanation Videos**
* [Diffusion Model Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) by [Outlier](https://twitter.com/dome39931447)
* [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU) by [Ari Seff](https://www.ariseff.com)
* [Diffusion Models Beat GANs on Image Synthesis](https://www.youtube.com/watch?v=W-O7AZNzbzQ) the research paper explained by [Yannic Kilcher](https://twitter.com/ykilcher)

:wrench: **Implementation Videos**
* [Diffusion Models PyTorch Implementation](https://www.youtube.com/watch?v=TBCRlnwJtZU) by [Outlier](https://twitter.com/dome39931447)
* [High-Resolution Image Synthesis with LDMs | ML Coding Series](https://www.youtube.com/watch?v=f6PtJKdey8E) by [Aleksa Gordić](https://www.youtube.com/@TheAIEpiphany)

:mortar_board: **Video Lectures/Tutorials**
* [Diffusion Probabilistic Models](https://www.youtube.com/watch?v=XCUlnHP1TNM) - MIT 6.S192 lecture by [Jascha Sohl-Dickstein](http://www.sohldickstein.com)
* [Generative art using diffusion](https://www.youtube.com/watch?v=xYJEvihz3OI) - MIT 6.S192 lecture by [Prafulla Dhariwal](https://prafulladhariwal.com)
* [Learning to Generate Data by Estimating Gradients of the Data Distribution](https://www.youtube.com/watch?v=nv-WTeKRLl0) by [Yang Song](https://yang-song.net)
* [Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://www.youtube.com/watch?v=cS6JQpEY9cs) tutorial presented at CVPR2022 by [Karsten Kreis](https://twitter.com/karsten_kreis), [Ruiqi Gao](https://twitter.com/RuiqiGao) and [Arash Vahdat](https://twitter.com/ArashVahdat)
* [Generative Modeling by Estimating Gradients of the Data Distribution](https://www.youtube.com/watch?v=8TcNXi3A5DI) by [Stefano Ermon](https://mobile.twitter.com/stefanoermon)
* [Variational autoencoders and Diffusion Models](https://www.youtube.com/watch?v=pea3sH6orMc) by [Tim Salimans](https://mobile.twitter.com/TimSalimans)
