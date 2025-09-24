# Easy Turn: Integrating Acoustic and Linguistic Modalities for Robust Turn-Taking in Full-Duplex Spoken Dialogue Systems

<p align="center">
  Guojian Li<sup>1</sup>, Chengyou Wang<sup>1</sup>, Hongfei Xue<sup>1</sup>, 
  Shuiyuan Wang<sup>1</sup>, Dehui Gao<sup>1</sup>, Zihan Zhang<sup>2</sup>, 
  Yuke Lin<sup>2</sup>, Wenjie Li<sup>2</sup>, Longshuai Xiao<sup>2</sup>, 
  Zhonghua Fu<sup>1</sup><sup>,╀</sup>, Lei Xie<sup>1</sup><sup>,╀</sup>
</p>

<p align="center">
  <sup>1</sup> Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University <br>
  <sup>2</sup> Huawei Technologies, China <br>
</p>

<p align="center">
🌐 <a href="https://huggingface.co/collections/ASLP-lab/easy-turn-68d3ed0b294df61214428ea7"> Huggingface</a> &nbsp&nbsp  |   &nbsp&nbsp 🤖 <a href="https://huggingface.co/ASLP-lab/Easy-Turn"> Easy Turn Model</a> &nbsp&nbsp 
<br>
📑 <a href="https://arxiv.org">Paper</a> &nbsp&nbsp  |  &nbsp&nbsp 🎤 <a href="https://aslp-lab.github.io">Demo Page</a> &nbsp&nbsp
</p>

## Download
* The Easy Turn models are available at [Easy Turn](https://huggingface.co/ASLP-lab/Easy-Turn).
* The Easy Turn trainset is available at [Easy Turn Trainset](https://huggingface.co/datasets/ASLP-lab/Easy-Turn-Trainset).
* The Easy Turn testset is available at [Easy Turn Testset](https://huggingface.co/datasets/ASLP-lab/Easy-Turn-Testset).

## Easy Turn
Full-duplex interaction is crucial for natural human–machine communication, yet remains challenging as it requires robust turn-taking detection to decide when the system should speak, listen, or remain silent. Existing solutions either rely on dedicated turn-taking models, most of which are not open-sourced. The few available ones are limited by their large parameter size or by supporting only a single modality, such as acoustic or linguistic. Alternatively, some approaches finetune LLM backbones to enable full-duplex capability, but this requires large amounts of full-duplex data, which remain scarce in open-source form. To address these issues, we propose Easy Turn—an open-source, modular turn-taking detection model that integrates acoustic and linguistic bimodal information to predict four dialogue turn states: complete, incomplete, backchannel, and wait, accompanied by the release of Easy Turn trainset, a 1,145-hour speech dataset designed for training turn-taking detection models. Compared to existing open-source models like TEN Turn Detection and Smart Turn V2, our model achieves state-of-the-art turn-taking detection accuracy on our open-source Easy Turn testset.
<div align="center"><img width="600px" src="src/figs/Chuan-Pipeline.png" /></div>

## Easy Turn Trainset
The Easy Turn Trainset is a large-scale audio dataset for turn-taking detection, comprising both real and synthetic data. It contains four subsets corresponding to different conversational turn-taking states: 580 hours of complete turns, 532 hours of incomplete turns, 10 hours of backchannel turns, and 23 hours of wait turns, totaling approximately 1,100 hours. All audio is recorded at 16 kHz sampling rate, 16-bit resolution, and single-channel format. Each recording is accompanied by a text transcription and labeled with one of the four turn-taking states. 
<div align="center"><img width="600px" src="src/figs/Chuan-Pipeline.png" /></div>

## EXPERIMENTS
### Main Results

### Ablation Study

## Citation
Please cite our paper if you find this work useful:


