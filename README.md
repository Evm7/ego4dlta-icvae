#  Intention-Conditioned Long-Term Human Egocentric Action Forecasting @ EGO4D Challenge 2022
[Arxiv2022]  [Intention-Conditioned Long-Term Human Egocentric Action Forecasting : @EGO4D LTA Challenge 2022](https://arxiv.org/abs/2207.12080)

This work ranked first in the EGO4D LTA Challenge!


<img src="/figures/overall_architecture.png" alt="ICVAE" style="zoom:45%;" />

## 📢 News
- [14.06.2022] Our ICVAE won [**1st place** in LTA](https://eval.ai/web/challenges/challenge-page/1598/overview)
- [25.07.2022] We release the first version of the ICVAE codebase.
- [26.07.2022] We release the arXiv paper.

## 📝 Preparation
### Install dependencies 
```bash
conda create -n icvae python=3.8
source activate icvae
cd [Path_To_This_Code]
pip install -r requirements.txt
mkdir outputs
```

### Ego4D videos and metadata
- Follow the procedure indicated in [EGO4D main page](https://ego4d-data.org/docs/data/features) on how to download the pre-extracted visual features from Slowfast 8x8 R101 pretrained in Kinetics400.
- Download all the annotations for FHO_LTA challenge (train,test, and validation) and add them into the annotations directory (together with json files preprocessed by us)/
- Modify config.yaml file and paths.py from data directory by adding the path information to this code ([PATH_TO_CODE_BASE]) and to the feature dataset downloaded from Ego4d and padded ([PATH_TO_DATASET])


## 🏋️‍️ Pretraining
This section is still under development. In the next days we will publish more information on how to test/train the project.

## 🎓 Citation

If you find our work helps, please cite our paper.

@misc{https://doi.org/10.48550/arxiv.2207.12080,
  doi = {10.48550/ARXIV.2207.12080},
  url = {https://arxiv.org/abs/2207.12080},
  author = {Mascaro, Esteve Valls and Ahn, Hyemin and Lee, Dongheui},
  title = {Intention-Conditioned Long-Term Human Egocentric Action Forecasting @ EGO4D Challenge 2022},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

## ✉️ Contact

This repo is maintained by [Esteve](https://github.com/Evm7).

## LICENSE

MIT
