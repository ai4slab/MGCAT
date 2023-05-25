# MGCAT

This code implements the model proposed in the
paper ["Multi-view graph neural network with cascaded attention for lncRNA-miRNA interaction prediction
"](https://doi.org/10.1016/j.knosys.2023.110492) (accepted by _Knowledge-Based Systems_).

## Requirements

Please make sure that all the following packages are successfully installed:

* pytorch==1.10.2
* torch_geometric==2.0.3
* pandas==1.3.5
* numpy== 1.22.2
* scikit-learn==1.0.2

To get the environment resolved quickly, run:

    cd code/
    pip install -r requirements.txt

## Usage

Please download code and data, then execute the following command:

    cd code/
    python main.py

## Citation

If it helps you, please kindly cite this paper:

```
@article{li2023multi,
  title={Multi-view graph neural network with cascaded attention for lncRNA-miRNA interaction prediction},
  author={Li, Hui and Wu, Bin and Sun, Miaomiao and Ye, Yangdong and Zhu, Zhenfeng and Chen, Kuisheng},
  journal={Knowledge-Based Systems},
  volume={268},
  pages={110492},
  year={2023},
  publisher={Elsevier}
}
```