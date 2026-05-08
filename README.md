# NeurIPS2025-LEAR
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![Paper](https://img.shields.io/badge/Paper-OpenReview-red)](https://openreview.net/pdf?id=uXKgVqYTJ2) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-purple.svg) 

The official implementation for **"[Learning Expandable and Adaptable Representations for Continual Learning](https://openreview.net/pdf?id=uXKgVqYTJ2)"** (NeurIPS2025) 

------

## ▶️ Usage

### **1. Create env and install requirements**

```bash
conda create -n LEAR python=3.10
conda activate LEAR
pip install -r requirements.txt
```

### **2. Run the example training script**

```bash
bash LEAR.sh
```

### Project structure overview

```bash
LEAR/
├── backbone/                 # Pre-trained backbone models
│   ├── LEAR.py               # LEAR backbone implementation
│   └── ...
├── datasets/                 # Dataset loaders
|   ├── init.py               # Modify domain sequence                
│   └── ...
├── models/                   # CL Method implementations
│   └── LEAR.py               # LEAR method implementation
├── utils/                    # Helper tools
|   ├── train_domain.py       # Training scripts                
│   └── ...
├── main_domain.py            # Main entry
├── LEAR.sh
└── README.md
```

------

## 📝 Citation

If you find this repository helpful, please click the ⭐Star and cite our paper:

```
@inproceedings{yulearning,
  title={Learning Expandable and Adaptable Representations for Continual Learning},
  author={Yu, Ruilong and Liu, Mingyan and Ye, Fei and Bors, Adrian G and Hu, Rongyao and others},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

------

## 🙏 Acknowledgement

Thanks for the awesome continual learning framework **[Mammoth](https://github.com/aimagelab/mammoth)**.