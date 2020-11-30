# ACSiam - PyTorch

Design of Asynchronous Correlation Discriminant Single Object Tracker Based on Siamese Network

## Dependencies

Install PyTorch, opencv-python and GOT-10k toolkit:

```bash
pip install torch
pip install opencv-python
pip install --upgrade git+https://github.com/got-10k/toolkit.git@master
```

[GOT-10k toolkit](https://github.com/got-10k/toolkit) is a visual tracking toolkit that implements evaluation metrics and tracking pipelines for 7 main datasets (GOT-10k, OTB, VOT, UAV123, NfS, etc.).

## Running the tracker

Run:

```
python run_tracking.py
```
