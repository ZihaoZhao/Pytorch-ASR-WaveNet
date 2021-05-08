# Intro

A Pytorch implementation of WaveNet ASR (Automatic Speech Recognition)

Reference:

https://github.com/kingstarcraft/speech-to-text-wavenet2

https://github.com/buriburisuri/speech-to-text-wavenet
# Install

```
conda install pytorch=1.4.0
```

# Run

```
python train.py --exp test 
```

# Note

1. ctcloss in pytorch and tensorflow have the different default blanks. " " is not the blank, however, "<EMP>" is the blank.

2. Xaiver init is better.