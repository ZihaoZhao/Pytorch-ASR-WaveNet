import random
import librosa
import numpy as np
import string


def collate_fn_(batch_data, max_len=40000):
    audio = batch_data[0]
    audio_len = audio.size(1)
    if audio_len > max_len:
        idx = random.randint(0,audio_len - max_len)
        return audio[:,idx:idx+max_len]
    else:
        return audio

class Data:
  num_channel = 40
  vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<EMP>']

  decoder_vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
  sample_rate = 16000

def read_wave(filepath):

    # load wave file
    wave, sr = librosa.load(filepath, mono=True, sr=None)

    # re-sample ( 48K -> 16K )
    wave = wave[::3]

    # get mfcc feature
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=Data.num_channel)

    #   wave, sr = librosa.load(filepath, mono=True, sr=Data.sample_rate)
    #   mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=Data.num_channel)
    return mfcc

def read_txt(filepath):
  txt = open(filepath).read()
  txt = ' '.join(txt.split())
  txt = txt.translate(string.punctuation).lower()
  reval = []
  for ch in txt:
    try:
      if ch in Data.vocabulary:
        reval.append(Data.vocabulary.index(ch))
      # else:
      #   glog.warning('%s was not in vocabulary at %s' % (ch, filepath))
    except KeyError:
      pass
  return reval

def cvt_np2string(inputs):
  outputs = []
  for input in inputs:
    output = ''
    for i in input:
    #   ch = i.decode('utf-8')
    #   if ch == '<EMP>':
    #     continue
      output += i#.decode('utf-8')
    outputs.append(output)
  return outputs

def _find_best_match2(inputs):
  def _find_node(values, start=(-1, -1)):
    node = []
    for index in range(start[1] + 1, len(values)):
      value = values[index]
      if len(value) > 0:
        for v in value:
          if v > start[0]:
            if len(node) == 0:
              node.append((v, index))
            elif v < node[-1][0]:
              node.append((v, index))
    return node

  def _find_nodes(values):
    nodes = []
    while True:
      if len(nodes) == 0:
        node = _find_node(values)
        if len(node) == 0:
          break
        for n in node:
          nodes.append([n])
      else:
        tmps = []
        change = False
        for tmp in nodes:
          node = _find_node(values, tmp[-1])
          if len(node) == 0:
            tmps.append(tmp)
          else:
            for n in node:
              tmps.append(tmp + [n])
            change = True

        if change:
          nodes = tmps
        else:
          break
    return nodes

  nodes = _find_nodes(inputs)
  if len(nodes) == 0:
    return []
  else:
    return sorted(nodes, key=lambda iter: len(iter), reverse=True)[0]

def _normalize(inputs):
  inputs = inputs.split(' ')
  outputs = []
  for input in inputs:
    if input != '':
      outputs.append(input)
  return outputs


def evalute(predicts, labels):
  predicts = _normalize(predicts)
  labels = _normalize(labels)
  matches = []
  for label in labels:
    match = []
    for j, predict in enumerate(predicts):
      if label == predict:
        match.append(j)
    if len(match) > 0:
      matches.append(match)
  match = _find_best_match2(matches)
  return len(match), len(predicts), len(labels)


def evalutes(predicts, labels):
  size = min(len(predicts), len(labels))
  tp = 0
  pred = 0
  pos = 0
  for i in range(size):
    data = evalute(predicts[i], labels[i])
    tp += data[0]
    pred += data[1]
    pos += data[2]
  return tp, pred, pos

if __name__ == '__main__':
    mfcc = read_wave('/lyuan/dataset/VCTK/VCTK-Corpus/wav48/p225/p225_001.wav')
    print(mfcc.shape)
    reval = read_txt('/lyuan/dataset/VCTK/VCTK-Corpus/txt/p225/p225_001.txt')
    print(reval.shape)

