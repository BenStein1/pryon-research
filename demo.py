#!/usr/bin/env python3
"""Firmware-free engineering demo. Synthetic features, not speech accuracy or a vendor encoder run."""
import json
import numpy as np
from pryon_runtime_analysis import lfbe
from pryon_decoder import decode
from pryon_qat import train_qat
from train_pryon_streaming import temporal_windows


def demo():
    pcm=((np.arange(720,dtype=np.int64)*73)%20001-10000).astype(np.int16)
    features=lfbe(pcm)
    rng=np.random.default_rng(9)
    activations=[rng.uniform(.1,1,(33,25,100)).astype(np.float32) for _ in range(2)]
    activations[0][...,0]=2;activations[1][...,0]=0
    fixed=(np.zeros((2,100),np.int8),np.ones(2,np.float32),np.zeros(2,np.int32))
    _,_,r=train_qat(activations,[1,0],fixed,np.zeros(2,np.float32),np.zeros(100,np.float32),-2.,steps=60,learning_rate=.03)
    p=np.zeros((35,25,3),np.float32);p[...,0]=1
    for j in range(3):
        lane=(30-15-j+1)%25;p[30,lane]=[.01,.99,0]
    events=list(decode(p))
    assert features.shape==(3,64) and r['exported_loss']<r['initial_loss']-.05
    assert any(e[0]==2 for e in events)
    return dict(synthetic_only=True,vendor_encoder_executed=False,frontend_shape=list(features.shape),
                classifier_sequence_shape=list(activations[0].shape),temporal_window_shape=list(temporal_windows(activations[0]).shape),
                initial_quantized_training_loss=r['initial_loss'],exported_quantized_training_loss=r['exported_loss'],
                synthetic_decoder_accepts=sum(e[0]==2 for e in events))

if __name__=='__main__':print(json.dumps(demo(),indent=2))
