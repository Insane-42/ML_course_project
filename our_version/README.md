# Unsupervised learning of digit recognition using spike-timing-dependent plasticity
In this directory, we reproduce the paper Unsupervised learning of digit recognition using spike-timing-dependent plasticity using SpikingJelly and Brian2.

## Environment Setup
```shell
conda create -n stdp python=3.7
conda activate stdp
pip install -r requirements.txt
```

## SpikingJelly
To train STDP using SpikingJelly, run:
```shell
sh scripts/train_spikingjelly.sh
```
To test STDP using Brian2, run:
```shell
sh scripts/test_spikingjelly.sh
```

## Brian2
To train STDP using Brian2, run the following scripts until the simulation finished:
```shell
sh scripts/train_brian2.sh
```
To test STDP using Brian2, run the following scripts and you will get the accuracy:
```shell
sh scripts/test_brian2.sh
```



