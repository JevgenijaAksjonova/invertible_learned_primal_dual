Invertible Learned Primal-Dual
==================================

This repository contains the code for the article "[Invertible Learned Primal-Dual](link to the paper)".

Contents
--------
The code contains the following

* Training and evaluation using anthropomorphic data from Mayo Clinic (2D).
* Training and evaluation using downsampled Walnut dataset (3D).
* Implementation of the baseline methods.

Dependencies
------------
The code depends on pytorch, [ODL](https://github.com/odlgroup/odl), [ASTRA toolbox](https://www.astra-toolbox.com/) and [MemCNN](https://github.com/silvandeleemput/memcnn). 
ODL, ASTRA and MemCNN can be installed by 

```bash
$ git clone https://github.com/odlgroup/odl
$ cd odl
$ pip install --editable .
$ conda install -c astra-toolbox/label/dev astra-toolbox
$ conda install -c silvandeleemput -c pytorch -c simpleitk -c conda-forge memcnn
```

Contact
-------
Jevgenija Rudzusika,  
KTH Royal Institute of Technology,   
jevaks@kth.se

Buda Bajic,  
KTH Royal Institute of Technology,  
budabp@kth.se

