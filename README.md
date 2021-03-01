# Polar-Encodings
Learning Polar Encodings for Arbitrary-Oriented Ship Detection in SAR Images


# Introduction

Common horizontal bounding box (HBB)-based methods are not capable of accurately locating slender ship targets with arbitrary orientations in synthetic aperture radar (SAR) images. Therefore, in recent years, methods based on oriented bounding box (OBB) have gradually received attention from researchers. However, most of the recently proposed deep learning-based methods for OBB detection encounter the boundary discontinuity problem in angle or key point regression. In order to alleviate this problem, researchers propose to introduce some manually set parameters or extra network branches for distinguishing the boundary cases, which make training more difficult and lead to performance degradation. In this paper, in order to solve the boundary discontinuity problem in OBB regression, we propose to detect SAR ships by learning polar encodings. The encoding scheme uses a group of vectors pointing from the center of the ship target to the boundary points to represent an OBB. The boundary discontinuity problem is avoided by training and inference directly according to the polar encodings. In addition, we propose an Intersect over Union (IOU) -weighted regression loss, which further guides the training of polar encodings through the IOU metric and improves the detection performance. Experiments on the Rotating SAR Ship Detection Dataset (RSSDD) show that the proposed method can achieve better detection performance over other comparison algorithms and other OBB encoding schemes, demonstrating the effectiveness of our method.


# Dependencies

Ubuntu 18.04, Python 3.6.10, PyTorch 1.6.0, OpenCV-Python 4.3.0.36 

# How to start

## Train the model
```ruby
python main.py --data_dir dataPath --epochs 150 --batch_size 8 --dataset ssdd --phase train
```

## Test the model
```ruby
python main.py --data_dir dataPath --epochs 150 --batch_size 8 --dataset ssdd --phase test
```


## Evaluate the model
You may adjust the conf_thresh to get a better mAP
```ruby
python main.py --data_dir dataPath --epochs 80 --conf_thresh 0.1 --batch_size 8 --dataset ssdd --phase eval
```

# Reference
1. [CenterNet](https://github.com/xingyizhou/CenterNet)
2. [BBAVectors](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection)
3. [R-CenterNet](https://github.com/ZeroE04/R-CenterNet)
4. [MBB](https://bitbucket.org/william_rusnack/minimumboundingbox/src/master/)