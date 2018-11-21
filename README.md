# ShoesKerasGAN
Implementation of a DCGAN in Keras generating Images of Adidas Shoes and testing through ImageNet 1000 and VGG 16. 
This was the core part of my undergraduate thesis that I wrote after my internship at IMPLab (pattern recognition and AI lab located at University of Parma).

The DCGAN I implemented is mostly based of the one used in Radford paper with some substantial changings that helped to increase the accuracy of the network by a good margin. I implemented it in Keras because the code is far more comprehensible than other libraries used for Deep Learning.
This Network was trained using a training set provided by Adidas which I can't share online due to copyright issues. So you need to train the newtork on your own dataset. If you want, I can email you the training model I got through training the net on a 1080TI for 200k iterations, just ask at carretta.mtt@gmail.com, so you can run the "ImageNet" tests on your own machine. 

In order to run this code on your system, you have to install different libraries:
- Python 3 or greater
- Tensorflow
- Keras
- OpenCV
- Numpy
- Matplotlib
- tqdm
- glob

The file containing the system is ShoesGAN.py, the ones relative to testing are ImageNet.py and ImageNetAllArch1.py. The first one loads the model of a single epoch in the generator and generates 1k images that will be evaluated by VGG16, while the second one does this process every 10k epochs, loading every model of the net and printing than the accuracy related to each epoch through training, to find the best epoch.
The best epoch found is the 30k, with an accuracy score of 37.1% through ImageNet test, but a great 84% by drag and drop the images in Google Images and see if it was recognized by shoes! 
The results can be improved by using a larger scale of the images I suppose, but the 1080TI and 16 GBs of RAM I had to train it on were not sufficient to having images greater than 256*128.

## Results
Here are some of the images obtained:


![21](https://user-images.githubusercontent.com/20916106/48858625-2b3f3f80-edbc-11e8-9d45-06d2c701f066.png)
![28](https://user-images.githubusercontent.com/20916106/48858659-398d5b80-edbc-11e8-937e-857eb5408c7d.png)
![38](https://user-images.githubusercontent.com/20916106/48858663-3d20e280-edbc-11e8-82eb-1961b13f8181.png)
![50](https://user-images.githubusercontent.com/20916106/48858666-3f833c80-edbc-11e8-9ab0-549531341710.png)

And here are the results obtained through VGG16 and an example on how the shoes are recognized as Adidas shoes by Google Images:
![plottest2](https://user-images.githubusercontent.com/20916106/48858671-43af5a00-edbc-11e8-80c5-eed53b7fed05.png)
![testadidas](https://user-images.githubusercontent.com/20916106/48858672-43af5a00-edbc-11e8-85d5-054e9e7b4a27.png)

