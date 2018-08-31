## Self Driving Car
Building a Self Driving Car based upon Nvidia's paper: **End to End Learning for Self-Driving Cars**

## Research
[Nvidia Developer Blog](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

[End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

## Dependencies

Run this commands on your terminal first 

``` 
pip3 install --upgrade setuptools socketio eventlet Flask pillow numpy opencv-python
```
then, you will need at most python 3.6 not (3.7) to install tensorflow

```
pip3 install --upgrade tensorflow

pip3 install --upgrade tensorflow-gpu

```

**Note**: install Tensorflow with gpu if you have cuda supported and installed for [Windows](http://julip.co/2009/09/how-to-install-and-configure-cuda-on-windows/) or [Ubuntu 16.04](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04)

Last but not least ........ **Keras**

```
pip3 install --upgrade keras
```

## Simulator

Download For [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip) or [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip) or [Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip)

Extract the zip file anywhere you like and you can fire the simulator from **beta_simulator.exe**

Choose the track you like and start with the **Manual mode** so you can capture the data you need then after you train your Network you can start the autonomous mode with a terminal running the **drive.py** with your model.

## Testing

This easiet section by far, Type this in your terminal:

```
git clone https://github.com/MohamedAliRashad/self_driving_car.git\

cd self_driving_car

python3 drive.py model.h5
```
And run the simulator in the Autonomous mode and Enjoy!

**You can see a video of it working [Here](https://youtu.be/jOQ44nGenpw)**

## Training

I tweeked the suggested Neural Network form the paper after i read this [Article](https://towardsdatascience.com/deep-learning-on-car-simulator-ff5d105744aa) and the problem of over fitting.

I found that 10 epoches was the best number for smooth movement and high speed handling

**The Network Archticture**

![all text](https://cdn-images-1.medium.com/max/800/1*NbRV5KtuZwtPy50onUe_LQ.png)

