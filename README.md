# Temporal-Causal-Lane-Fast-Detection
即时的车道线检测是计算机视觉课程中常见的实验任务。 在基于高斯滤波 + canny边缘检测 + hough变换的车道线检测实验中，我意识到精细地逐帧检测车道直线容易造成运行耗时过长。考虑到前一时刻和后一时刻车辆摄像头拍摄内容具有显著“关联”。我尝试从理论角度分析其因果性质，并给出一种更快速的车道线检测方法。 我所使用的实验视频和代码文件将陆续加入仓库之中。

您可以阅读report-experiment.doc来详细了解我使用的方法，在第二部分我给出了新方法的原理推导和设计思路。您可以在labCausalFast.py中找到实现新型车道线检测方法的代码。并在outputFast.mp4中找到实验结果。本项目还在lab.py，标准结果.avi.zip，input_video.mp4.zip中分别给出原始方法代码，标准检测结果与输入视频。

Real-time lane line detection is a common experimental task in computer vision courses. In the lane line detection experiment based on Gaussian filtering + Canny edge detection + Hough transform, I realized that precisely detecting lane lines frame by frame can easily lead to excessive running time. Considering that the content captured by the vehicle's camera at the previous moment and the next moment has significant "correlation", I attempted to analyze its causal nature from a theoretical perspective and propose a faster lane line detection method. The experimental video and code files I used will be gradually added to the repository. 

You can read report-experiment.doc to learn in detail about the methods I used. In the second part, I provided the principle derivation and design ideas of the new method. You can find the code for implementing the new lane line detection method in labCausalFast.py. The experimental results can be found in outputFast.mp4. This project also provides the original method code, standard detection results, and input video in lab.py, standard_result.avi.zip, and input_video.mp4.zip respectively.
