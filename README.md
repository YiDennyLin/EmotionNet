# EmotionNet
<img title="EmotionNet Structure" alt="This is the EmotionNet Model Structure" src="./emotionNet.png">

## emotionNet_Notebook.ipynb
This is the implementation notebook of the whole experiment.
## EmotionNet.py
This is the EmotionNet Structure file
## main.py
This is the main file about our experiments(including training, testing and model evaluation)

# How to Use (Two Options)
## First Step
+ Download SFEW dataset.
+ Install all the packages, like:\
`pip install onnx`\
`pip install grad-cam`

## Choose one option
1. Run the notebook file(emotionNet_Notebook.ipynb), remember changing saving path and dataset path to your own version.
2. Run main.py, remember changing saving path and dataset path to your own version.


# Appendix
We use the open-source Grad CAM method and display visualization effects in our model. \
See the GitHub documentation on [Grad_CAM](https://github.com/jacobgil/pytorch-grad-cam).
