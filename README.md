# snap-it-find-it
Snap-it Find-it Telegram Bot link
https://t.me/SnapitFindit_bot

**Background and Problem Statement:**<br />
Searching for furniture is a surprisingly time-consuming and mundane process. There are multiple different furniture retailers such as Ikea and HipVan which have a high volume of furniture in their catalogs. Typically, people would spend hours browsing through the catalogs trying to find something they would like, or spend hours visiting the physical stores in an attempt to find something that fits into the theme of their new apartment. Often, people may draw inspiration when they unexpectedly come across a nice piece of furniture, and decide they want to own a similar piece. The problem statement surfaces: Where can I find something similar to this piece of furniture? Does Ikea or HipVan have something similar?<br />
<br />
This presents the problem statement the team would be trying to tackle. Given an input RGB image of certain household objects or furniture, develop a computer vision system to return images of similar objects or furniture from Ikea and HipVan catalog. The computer vision system would include an object detector module and an image similarity module, neatly packaged into a telegram bot as the user interface. This system can help users narrow down quickly which major furniture vendor the user should visit, and help make the entire furniture hunting process more efficient.<br />
<br />

# Medium post
https://jensen-wong.medium.com/snap-it-find-it-your-shopping-companion-bot-8101494545a8<br />
<br />

# Here’s a simple guideline if you wish to run this on Linux VM:
$ sudo apt update<br />
$ sudo apt install build-essential<br />
$ sudo apt-get install bzip2 libxml2-dev<br />
-- to install miniconda<br />
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh<br />
$ bash Miniconda3-latest-Linux-x86_64.sh<br />
$ rm Miniconda3-latest-Linux-x86_64.sh<br />
$ source .bashrc<br />
-- install dependency<br />
$ conda install pytorch torchvision torchaudio -c pytorch<br />
$ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'<br />
$ pip install keras tensorflow pandas sklearn opencv-python python-telegram-bot<br />
<br />

# Below are the explanation for each of the folder and file:
- CNN_F1_Score.ipynb (to calculate the F1 score for each model)
- CNN_Keras_Feature_Extractor.ipynb (a notebook that extract feature from multiple CNN model and benchmarking of precision / accuracy)
- data (store all the feature extraction file and model weight)
- hipvan_image (images scraped from Hipvan)
- ikea_image (images scraped from Ikea)
- image_search_vgg.ipynb / .py (image extraction and similarity search using VGG16)
- image_search_vgg.ipynb / .py (image extraction and similarity search using EfficientNetB7)
- main.ipynb / .py (main script that also serve as telegram script that integrate with the rest of the .py file)
- mask_rcnn_coco.hy (weight for mask rcnn for coco dataset)
- object_detection_background_removal.ipynb / .py (object detection and background removal using Detectron2)
- object_detection.ipynb / .py (object detection and background removal using YOLOv3)
- telegram-testing.ipynb (telegram testing bot - for testing purpose on secondary bot)
- web_scrapper.ipynb (web scraping and download image script for Ikea and Hipvan)


