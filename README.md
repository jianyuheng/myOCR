:blush::blush::blush::blush::blush::blush:
# Scene Text Localization & Recognition Resources
A curated list of resources dedicated to scene text localization and recognition. Any suggestions and pull requests are welcome.

## Foundation
- __[2015-arxiv] CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition__[`paper`](https://arxiv.org/abs/1507.05717)[`translation`](http://noahsnail.com/2017/08/21/2017-8-21-CRNN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

- __[2016-ECCV] CTPN: Detecting Text in Natural Image with Connectionist Text Proposal Network__[`paper`](https://arxiv.org/abs/1609.03605)
[`translation`](http://noahsnail.com/2018/02/02/2018-02-02-Detecting%20Text%20in%20Natural%20Image%20with%20Connectionist%20Text%20Proposal%20Network%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)
- __[2017-AAAI] TextBoxes: A Fast Text Detector with a Single Deep Neural Network__[`paper`](https://arxiv.org/abs/1611.06779) [`code`](https://github.com/MhLiao/TextBoxes)

## Papers & Code
- __[2018-AAAI] SEE: Towards Semi-Supervised End-to-End Scene Text Recognition__ [`paper`](http://arxiv.org/abs/1712.05404)
[`code`](https://github.com/Bartzi/see)
- __[2018-arxiv] TextBoxes++: A Single-Shot Oriented Scene Text Detector__[`paper`](https://arxiv.org/pdf/1801.02765.pdf)[`code`](https://github.com/MhLiao/TextBoxes_plusplus)

- [2018-CVPR] Rotation-Sensitive Regression for Oriented Scene Text Detection[`paper`](https://arxiv.org/pdf/1803.05265.pdf)
- [2018-CVPR] Single Shot Text Spotter with Explicit Alignment and Attention[`paper`](https://arxiv.org/pdf/1803.03474.pdf)
- [2018-CVPR] Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation [`paper`](https://arxiv.org/abs/1802.08948)
- [2018-arxiv] FOTS: Fast OrientedText Spotting with a Unified Network[`paper`](https://arxiv.org/pdf/1801.01671.pdf)
- [2018-AAAI] PixelLink: Detecting Scene Text via Instance Segmentation[`paper`](https://arxiv.org/pdf/1801.01671.pdf)

- [2017-ICCV] Deep TextSpotter: An End-to-End Trainable Scene Text Localization and
Recognition Framework [`paper`](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busta_Deep_TextSpotter_An_ICCV_2017_paper.pdf)
[`code`](https://github.com/MichalBusta/DeepTextSpotter)
- [2017-CVPR] EAST: An Efficient and Accurate Scene Text Detector [`paper`](https://arxiv.org/abs/1704.03155) [`code`](https://github.com/argman/EAST)
- [2017-CVPR] Detecting oriented text in natural images by linking segments [`paper`](http://mc.eistar.net/UpLoadFiles/Papers/SegLink_CVPR17.pdf) [`code`](https://github.com/bgshih/seglink)

## Repository
- [运用tensorflow实现自然场景文字检测,keras/pytorch实现crnn+ctc实现不定长中文OCR识别](https://github.com/chineseocr/chinese-ocr)
- [Training SEE: Towards Semi-Supervised End-to-End Scene Text Recognition](https://github.com/Bartzi/see)
- [Train TextBoxes++: A Single-Shot Oriented Scene Text Detector](https://github.com/MhLiao/TextBoxes_plusplus) 
- [Train CTPN](https://github.com/eragonruan/text-detection-ctpn)

## Datasets
- [`COCO-Text (Computer Vision Group, Cornell)`](http://vision.cornell.edu/se3/coco-text/)   `2016`
  - 63,686 images, 173,589 text instances, 3 fine-grained text attributes.
  - Task: text location and recognition
  - [`COCO-Text API`](https://github.com/andreasveit/coco-text)

- [`Synthetic Word Dataset (Oxford, VGG)`](http://www.robots.ox.ac.uk/~vgg/data/text/)   `2014`
  - 9 million images covering 90k English words
  - Task: text recognition, segmantation
  - [`download`](http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz)

- [`IIIT 5K-Words`](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)   `2012`
  - 5000 images from Scene Texts and born-digital (2k training and 3k testing images)
  - Each image is a cropped word image of scene text with case-insensitive labels
  - Task: text recognition
  - [`download`](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz)

- [`StanfordSynth(Stanford, AI Group)`](http://cs.stanford.edu/people/twangcat/#research)   `2012`
  - Small single-character images of 62 characters (0-9, a-z, A-Z)
  - Task: text recognition
  - [`download`](http://cs.stanford.edu/people/twangcat/ICPR2012_code/syntheticData.tar)

- [`MSRA Text Detection 500 Database (MSRA-TD500)`](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))   `2012`
  - 500 natural images(resolutions of the images vary from 1296x864 to 1920x1280)
  - Chinese, English or mixture of both
  - Task: text detection

- [`Street View Text (SVT)`](http://tc11.cvc.uab.es/datasets/SVT_1)   `2010`
  - 350 high resolution images (average size 1260 × 860) (100 images for training and 250 images for testing)
  - Only word level bounding boxes are provided with case-insensitive labels
  - Task: text location

- [`KAIST Scene_Text Database`](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database)   `2010`
  - 3000 images of indoor and outdoor scenes containing text
  - Korean, English (Number), and Mixed (Korean + English + Number)
  - Task: text location, segmantation and recognition

- [`Chars74k`](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)   `2009`
  - Over 74K images from natural images, as well as a set of synthetically generated characters 
  - Small single-character images of 62 characters (0-9, a-z, A-Z)
  - Task: text recognition
  
- `ICDAR Benchmark Datasets`

|Dataset| Discription | Competition Paper |
|---|---|----
|[ICDAR 2015](http://rrc.cvc.uab.es/)| 1000 training images and 500 testing images|`paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://rrc.cvc.uab.es/files/Robust-Reading-Competition-Karatzas.pdf)|
|[ICDAR 2013](http://dagdata.cvc.uab.es/icdar2013competition/)| 229 training images and 233 testing images |`paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://dagdata.cvc.uab.es/icdar2013competition/files/icdar2013_competition_report.pdf)|
|[ICDAR 2011](http://robustreading.opendfki.de/trac/)| 229 training images and 255 testing images |`paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://www.iapr-tc11.org/archive/icdar2011/fileup/PDF/4520b491.pdf)|
|[ICDAR 2005](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2005_Robust_Reading_Competitions)| 1001 training images and 489 testing images |`paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://www.academia.edu/download/30700479/10.1.1.96.4332.pdf)|
|[ICDAR 2003](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)| 181 training images and 251 testing images(word level and character level) |`paper`  [![link](https://www.lds.org/bc/content/shared/content/images/gospel-library/manual/10735/paper-icon_1150845_tmb.jpg)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.332.3461&rep=rep1&type=pdf)|


# __Prior Knowledge__
## Image Classification

* AlexNet  
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
[中文版](http://noahsnail.com/2017/07/18/2017-7-18-AlexNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/07/04/2017-7-4-AlexNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91/)

* VGG  
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
[中文版](http://noahsnail.com/2017/08/17/2017-8-17-VGG%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/08/17/2017-8-17-VGG%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* ResNet  
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
[中文版](http://noahsnail.com/2017/07/31/2017-7-31-ResNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/07/31/2017-7-31-ResNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* GoogLeNet  
[Going Deeper With Convolutions](https://arxiv.org/abs/1409.4842)
[中文版](http://noahsnail.com/2017/07/21/2017-7-21-GoogleNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/07/21/2017-7-21-GoogleNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* BN-GoogLeNet  
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
[中文版](http://noahsnail.com/2017/09/04/2017-9-4-Batch%20Normalization%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/09/04/2017-9-4-Batch%20Normalization%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* Inception-v3  
[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
[中文版](http://noahsnail.com/2017/10/09/2017-10-9-Inception-V3%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/10/09/2017-10-9-Inception-V3%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* SENet  
[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
[中文版](http://noahsnail.com/2017/11/20/2017-11-20-Squeeze-and-Excitation%20Networks%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/11/20/2017-11-20-Squeeze-and-Excitation%20Networks%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

## Object Detection

* YOLO   
[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
[中文版](http://noahsnail.com/2017/08/02/2017-8-2-YOLO%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/08/02/2017-8-2-YOLO%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* SSD  
[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
[中文版](http://noahsnail.com/2017/12/11/2017-12-11-Single%20Shot%20MultiBox%20Detector%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/12/11/2017-12-11-Single%20Shot%20MultiBox%20Detector%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* YOLO9000  
[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
[中文版](http://noahsnail.com/2017/12/26/2017-12-26-YOLO9000,%20Better,%20Faster,%20Stronger%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/12/26/2017-12-26-YOLO9000,%20Better,%20Faster,%20Stronger%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* Deformable-ConvNets  
[Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)
[中文版](http://noahsnail.com/2017/11/29/2017-11-29-Deformable%20Convolutional%20Networks%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2017/11/29/2017-11-29-Deformable%20Convolutional%20Networks%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* Faster R-CNN  
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
[中文版](http://noahsnail.com/2018/01/03/2018-01-03-Faster%20R-CNN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2018/01/03/2018-01-03-Faster%20R-CNN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

* R-FCN  
[R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409)
[中文版](http://noahsnail.com/2018/01/22/2018-01-22-R-FCN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
[中英文对照](http://noahsnail.com/2018/01/22/2018-01-22-R-FCN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)
