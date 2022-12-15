# SinDDM

[Project](https://matankleiner.github.io/sinddm/) | [Arxiv](https://arxiv.org/pdf/2211.16582.pdf) | [Supplementary materials](https://matankleiner.github.io/sinddm/resources/sinddm_supp.pdf)
### Official pytorch implementation of the paper: "SinDDM: A Single Image Denoising Diffusion Model"


## Random samples from a *single* image
With SinDDM, you can train a generative model from a single natural image, and then generate random samples from the given image, for example:

![](imgs/teaser.PNG)


## SinDDM's applications
SinDDM can be also used for a line of image manipulation tasks, for example:
 ![](imgs/manipulation.PNG)


See section 3 in our [paper](https://arxiv.org/pdf/2211.16582.pdf) for more details.


### Citation
If you use this code for your research, please cite our paper:

```
@article{kulikov2022sinddm,
  title      = {SinDDM: A Single Image Denoising Diffusion Model},
  author     = {Kulikov, Vladimir and Yadin, Shahar and Kleiner, Matan and Michaeli, Tomer},
  journal    = {arXiv preprint arXiv:2211.16582},
  year       = {2022}
}
```

## Code
Note: This is an early code release which provides full functionality, but is not yet fully organized or optimized. We will be extensively updating this repository in the coming weeks. 
### Install dependencies

```
--coming soon--
python -m pip install -r requirements.txt
```

This code was tested with python 3.8, torch 1.13

###  Train
To train a SinDDM model on your own image e.g. "pyramids.jpg", put the desired training image under ./datasets/pyramids/, and run

```
python main.py --scope pyramids --mode train --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ 
```

This will also use the resulting trained model to generate random samples starting from the coarsest scale (s=0).


#### Pretrained models
We provide several pre-trained models for you to use under /results/ folder. More models will be available soon.

###  Random sampling
To generate random samples, please first train a SinDDM model on the desired image (as described above) or use a provided pretrained model, then run 

```
python main.py --scope pyramids --mode sample --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ --load_milestone 12
```

###  Random samples of arbitrary sizes 
To generate random samples of arbitrary sizes, use the '--scale_mul h w' argument.
For example, to generate an image with the width dimension 2 times larger run: 
```
python main.py --scope pyramids --mode sample --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ --load_milestone 12 --scale_mul 1 2
```

###  Text guided content generation

To guide the generation to create new content using a given text, run: 

```
python main.py --scope pyramids --mode clip_content --clip_text "Volcano Eruption" --strength 0.6 --fill_factor 0.3 --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ --load_milestone 12 --sample_batch_size 4
```
Where *strength* and *fill_factor* are controllable parameters explained in the paper,
This will automatically start a new training phase with noise padding mode.


###  Text guided style generation

To guide the generation to create a new style for the image using a given text, run: 

```
python main.py --scope pyramids --mode clip_style_gen --clip_text "Van Gogh Starry Night" --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ --load_milestone 12 --scale_mul 1 2 --sample_batch_size 4
```
Note: We can add the '--scale_mul y x' argument to generate an arbitrary size sample with the given style.

###  Text guided style transfer

To create a new style for a given image, without changing the original image global structure, run:

```
python main.py --scope pyramids --mode clip_style_trans --clip_text "Monet Haystack" --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ --load_milestone 12 --sample_batch_size 4
```

###  Text guided ROI

To modify an image in a specified ROI(Region Of Interest) with a given text, run:

```
python main.py --scope pyramids --mode clip_roi --clip_text "Cracks" --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ --load_milestone 12 --sample_batch_size 4
```
Note: A Graphical prompt will open and ask to select a ROI within the image.

###  Harmonization

To harmonize a pasted object into an image, place a naively pasted reference image and the selected mask into ./datasets/<trained_image_folder>/i2i/ and then run:


```
python main.py --scope pyramids --mode harmonization --harm_mask <mask_name> --input_image <naively-pasted-image> --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ --load_milestone 12 --sample_batch_size 4

```

###  Style Transfer

To use the content of an input image with the style of the training image, place the input image into ./datasets/<trained_image_folder>/i2i/ and then run:

```
python main.py --scope pyramids --mode style_transfer --input_image <input_image_name> --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ --load_milestone 12 --sample_batch_size 4

```

###  ROI guided generation

Here, the user can mark a specific training image ROI and choose where it should appear in the generated samples.
```
python main.py --scope pyramids --mode roi --dataset_folder ./datasets/pyramids/ --image_name pyramids.jpg --results_folder ./results/ --load_milestone 12 --scale_mul 1 2 --sample_batch_size 4

```
A graphical prompt will open and allow the user to choose a ROI from the training image and where it should appear in the resulting samples.
Here as well, we can generate an image with arbitrary shapes using --scale_mul y x
