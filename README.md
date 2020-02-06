# Structure-emphasized Multimodal Style Transfer
![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.3602064.svg)


Pytorch(1.0+) implementation of My master paper ["Structure-emphasized Multimodal Style Transfer"](https://drive.google.com/open?id=1Y77Zy25gtuapEQCEfysoRmhiuEnc48Op).

We proposed 2 models, called SEMST_Original and SEMST_Auto in this work. More details can be founed in the paper.

This repository provides  pre-trained models for you to generate your own image given content image and style image. Also, you can download the training dataset or prepare your own dataset to train the model from scratch.

If you have any question, please feel free to contact me. (Language in English/Japanese/Chinese will be ok!)


---
If you find this work useful for you, please cite it as follow in your paper. Thanks a lot.

```
@misc{Chen2020,
  author = {Chen Chen},
  title = {Structure-emphasized Multimodal Style Transfer},
  year = {2020},
  month = 1,
  doi = 10.5281/zenodo.3602064
  publisher = {Zenodo},
  url = {https://doi.org/10.5281/zenodo.3602064},
}
```
---


## Requirements

- Python 3.7+
- PyTorch 1.0+
- TorchVision
- Pillow

Anaconda environment recommended here!

(optional)

- GPU environment 

---

# Result

Some results of content image will be shown here.

![image](https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer/blob/master/result/5_1.png)
![image](https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer/blob/master/result/5_2.png)
![image](https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer/blob/master/result/5_3.png)
![image](https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer/blob/master/result/5_4.png)
![image](https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer/blob/master/result/5_5.png)
![image](https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer/blob/master/result/5_6.png)
![image](https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer/blob/master/result/5_7.png)
![image](https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer/blob/master/result/5_8.png)

---
---

## Notice: The train and test procedures as follow are the same for SEMST_Original and SEMST_Auto.
---
---

## Test
1. Clone this repository 

   ```bash
   git clone https://github.com/irasin/Structure-emphasized-Multimodal-Style-Transfer
   cd Structure-emphasized-Multimodal-Style-Transfer
   cd SEMST_XXX(XXX means Original or Auto)
   ```

2. Prepare your content image and style image. I provide some in the `content` and `style` and you can try to use them easily.

3. Download the pretrained model [SEMST_Original](https://drive.google.com/file/d/1G9S9nxaMa9N9QJwPfwfP9iZ6LMFYWrln/view?usp=sharing), [SEMST_Auto](https://drive.google.com/file/d/151Fo-O-ImtKNrr5n82oUbRH5ajAlpit9/view?usp=sharing) and put them under the SEMST_XXX respectively.

4. Generate the output image. A transferred output image w/&w/o style image and a NST_demo_like image will be generated.

   ```python
   python test.py -c content_image_path -s style_image_path
   ```

  ```
  usage: test.py [-h] [--content CONTENT] [--style STYLE]
                [--output_name OUTPUT_NAME] [--alpha ALPHA] [--gpu GPU]
                [--model_state_path MODEL_STATE_PATH]

   ```

   If output_name is not given, it will use the combination of content image name and style image name.


------

## Train

1. Download [COCO](http://cocodataset.org/#download) (as content dataset)and [Wikiart](https://www.kaggle.com/c/painter-by-numbers) (as style dataset) and unzip them, rename them as `content` and `style`  respectively (recommended).

2. Modify the argument in the` train.py` such as the path of directory, epoch, learning_rate or you can add your own training code.

3. Train the model using gpu.

4. ```python
   python train.py
   ```

   ```
   usage: train.py [-h] [--batch_size BATCH_SIZE] [--epoch EPOCH] [--gpu GPU]
                [--learning_rate LEARNING_RATE]
                [--snapshot_interval SNAPSHOT_INTERVAL] [--alpha ALPHA]
                [--gamma GAMMA] [--train_content_dir TRAIN_CONTENT_DIR]
                [--train_style_dir TRAIN_STYLE_DIR]
                [--test_content_dir TEST_CONTENT_DIR]
                [--test_style_dir TEST_STYLE_DIR] [--save_dir SAVE_DIR]
                [--reuse REUSE]
   ```


