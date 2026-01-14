# FAFN
Code release for **Multi-Scale Spatial–Frequency Collaboration Network for Few-Shot Fine-Grained Learning**.

## Dataset

The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN), but the categories  of train, val, and test follows split.txt. And then move the processed dataset  to directory ./data.

- CUB_200_2011 \[[Download Link](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view)\]
- cars \[[Download Link](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link)\]
- dogs \[[Download Link](https://drive.google.com/file/d/13avzK22oatJmtuyK0LlShWli00NsF6N0/view?usp=drive_link)\]

## Train

* To train MS-SFC on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  python train.py -batch 64 -dataset cub -gpu 0 -extra_dir cub
  ```


## Test

```shell
python test.py -batch 64 -dataset cub -gpu 0 -extra_dir cub 
```

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:

- yly@stu.xidian.edu.cn
