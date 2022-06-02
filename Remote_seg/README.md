# BiSeNet
My implementation of [BiSeNet](https://arxiv.org/abs/1808.00897). My environment is pytorch1.0 and python3, the code is not tested with other environments, but it should also work on similar environments.


### Get cityscapes dataset
Register and download the dataset from the official [website](https://www.cityscapes-dataset.com/). Then decompress them in the `data/` directory:  
```
    $ mkdir -p data
    $ mv /path/to/leftImg8bit_trainvaltest.zip data
    $ mv /path/to/gtFine_trainvaltest.zip data
    $ cd data
    $ unzip leftImg8bit_trainvaltest.zip
    $ unzip gtFine_trainvaltest.zip
```

### Train and evaluation
Just run the train script: 
```
    $ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```
This would take almost one day on a two 1080ti gpu, and the mIOU will also be computed after training is done.  
You can also run the evaluate alone script after training:  
```
    $ python evaluate.py
```


### Pretrained models
In order to prove myself not a cheater, I prepared pretrained models. You may download the model [here](https://pan.baidu.com/s/1z4z01v8kiqyj0fxUB89KNw) with extraction code `4efc`. Download the `model_final.pth` file and put it in the `res/` directory and then run:
```
    $ python evaluate.py
```
After half a hour, you will see the result of 78.45 mIOU.  

I recommend you to use the 'diss' version which does not contain the `spatial path`. This version is faster and lighter without performance reduction. You can download the pretrained model with this [link](https://pan.baidu.com/s/1wWhYZcABWMceZdmJWF_wxQ) and the extraction code is `4fbx`. Put this `model_final_diss.pth` file under your `res/` directory and then you can run this script to test it:  
```
    $ python diss/evaluate.py
```
This model achieves 78.48 mIOU.  


Note:  
Since I used randomly generated seed for the random operations, the results may fluctuate within the range of [78.17, 78.72], depending on the specific random status during training. I am lucky to have captured a result of 78.4+ mIOU. If you want to train your own model from scratch, please make sure that you are lucky too.



### Tricks:  
These are the tricks that I find might be useful:  
1. use online hard example mining loss. This let the model be trained more efficiently.  
2. do not add weight decay when bn parameters and bias parameters of nn.Conv2d or nn.Linear are tuned.  
3. use a 10 times larger lr at the model output layers.  
4. use crop evaluation. We do not want the eval scale to be too far away from the train scale, so we crop the chips from the images to do evaluation and then combine the results to make the final prediction.  
5. multi-scale training and multi-scale-flip evaluating. On each scale, the scores of the original image and its flipped version are summed up, and then the exponential of the sum is computed to be the prediction of this scale.   
6. warmup of 1000 iters to make sure the model better initialized.   



## Diss this paper:  

#### Old Iron Double Hit 666

<p align='center'>
<img src='pic.jpg'>
</p>

Check it out:  

The authors have proposed a new model structure which is claimed to achieve the state of the art of 78.4 mIOU on cityscapes. However, I do not think this two-branch structure is the key to this result. It is the tricks which are used to train the model that really helps. 

Yao~ Yao~

If we need some features with a downsample rate of 1/8, we can simply use the resnet feature of the layer `res3b1`, like what the [DeepLabv3+](https://arxiv.org/abs/1802.02611) does. It is actually not necessary to add the so-called spatial path. To prove this, I changed the model a little by replacing the spatial path feature with the resnet `res3b1` feature. The associated code is in the `diss` folder. We can train the modified model like this:   
```
    $ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 diss/train.py
```
After 20h training, you can see a result of `mIOU=78.48`, which is still close to the result reported in the paper(mIOU=78.4).  

What is worth mentioning is that, the modified model can be trained faster than the original version and requires less memory since we have eliminated the cost brought by the spatial path.  

Yao Yao Yao~

From the experiment, we can know that this model proposed in the paper is just some encoder-decoder structure with some attention modules added to improve its complication. By using the u-shape model with the same tricks, we can still achieve the same result. Therefore, I feel that the real contribution of this paper is the successful usage of the training and evaluating tricks, though the authors made little mention of these tricks and only advocates their model structures in the paper.   

Skr Skr~
