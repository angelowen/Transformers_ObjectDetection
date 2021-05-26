**DE⫶TR**: End-to-End Object Detection with Transformers
========


PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer).
We replace the full complex hand-crafted object detection pipeline with a Transformer, and match Faster R-CNN with a ResNet-50, obtaining **42 AP** on COCO using half the computation power (FLOPs) and the same number of parameters. Inference in 50 lines of PyTorch.

![DETR](.github/DETR.png)

**What it is**. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. 
Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.


## Train
`python main.py --dataset_file word --data_path ./my-dataset/  --resume detr-r101-dc5.pth --epochs 300 `

## Test
`python test.py --data_path ./test_images --resume ./output/checkpoint.pth`


# Pretrained Model 
* https://github.com/angelowen/Transformers_ObjectDetection#model-zoo

## Data preparation

```
CustomDataset/
  annotations/  # annotation json files
  train/    # train images
  val/      # val images
```

