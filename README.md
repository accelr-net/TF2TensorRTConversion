# **Tensorflow Model to TensorRT Conversion**

<br> </br>

## **Initial Order of executing the program**
<ol>
<li>build_imagenet_data.py</li>
<li>model_conversion.py</li>
<li>validation.py</li>
</ol>


## Initial Requirments :

* The working environment can be setup in 2 ways
    <ol>
    <li>Using Nvidia Docker image
    <li>Setting up a conda environment
    <li>Setting up python virtual environment
    </ol>

<br> </br>

### Using Nvidia Docker image

* First, Install the docker daemon and nvidia-docker packages. [Setting Up Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) .
* Then run the docker image using the following command.


#### Run the Following argument for setting up tensorflow docker container
```bash
nvidia-docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it -p 8880:8880 -v <host drive assests path>:/data  -v <host drive workspace path>:/space --name TFTRT nvcr.io/nvidia/tensorflow:21.04-tf2-py3 
```

* By running above command it will download the necessary tensorflow image if not download previously and create the docker container

<br> </br>

### Setting up a conda environment

* First install and setup the necessary CUDA , cuDNN and TensorRT versions respectivly and setup environment variables as mention in those packages installation guides/

    <ol> 
    <li>CUDA = 11.3 <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/">Installation Guide </a>
    <li>cuDNN = 8.2.0 <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/">Installation Guide </a>
    <li>TensorRT = 7.2.3.4 <a href="https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar">Installation Guide </a>
    </ol>

* Then install the anaconda package in the host machine.
* After setting up conda install the required packages (as required) by importing the requiremensts.yml 

```

conda create -n <environment-name> --rquirements.yml

```
<br> </br>

### Setting up a python environment

* First install and setup the necessary CUDA , cuDNN and TensorRT versions respectivly and setup environment variables as mention in those packages installation guides/

    <ol> 
    <li>CUDA = 11.3 <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/">Installation Guide </a>
    <li>cuDNN = 8.2.0 <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/">Installation Guide </a>
    <li>TensorRT = 7.2.3.4 <a href="https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar">Installation Guide </a>
    </ol>

* Then install the python package in the host machine.
* After setting up install the required packages (as required) by importing the requiremensts.txt 

```

env2/bin/pip install -r requirements.txt

```

<br> </br>


## **Preparing TFRecords using imagent validation dataset**

<br> </br>

* By using build_imagenet.py  code , can generate TFRecords of the imagnet validation dataset.

* This script is a revised version of [TensorFlow-Slim's] (https://github.com/tensorflow/models/tree/master/research/slim) **build_imagenet_data.py** with the difference that this targets the classification task only. Purpose of this script is to convert a set of properly arranged images from Image-Net into TF-Record format.


## Format

The Image-Net images should be in unique synset label name folders, in the following format *(below example is for validation set - 50K images)* :

n01694178  n01843065  n02037110  n02096051  n02107683   ..... n04111531  n04273569  n04456115  n04597913  n07802026


## Initial Steps
* Step 1 - put the imagnet validation dataset path in ```-validation_directory```
* Step 2 - put the tfrecord output path in ```-output_directory``` 


#### Run the Following argument for generating tfrecords

```
python build_imagenet_data.py -validation_directory <input_image_path<> -output_directory <path-of-tf-record-directory>

```

## Output

```
[thread 0]: Processed 1000 of 50000 images in thread batch.
[thread 0]: Processed 2000 of 50000 images in thread batch.
[thread 0]: Processed 3000 of 50000 images in thread batch.
[thread 0]: Processed 4000 of 50000 images in thread batch.
...
...
[thread 0]: Processed 49000 of 50000 images in thread batch.
[thread 0]: Processed 50000 of 50000 images in thread batch.
```

The tf-record file should be inside `path-of-tf-record-directory/validation-00000-of-00001`.

<br> </br>


## **Converting Tensorflow Saved model to TF-TRT**

* By using model_conversion.py  code , can convert TensorFlow Saved model to TFTRT optimized model.

## Initial Steps
* Step 1 - put the tensorflow saved model path in ```--model_path```
* Step 2 - put the tensorrt engine output path in ```--output```
* Step 3 - put the presicion type in ```---precision```

``` Python

python model_conversion.py --input=<directory_path of the tensorflow saved model> --output=<output path tftrt model> --presicion=<presicion value of the model> 

```
<br> </br>
## **Converting Tensorflow Saved model TensorRT Engine**

* For the TensorRT conversion, there are 2 steps need to be followed.

    1. Tensorflow to ONNX
    2. ONNX to TensorRT

<br> </br>
## ***Tensorflow to ONNX*** ##

## Initial Steps
* Step 1 - put the tensorflow saved model path in ```--saved_model```
* Step 2 - put the onnx model output path in ```--output```. 
* Step 3 - put the opset config version in ```--opset```

#### Run the Following argument in the terminal to create ONNX model
``` Python

python -m tf2onnx.convert --saved-model tensorflow-model-path --opset 13 --output model.onnx

```

<br> </br>
## ***ONNX to TensorRT*** ##

## Initial Steps
* Step 1 - put the ONNX model path in ```--onnx```
* Step 2 - put the tensorrt engine output path in ```--saveEngine```. 
* Step 3 - put the  ```--fp16``` tag if the tensorrt engine to be in FP16 precision or omit the tag if the model need to be in FP32.
* Step 4 - put the  ```--explicitBatch``` tag if need to enable explicit batch mode, allowing you to specify the batch size during inference.
* Step 5 - put the batch size in  ```--batch```.

#### Run the Following argument in the terminal to create ONNX model
``` Python

trtexec --onnx=model.onnx --saveEngine=engine.trt --fp16 --explicitBatch --batch=8

```
<br> </br>

## **Validation of Generated TensorRT engines**

* By using validation.py  code , can can validate both Tensorflow, TFTRT and TensorRT optimized model for imagnet validation dataset.


## Initial Steps
* Step 1 - put the generated tfrecord path in ```--input```
* Step 2 - put the corresponding model type wish to validate in ```--type``` . (keywords : "Tensorflow" for Tensorflows models and "TensorRT" for TensorRT engines)
* Step 3 - put the path of saved model or engine file in ```--model``` (when providing the tensorflowsaved model path, provide the path to saved file location and when providing tensorrt engine file, provide the path to engine includine engine file name as well)
* Step 4 - provide the batch size need to be validated in ```--batch_size``` ( when providing batch sizes for tensorrt engine validation, provided the inclusive batch size of the engine)


``` Python

python validation.py --model_path=<directory_path of the model> --input=<tfrecords path> --type=<model_type> --batch_size=<size of the batch>

```