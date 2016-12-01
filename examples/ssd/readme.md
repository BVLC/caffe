To run the SSD example:

1) Prepare/Download the trained model, for example, download from http://www.cs.unc.edu/%7Ewliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz

2) Run the SSD example: ./build/examples/ssd/ssd_detect.bin ./examples/ssd/deploy.prototxt models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel ./examples/ssd/images.txt

Note: Please use the modified deploy.prototxt to get much better performance.
