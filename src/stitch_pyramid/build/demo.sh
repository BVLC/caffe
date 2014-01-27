#./stitch_pyramid ../../../images_640x480/carsgraz_001.image.jpg
#./stitch_pyramid --padding 8 ../../images_640x480/carsgraz_001.image.jpg

./test_stitch_pyramid --padding 8 --output-stitched-dir ./stitched_results ../../../python/caffe/imagenet/pascal_009959.jpg

#for paper figs:
#./test_stitch_pyramid --padding 16 --output-stitched-dir ~/paper-writing/ICML14_dense_convnet/figures/featpyramid_figs/stitched_img ~/paper-writing/ICML14_dense_convnet/figures/bicycle.jpg


