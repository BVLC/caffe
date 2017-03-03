#!/usr/bin/env python
# 
# All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
from mincepie import mapreducer, launcher
import gflags
import os
import cv2
from PIL import Image

# gflags
gflags.DEFINE_string('image_lib', 'opencv',
                     'OpenCV or PIL, case insensitive. The default value is the faster OpenCV.')
gflags.DEFINE_string('input_folder', '',
                     'The folder that contains all input images, organized in synsets.')
gflags.DEFINE_integer('output_side_length', 256,
                     'Expected side length of the output image.')
gflags.DEFINE_string('output_folder', '',
                     'The folder that we write output resized and cropped images to')
FLAGS = gflags.FLAGS

class OpenCVResizeCrop:
    def resize_and_crop_image(self, input_file, output_file, output_side_length = 256):
        '''Takes an image name, resize it and crop the center square
        '''
        img = cv2.imread(input_file)
        height, width, depth = img.shape
        new_height = output_side_length
        new_width = output_side_length
        if height > width:
            new_height = output_side_length * height / width
        else:
            new_width = output_side_length * width / height
        resized_img = cv2.resize(img, (new_width, new_height))
        height_offset = (new_height - output_side_length) / 2
        width_offset = (new_width - output_side_length) / 2
        cropped_img = resized_img[height_offset:height_offset + output_side_length,
                                  width_offset:width_offset + output_side_length]
        cv2.imwrite(output_file, cropped_img)

class PILResizeCrop:
## http://united-coders.com/christian-harms/image-resizing-tips-every-coder-should-know/
    def resize_and_crop_image(self, input_file, output_file, output_side_length = 256, fit = True):
        '''Downsample the image.
        '''
        img = Image.open(input_file)
        box = (output_side_length, output_side_length)
        #preresize image with factor 2, 4, 8 and fast algorithm
        factor = 1
        while img.size[0]/factor > 2*box[0] and img.size[1]*2/factor > 2*box[1]:
            factor *=2
        if factor > 1:
            img.thumbnail((img.size[0]/factor, img.size[1]/factor), Image.NEAREST)

        #calculate the cropping box and get the cropped part
        if fit:
            x1 = y1 = 0
            x2, y2 = img.size
            wRatio = 1.0 * x2/box[0]
            hRatio = 1.0 * y2/box[1]
            if hRatio > wRatio:
                y1 = int(y2/2-box[1]*wRatio/2)
                y2 = int(y2/2+box[1]*wRatio/2)
            else:
                x1 = int(x2/2-box[0]*hRatio/2)
                x2 = int(x2/2+box[0]*hRatio/2)
            img = img.crop((x1,y1,x2,y2))

        #Resize the image with best quality algorithm ANTI-ALIAS
        img.thumbnail(box, Image.ANTIALIAS)

        #save it into a file-like object
        with open(output_file, 'wb') as out:
            img.save(out, 'JPEG', quality=75)

class ResizeCropImagesMapper(mapreducer.BasicMapper):
    '''The ImageNet Compute mapper. 
    The input value would be the file listing images' paths relative to input_folder.
    '''
    def map(self, key, value):
        if type(value) is not str:
            value = str(value)
        files = [value]
        image_lib = FLAGS.image_lib.lower()
        if image_lib == 'pil':
            resize_crop = PILResizeCrop()
        else:
            resize_crop = OpenCVResizeCrop()
        for i, line in enumerate(files):
            try:
                line = line.replace(FLAGS.input_folder, '').strip()
                line = line.split()
                image_file_name = line[0]
                input_file = os.path.join(FLAGS.input_folder, image_file_name)
                output_file = os.path.join(FLAGS.output_folder, image_file_name)
                output_dir = output_file[:output_file.rfind('/')]
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                feat = resize_crop.resize_and_crop_image(input_file, output_file,
                                                              FLAGS.output_side_length)
            except Exception, e:
                # we ignore the exception (maybe the image is corrupted?)
                print line, Exception, e
        yield value, FLAGS.output_folder

mapreducer.REGISTER_DEFAULT_MAPPER(ResizeCropImagesMapper)

mapreducer.REGISTER_DEFAULT_READER(mapreducer.FileReader)
mapreducer.REGISTER_DEFAULT_WRITER(mapreducer.FileWriter)
 
if __name__ == '__main__':
    launcher.launch()
