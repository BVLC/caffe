import os
import sys

def readFile(filePath):
    file = open(filePath, 'r')
    lines = file.readlines()
    file.close()
    return lines

def writeFile(filePath, lines):
    file = open(filePath, 'w')
    for line in lines:
        file.write(line)
    file.close()

def gen_header(input_file):
    lines = readFile(input_file)
    outputs = []
    for index in range(1, len(lines)):
        line = lines[index].strip()
        output = ""
        items = line.split("\t")
        if len(items) == 1: continue
        #['conv1_1', '3', '64', '300', '300', '300', '300', '3', '3', '1', '', '', '154829568', '0.52%', '1', '1']
        conv_name = items[0]
        ic = items[1]
        oc = items[2]
        ih = items[3]
        iw = items[4]
        kh = items[7]
        kw = items[8]
        ph = items[14]
        pw = items[15]
        s = items[9]
        output += "# if defined(SSD_VGG16_" + conv_name.upper() + "_" + kh + "X" + kw + ")\n"
	output += "    # define NUM_IFMs " + ic + "\n"
	output += "    # define NUM_OFMs " + oc + "\n"
	output += "    # define KERNEL_H " + kh + "\n"
	output += "    # define KERNEL_W " + kw + "\n"
	output += "    # define IFM_H_NOPAD " + ih + "\n"
	output += "    # define IFM_W_NOPAD " + iw + "\n"
	output += "    # define PAD_H " + ph + "\n"
	output += "    # define PAD_W " + pw + "\n"
	output += "    # define STRIDE_H " + s + "\n"
	output += "    # define STRIDE_W " + s + "\n"
        # FIXME: enhance to detect whether RELU and RESIDUAL is after convolution
	output += "    # define RELU 1\n"
	output += "    # define RESIDUAL 0\n"
        output += "#endif\n\n"
        outputs.append(output)
       
    writeFile("header.h", outputs)
 
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usgae: python gen_header.py input_file"
        sys.exit(0)

    gen_header(sys.argv[1])
