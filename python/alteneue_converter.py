__author__ = 'jeremy'
import argparse

'''
#This takes an 'old timey' net and converts to the new format 
#layers->layer
#old data format to new format
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert old caffe style to new')
    parser.add_argument('input_file', help='old caffe prototxt')
    parser.add_argument('output_file', help='new caffe prototxt')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        with open(args.output_file,'w') as g:
            in_data_section = False
            for line in f:
#                print('incoming line:'+line)
                if in_data_section:
#                    print('already in data section')
                    if not 'input_dim' in line:
#                        print('end of data section')
                        in_data_section = False
                        input_param_line = input_param_line + ' } } \n}\n'
                        g.write(input_param_line)
                    else:
                        val = line.split()[-1]
                        input_param_line = input_param_line+' dim: '+str(val)
                        continue
                if ('input' in line and 'data' in line.lower()):
#                    print('now in data section')
                    in_data_section = True
                    line = 'layer { \n  name: \"data\"\n  type:\"Input\" \n  top: \"data\" \n '
                    input_param_line = ' input_param { shape: { '
                if 'layers' in line:
                    line = line.replace('layers','layer')
                if 'type' in line and not 'data' in line:   #e.g. turn type POOLING into type "Pooling"
                    the_type = line.split()[-1]
                    new_type = the_type.lower()
                    new_type = new_type.capitalize()
#                    new_type = new_type.title()
                    if '_' in line:  #e.g. turn type INNER_PRODUCT into type "InnerProduct"
                        parts = new_type.split('_')
                        new_type = ''
                        for part in parts:
                            print('part '+part.capitalize())
                            new_type = new_type + part.capitalize()
                    if not '"' in new_type:
                        new_type = '\"'+ new_type+ '\"'
                    line = ' type: '+new_type + '\n'
#                print('new line:'+line)
                g.write(line)
