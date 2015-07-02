
import os
import sys
import shutil
import subprocess

INV_DURATION = 1
FPS = 10

def create_movie(input_dir, output_file):
	cmd_str = "ffmpeg -framerate %s -i %s/%%03d.png -c:v libx264 -r %s -pix_fmt yuv420p %s" % (INV_DURATION, input_dir, FPS, output_file)
	print cmd_str
	subprocess.call(cmd_str, shell=True)

def run_snapshots(model_file, snapshot_dir, out_dir, images):
	tmp_out = os.path.join(out_dir, "snapshots")
	iter_nums = list()
	for f in os.listdir(snapshot_dir):
		if not f.endswith(".caffemodel"):
			continue
		r = os.path.join(snapshot_dir, f)
		iter_num = int(f[f.rfind('_') + 1:-1 * len(".caffemodel")])
		iter_nums.append(iter_num)
		snapshot_out = os.path.join(tmp_out, str(iter_num))
		cmd_str = "python python/visualize_net.py %s %s %s %s" % (model_file, r, snapshot_out, " ".join(images))
		print cmd_str
		subprocess.call(cmd_str, shell=True)
	return iter_nums
	
def get_filter_names(out_dir):
	tmp_out = os.path.join(out_dir, "snapshots")
	sub_dir = os.path.join(tmp_out, os.listdir(tmp_out)[0], 'filters')
	return map(lambda f: os.path.splitext(f)[0], os.listdir(sub_dir))
	
def get_activation_names(out_dir):
	tmp_out = os.path.join(out_dir, "snapshots")
	sub_dir = os.path.join(tmp_out, os.listdir(tmp_out)[0])
	for s_dir in os.listdir(sub_dir):
		if s_dir != 'filters':
			return map(lambda f: os.path.splitext(f)[0], os.listdir(os.path.join(sub_dir, s_dir)))

def copy_filters(out_dir, iter_nums, filter_names):
	tmp_out = os.path.join(out_dir, "snapshots")
	filter_image_dir = os.path.join(out_dir, "filters", "images")
	os.makedirs(filter_image_dir)
	for _filter in filter_names:
		to_dir = os.path.join(filter_image_dir, _filter)
		os.mkdir(to_dir)
		for x, iter_num in enumerate(iter_nums, 1):
			src_file = os.path.join(tmp_out, str(iter_num), "filters", "%s.png" % _filter)
			dest_file = os.path.join(to_dir, "%03d.png" % x)
			shutil.copy(src_file, dest_file)
						
			
def copy_activations(out_dir, iter_nums, image_basenames, activation_names):
	tmp_out = os.path.join(out_dir, "snapshots")
	for image_basename in image_basenames:
		image_dir = os.path.join(out_dir, "images", image_basename, "images")
		os.makedirs(image_dir)
		for activation in activation_names:
			to_dir = os.path.join(image_dir, activation)
			os.mkdir(to_dir)
			for x, iter_num in enumerate(iter_nums, 1):
				src_file = os.path.join(tmp_out, str(iter_num), image_basename, "%s.png" % activation)
				dest_file = os.path.join(to_dir, "%03d.png" % x)
				shutil.copy(src_file, dest_file)

def create_filter_movies(out_dir, filter_names):
	filters_dir = os.path.join(out_dir, "filters")
	movies_dir = os.path.join(filters_dir, "movies")
	images_dir = os.path.join(filters_dir, "images")
	os.mkdir(movies_dir)
	for _filter in filter_names:
		in_dir = os.path.join(images_dir, _filter)
		out_file = os.path.join(movies_dir, "%s.mp4" % _filter)
		create_movie(in_dir, out_file)
		
def create_image_movies(out_dir, image_basenames, activation_names):
	for image_name in image_basenames:
		image_dir = os.path.join(out_dir, "images", image_name)
		movies_dir = os.path.join(image_dir, "movies")
		images_dir = os.path.join(image_dir, "images")
		os.mkdir(movies_dir)
		for activation in activation_names:
			in_dir = os.path.join(images_dir, activation)
			out_file = os.path.join(movies_dir, "%s.mp4" % activation)
			create_movie(in_dir, out_file)

def main(model_file, snapshot_dir, out_dir, images):
	# set up output file
	if os.path.exists(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir)
	
	iter_nums = run_snapshots(model_file, snapshot_dir, out_dir, images)
	iter_nums.sort()

	image_basenames = map(lambda f: os.path.basename(os.path.splitext(f)[0]), images)
	filter_names = get_filter_names(out_dir)
	activation_names = get_activation_names(out_dir)
	print "Image Basenames:", image_basenames
	print "Filter Names:", filter_names
	print "Activation Names:", activation_names
	
	copy_filters(out_dir, iter_nums, filter_names)
	copy_activations(out_dir, iter_nums, image_basenames, activation_names)
	
	create_filter_movies(out_dir, filter_names)
	create_image_movies(out_dir, image_basenames, activation_names)

if __name__ == "__main__":
	model_file = sys.argv[1]
	snapshot_dir = sys.argv[2]
	out_dir = sys.argv[3]
	images = sys.argv[4:]
	main(model_file, snapshot_dir, out_dir, images)
	
