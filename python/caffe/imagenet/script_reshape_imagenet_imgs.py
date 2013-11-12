from mincepie import mapreducer, launcher
import gflags
import glob
import os

gflags.DEFINE_string("input_folder", "",
                     "The folder that contains all input folders")
gflags.DEFINE_string("output_folder", "",
                     "The folder that we write output images to")
gflags.DEFINE_bool("subfolder", True,
                   "If subfolder is true, the input key is a subfolder"
                   " instead of a file name (like in ILSVRC train).")
gflags.DEFINE_string("ext", "JPEG", "The image extension.")
FLAGS = gflags.FLAGS

class ConvertMapper(mapreducer.BasicMapper):
    """The ImageNet Compute mapper. The input value would be a synset name.
    """
    def map(self, key, value):
        if FLAGS.subfolder:
            files = glob.glob(os.path.join(FLAGS.input_folder, value, '*.' + FLAGS.ext))
            try:
                os.makedirs(os.path.join(FLAGS.output_folder, value))
            except OSError:
                pass
            for file in files:
                os.system('convert %s -resize 256x256\\! %s' %
                    (file, os.path.join(FLAGS.output_folder, value, os.path.basename(file))))
        else:
            os.system('convert %s -resize 256x256\\! %s' %
                    (os.path.join(FLAGS.input_folder, value),
                     os.path.join(FLAGS.output_folder, value)))
        yield value, 'done'

mapreducer.REGISTER_DEFAULT_MAPPER(ConvertMapper)
mapreducer.REGISTER_DEFAULT_REDUCER(mapreducer.IdentityReducer)
mapreducer.REGISTER_DEFAULT_READER(mapreducer.FileReader)
mapreducer.REGISTER_DEFAULT_WRITER(mapreducer.FileWriter)

if __name__ == "__main__":
    launcher.launch()
