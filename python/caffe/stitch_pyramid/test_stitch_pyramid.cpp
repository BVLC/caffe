
#include "SimpleOpt.h"
#include "JPEGPyramid.h"
#include "JPEGImage.h" 
#include "Patchwork.h"
#include "PyramidStitcher.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace FFLD;
using namespace std;

// SimpleOpt array of valid options
enum
{
    OPT_HELP, OPT_PADDING, OPT_INTERVAL, OPT_OUTPUT_STITCHED_DIR
};

CSimpleOpt::SOption SOptions[] =
{
	{ OPT_HELP, "-h", SO_NONE },
	{ OPT_HELP, "--help", SO_NONE },
	{ OPT_PADDING, "-p", SO_REQ_SEP },
	{ OPT_PADDING, "--padding", SO_REQ_SEP },
	{ OPT_INTERVAL, "-e", SO_REQ_SEP },
	{ OPT_INTERVAL, "--interval", SO_REQ_SEP },
    { OPT_OUTPUT_STITCHED_DIR, "--output-stitched-dir", SO_REQ_SEP },
	SO_END_OF_OPTIONS
};

void showUsage(){
	cout << "Usage: test [options] image.jpg, or\n       test [options] image_set.txt\n\n"
			"Options:\n"
			"  -h,--help               Display this information\n"
    		"  -p,--padding <arg>      Amount of zero padding in JPEG images (default 8)\n"
			"  -e,--interval <arg>     Number of levels per octave in the JPEG pyramid (default 10)\n"
            "  --output-stitched-dir <arg>   Where to save stitched pyramids (default ../stitched_results)\n"
		 << endl;
}

// Parse command line parameters
//   put the appropriate values in (padding, interval, file) based on cmd-line args
void parseArgs(int &padding, int &interval, string &file, string &output_stitched_dir, int argc, char * argv[]){
	CSimpleOpt args(argc, argv, SOptions);

	while (args.Next()) {
		if (args.LastError() == SO_SUCCESS) {
			if (args.OptionId() == OPT_HELP) {
				showUsage();
				exit(0);
			}
			else if (args.OptionId() == OPT_PADDING) {
				padding = atoi(args.OptionArg());
				
				// Error checking
				if (padding <= 1) {
					showUsage();
					cerr << "\nInvalid padding arg " << args.OptionArg() << endl;
					exit(1);
				}
			}
			else if (args.OptionId() == OPT_INTERVAL) {
				interval = atoi(args.OptionArg());
				
				// Error checking
				if (interval <= 0) {
					showUsage();
					cerr << "\nInvalid interval arg " << args.OptionArg() << endl;
					exit(1);
				}
			}
            else if (args.OptionId() == OPT_OUTPUT_STITCHED_DIR) {
                output_stitched_dir = args.OptionArg();	
            }
        }
		else {
			showUsage();
			cerr << "\nUnknown option " << args.OptionText() << endl;
			exit(1);
		}
	}
	if (!args.FileCount()) {
		showUsage();
		cerr << "\nNo image/dataset provided" << endl;
		exit(1);
	}
	else if (args.FileCount() > 1) {
		showUsage();
		cerr << "\nMore than one image/dataset provided" << endl;
		exit(1);
	}

	// The image/dataset
    file = args.File(0);
	const size_t lastDot = file.find_last_of('.');
	if ((lastDot == string::npos) ||
		((file.substr(lastDot) != ".jpg") && (file.substr(lastDot) != ".txt"))) {
		showUsage();
		cerr << "\nInvalid file " << file << ", should be .jpg or .txt" << endl;
		exit(1);
	}

	// Try to load the image
	if (file.substr(lastDot) != ".jpg") {
        cout << "need to input a JPG image" << endl;
        exit(1);
    }
}

//e.g. file = ../../images_640x480/carsgraz_001.image.jpg
void TEST_file_parsing(string file){
    size_t lastSlash = file.find_last_of("/\\");
    size_t lastDot = file.find_last_of('.');
    cout << "    file.substr(lastDot) = " << file.substr(lastDot) << endl; // .jpg
    cout << "    file.substr(lastSlash) = " << file.substr(lastSlash) << endl; // /carsgraz_001.image.jpg
    cout << "    file.substr(lastSlash, lastDot) = " << file.substr(lastSlash, lastDot-lastSlash) << endl; // /carsgraz_001.image
}

//e.g. file = ../../images_640x480/carsgraz_001.image.jpg
string parse_base_filename(string file){
    size_t lastSlash = file.find_last_of("/\\");
    size_t lastDot = file.find_last_of('.');

    string base_filename = file.substr(lastSlash, lastDot-lastSlash); // /carsgraz_001.image
    return base_filename;
}

void printScaleSizes(JPEGPyramid pyramid);
void writePyraToJPG(JPEGPyramid pyramid);
void writePatchworkToJPG(Patchwork patchwork, string output_stitched_dir, string base_filename);
void print_scaleLocations(vector<ScaleLocation> scaleLocations);
void print_scales(Patchwork patchwork);

//TODO: split this test into its own function, e.g. test_stitch_pyramid()
int main(int argc, char * argv[]){

	// Default parameters
    string file;
    string output_stitched_dir = "../stitched_results";
	int padding = 8;
	int interval = 10;

    //parseArgs params are passed by reference, so they get updated here
    parseArgs(padding, interval, file, output_stitched_dir, argc, argv); //update parameters with any command-line inputs
    string base_filename = parse_base_filename(file);

    printf("    padding = %d \n", padding);
    printf("    interval = %d \n", interval);
    printf("    file = %s \n", file.c_str());
    printf("    base_filename = %s \n", base_filename.c_str());
    printf("    output_stitched_dir = %s \n", output_stitched_dir.c_str());

    Patchwork patchwork = stitch_pyramid(file, padding, interval, -1); //planeDim = -1 (use defaults)
    //printScaleSizes(pyramid);
    //writePyraToJPG(pyramid);
    writePatchworkToJPG(patchwork, output_stitched_dir, base_filename); //outputs to output_stitched_dir/base_filename_[planeID].jpg

    int convnet_subsampling_ratio = 1; // we're not actually computing convnet features in this test, 
                                       // so there's no feature downsampling.
    vector<ScaleLocation> scaleLocations =  unstitch_pyramid_locations(patchwork, convnet_subsampling_ratio);
    //print_scaleLocations(scaleLocations);
    print_scales(patchwork);

   	return EXIT_SUCCESS;
}

void printScaleSizes(JPEGPyramid pyramid){
    int nlevels = pyramid.levels().size();

    for(int level = 0; level < nlevels; level++){ 
        int width = pyramid.levels()[level].width();
        int height = pyramid.levels()[level].height();
        int depth = pyramid.NbChannels;
        printf("        level %d: width=%d, height=%d, depth=%d \n", level, width, height, depth);
    }
}

void print_scaleLocations(vector<ScaleLocation> scaleLocations){
    printf("scaleLocations: \n");
    for(int i=0; i<scaleLocations.size(); i++){
        printf("    idx=%d, xMin=%d, xMax=%d, width=%d, yMin=%d, yMax=%d, height=%d, planeID=%d \n", 
               i, scaleLocations[i].xMin, scaleLocations[i].xMax, scaleLocations[i].width,
                  scaleLocations[i].yMin, scaleLocations[i].yMax, scaleLocations[i].height, scaleLocations[i].planeID); 
    }
}

//assumes NbChannels == 3
void writePyraToJPG(JPEGPyramid pyramid){
    int nlevels = pyramid.levels().size();

    for(int level = 0; level < nlevels; level++){
        ostringstream fname;
        fname << "../pyra_results/level" << level << ".jpg"; //TODO: get orig img name into the JPEG name.
        //cout << fname.str() << endl;

        pyramid.levels()[level].save(fname.str());
    }
}

void writePatchworkToJPG(Patchwork patchwork, string output_stitched_dir, string base_filename){
    int nplanes = patchwork.planes_.size();

    for(int planeID = 0; planeID < nplanes; planeID++){
        ostringstream fname;
        fname << output_stitched_dir << "/" << base_filename << "_plane" << planeID << ".jpg";
        //cout << fname.str() << endl;

        patchwork.planes_[planeID].save(fname.str());
    }
}

void print_scales(Patchwork patchwork){
    for(int i=0; i<patchwork.scales_.size(); i++){
        printf("    patchwork.scales_[%d] = %f \n", i, patchwork.scales_[i]);
    }
}

