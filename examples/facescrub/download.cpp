#include <algorithm>
#include <curl/curl.h>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef WINDOWS
    #include <direct.h>
    #define GetCurrentDir _getcwd
#else
    #include <unistd.h>
    #define GetCurrentDir getcwd
#endif

struct stat info;

// http://stackoverflow.com/questions/18100097/portable-way-to-check-if-directory-exists-windows-linux-c
bool existsDir(std::string& pathname) {

    if (stat(pathname.c_str(), &info) != 0)
        return false;
    else if (info.st_mode & S_IFDIR) // S_ISDIR() doesn't exist on my windows
        return true;
    else
        return false;
}

bool createDir(std::string& path) {

    if (existsDir(path)) {
        return true;
    }

    const int dir_err = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == dir_err)
    {
        printf("Error creating directory!n");
        exit(1);
    }

    return true;
}

bool removeFile(std::string& path) {
    if (remove(path.c_str()) != 0)
        return false;
    else
        return true;
    return 0;
}

size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    size_t written;
    written = fwrite(ptr, size, nmemb, stream);
    return written;
}

std::string getFileName(const std::string& s) {

    char sep = '/';

  #ifdef _WIN32
    sep = '\\';
  #endif

    size_t i = s.rfind(sep, s.length());
    if (i != std::string::npos) {
        return (s.substr(i + 1, s.length() - i));
    }

    return ("");
}

int main(int argc, char** argv) {

    if (argc < 2) {
        return 1;
    }

    std::ifstream file(argv[1]);
    std::vector<std::pair<int, int> > num_images_per_identity;
    std::string name;
    std::string face_id;
    std::string url;
    std::string bbox;

    std::string root_folder;
    if (argc >= 3) {
        root_folder = argv[2];
    }
    else {
        char cCurrentPath[FILENAME_MAX];
        if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath)))
        {
            return 1;
        }
        cCurrentPath[sizeof(cCurrentPath) - 1] = '\0';
        root_folder = cCurrentPath;
    }

    std::string last_name = "";
    int count = 0;
    int identity_count = -1;
    int num_identities = 3;

    std::ofstream sizefile;
    sizefile.open(root_folder + "/sizefile");

    std::ofstream listfile;
    listfile.open(root_folder + "/listfile");

    // create images folder if not exists
    std::string path = root_folder + "/images";
    createDir(path);

    CURL* curl;
    FILE* fp;
    CURLcode curl_code;
    curl = curl_easy_init();
    if (!curl) {
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {

        std::stringstream linestream(line);
        int i = 0;
        std::string value;
        while (getline(linestream, value, '\t')) {
            switch (i) {
            case 0:
                name = value;
            case 2:
                face_id = value;
            case 3:
                url = value;
            case 4:
                bbox = value;
            }
            i++;
        }

        if (name.compare("name") == 0) {
            continue;
        }

        if (last_name != name) {
            identity_count++;
            // if (identity_count == num_identities) {
            //   break;
            // }
            sizefile << std::to_string(identity_count) << "\t" << std::to_string(count) << "\n";

            // create directory for current identity if not exists
            std::string path = root_folder + "/images/" + std::to_string(identity_count);
            createDir(path);
        }

        last_name = name;

        // set absolute path
        std::string absolute_path = root_folder + "/images/" + std::to_string(identity_count) + "/" + getFileName(url);

        fp = fopen(absolute_path.c_str(), "wb");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        //curl_easy_setopt (curl, CURLOPT_VERBOSE, 1L);
        curl_code = curl_easy_perform(curl);

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code == 200 && curl_code != CURLE_ABORTED_BY_CALLBACK)
        {
            try {
                //crop image
                cv::Mat cv_img_full = cv::imread(absolute_path, cv::IMREAD_UNCHANGED);
                std::vector<int> corr;
                std::istringstream f(bbox);
                std::string s;
                while (getline(f, s, ',')) {
                    corr.push_back(std::stoi(s));
                }
                cv::Mat cropedImage = cv_img_full(cv::Rect(corr[0], corr[1], corr[2] - corr[0], corr[3] - corr[1]));
                if (cv::imwrite(absolute_path, cropedImage)) {
                    std::string relative_path = "/images/" + std::to_string(identity_count) + "/" + getFileName(url);
                    listfile << face_id << "\t" << std::to_string(identity_count) << "\t" << relative_path << "\n";
                    count++;
                    std::cout << absolute_path << std::endl;
                }
                else {
                    removeFile(absolute_path);
                }
            } catch (cv::Exception e) {
                removeFile(absolute_path);
            }
        }
        else
        {
            // log error
            removeFile(absolute_path);
        }

        fclose(fp);
        curl_easy_reset(curl);
    }

    curl_easy_cleanup(curl);

    sizefile << std::to_string(identity_count + 1) << "\t" << std::to_string(count) << "\n";

    std::cout << "sizefile and listfile written" << std::endl;

    sizefile.close();
    listfile.close();

    return 1;
}
