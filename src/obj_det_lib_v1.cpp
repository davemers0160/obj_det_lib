#define _CRT_SECURE_NO_WARNINGS

#include <cstdint>
#include <string>
#include <vector>

#if !defined(BUILD_LIB)
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

// Custom includes
#include "overlay_bounding_box.h"
#include "file_ops.h"
#include "obj_det_lib.h"
#include "obj_det_net_v4.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>

using namespace std;

//----------------------------------------------------------------------------------
// library internal state variables:
aobj_net_type net;
double pyr_scale;
std::vector<std::string> class_names;
std::vector<dlib::rgb_pixel> class_color;

//----------------------------------------------------------------------------------
void init_net(const char *net_name, unsigned int *num_classes, struct window_struct* &det_win, unsigned int *num_win)
{
    uint32_t idx;
    dlib::deserialize(net_name) >> net;

    // Get the type of pyramid the CNN used
    using pyramid_type = std::remove_reference<decltype(dlib::input_layer(net))>::type::pyramid_type;        
    pyramid_type tmp_pyr;
    pyr_scale = dlib::pyramid_rate(tmp_pyr);    
    
    // get the details about the loss layer -> the number and names of the classes
    dlib::mmod_options options = dlib::layer<0>(net).loss_details().get_options();
    
    *num_win = options.detector_windows.size();
    det_win = new window_struct[options.detector_windows.size()];
    
    std::set<std::string> tmp_names;
    std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
    for (idx = 0; idx < options.detector_windows.size(); ++idx)
    {
        std::cout << "detector window (w x h): " << options.detector_windows[idx].label << " - " << options.detector_windows[idx].width << " x " << options.detector_windows[idx].height << std::endl;

        det_win[idx].w = options.detector_windows[idx].width;
        det_win[idx].h = options.detector_windows[idx].height;
        strcpy(det_win[idx].label, options.detector_windows[idx].label.c_str());

        tmp_names.insert(options.detector_windows[idx].label);
    }
    std::cout << std::endl;

    // pull out the class names
    for (const auto &it : tmp_names) {
        class_names.push_back(it);
    }
    *num_classes = class_names.size();

    dlib::rand rnd(time(NULL));

    for (idx = 0; idx < *num_classes; ++idx)
    {
        class_color.push_back(dlib::rgb_pixel(rnd.get_random_8bit_number(), rnd.get_random_8bit_number(), rnd.get_random_8bit_number()));
    }

}

//----------------------------------------------------------------------------------
void run_net(unsigned char* input_img, unsigned int nr, unsigned int nc, unsigned char* &tiled_img, unsigned int *t_nr, unsigned int *t_nc, unsigned char* &det_img, unsigned int *num_dets, struct detection_struct* &dets)
{

    uint64_t r, c;
    uint64_t idx = 0;

    dlib::matrix<dlib::rgb_pixel> img(nr, nc);

    for (r = 0; r < nr; ++r)
    {
        for (c = 0; c < nc; ++c)
        {
            // static_cast<unsigned char>
            img(r, c).red = (*(input_img + (idx++)));
            img(r, c).green = (*(input_img + (idx++)));
            img(r, c).blue = (*(input_img + (idx++)));
        }
    }

    dlib::matrix<dlib::rgb_pixel> ti;
    std::vector<dlib::rectangle> rects;
    using pyramid_type = std::remove_reference<decltype(dlib::input_layer(net))>::type::pyramid_type;
    
    // create the pyramid using pyramid type
    dlib::create_tiled_pyramid<pyramid_type>(img, ti, rects,
        dlib::input_layer(net).get_pyramid_padding(),
        dlib::input_layer(net).get_pyramid_outer_padding());

    std::vector<dlib::mmod_rect> d = net(img);
    *num_dets = d.size();
    dets = new detection_struct[d.size()];

    // copy the image into tmp_img so that the original data is not modified
    dlib::matrix<dlib::rgb_pixel> tmp_img;
    dlib::assign_image(tmp_img, img);

    //overlay the dnn detections on the image
    for (idx = 0; idx < d.size(); ++idx)
    {
        auto& class_index = std::find(class_names.begin(), class_names.end(), d[idx].label);
        overlay_bounding_box(tmp_img, d[idx], class_color[std::distance(class_names.begin(), class_index)]);
        dets[idx] = detection_struct(d[idx].rect.left(), d[idx].rect.top(), d[idx].rect.width(), d[idx].rect.height(), d[idx].label.c_str());

    }

    // bring out the tiled image version of the input image
    tiled_img = new unsigned char[ti.nr()*ti.nc()*3];
    *t_nr = ti.nr();
    *t_nc = ti.nc();

    idx = 0;   
    for (r = 0; r < *t_nr; ++r)
    {
        for (c = 0; c < *t_nc; ++c)
        {
            tiled_img[idx++] = ti(r, c).red;
            tiled_img[idx++] = ti(r, c).green;
            tiled_img[idx++] = ti(r, c).blue;
        }
    }

    det_img = new unsigned char[tmp_img.nr()*tmp_img.nc() * 3];

    idx = 0;
    for (r = 0; r < nr; ++r)
    {
        for (c = 0; c < nc; ++c)
        {
            det_img[idx++] = tmp_img(r, c).red;
            det_img[idx++] = tmp_img(r, c).green;
            det_img[idx++] = tmp_img(r, c).blue;
        }
    }

}   // end of run_net

//----------------------------------------------------------------------------------
void get_layer_01(struct layer_struct *data, const float* &data_params)
{
    auto& lo = dlib::layer<1>(net).get_output();
    data->k = lo.k();
    data->n = lo.num_samples();
    data->nr = lo.nr();
    data->nc = lo.nc();
    data->size = lo.size();
    data_params = lo.host();
}

//----------------------------------------------------------------------------------
//void get_layer_02(layer_struct *data, const float **data_params)
//{
//    auto& lo = dlib::layer<2>(net).get_output();
//    data->k = lo.k();
//    data->n = lo.num_samples();
//    data->nr = lo.nr();
//    data->nc = lo.nc();
//    data->size = lo.size();
//    *data_params = lo.host();
//}
//
////----------------------------------------------------------------------------------
//void get_layer_05(layer_struct *data, const float **data_params)
//{
//    auto& lo = dlib::layer<5>(net).get_output();
//    data->k = lo.k();
//    data->n = lo.num_samples();
//    data->nr = lo.nr();
//    data->nc = lo.nc();
//    data->size = lo.size();
//    *data_params = lo.host();
//}
//
////----------------------------------------------------------------------------------
//void get_layer_08(layer_struct *data, const float **data_params)
//{
//    auto& lo = dlib::layer<8>(net).get_output();
//    data->k = lo.k();
//    data->n = lo.num_samples();
//    data->nr = lo.nr();
//    data->nc = lo.nc();
//    data->size = lo.size();
//    *data_params = lo.host();
//}
//
////----------------------------------------------------------------------------------
//void get_layer_09(layer_struct *data, const float **data_params)
//{
//    auto& lo = dlib::layer<9>(net).get_output();
//    data->k = lo.k();
//    data->n = lo.num_samples();
//    data->nr = lo.nr();
//    data->nc = lo.nc();
//    data->size = lo.size();
//    *data_params = lo.host();
//}
//
////----------------------------------------------------------------------------------
//void get_layer_12(layer_struct *data, const float **data_params)
//{
//    auto& lo = dlib::layer<12>(net).get_output();
//    data->k = lo.k();
//    data->n = lo.num_samples();
//    data->nr = lo.nr();
//    data->nc = lo.nc();
//    data->size = lo.size();
//    *data_params = lo.host();
//}

//----------------------------------------------------------------------------------
// check to see if we are building the library or a standalone executable
#if !defined(BUILD_LIB)

int main(int argc, char** argv)
{
    uint32_t idx;
    std::string program_root;
    std::string net_directory;
    std::string image_directory;
    std::string test_net_name;

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = std::chrono::system_clock::now();
    auto stop_time = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast<d_sec>(stop_time - start_time);

    std::vector<std::string> test_images = { "test1.png", "test2.png", "test3.png", "test4.png", "test5.png", "test6.png", "test7.png", "test8.png", "test9.png", "test10.png" };
    
    unsigned int num_classes, num_win;

    unsigned char* tiled_img = NULL;
    unsigned char* det_img = NULL;

    unsigned int t_nr = 0, t_nc = 0;
    window_struct* det;
    unsigned int num_dets = 0;
    struct detection_struct* dets;
    cv::Mat img;
    long nr, nc;

    // setup save variable locations
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
    program_root = path_check(get_path(get_path(get_path(std::string(argv[0]), "\\"), "\\"), "\\"));
#else 
    program_root = get_ubuntu_path();
#endif

    net_directory = program_root + "nets/";
    image_directory = "D:/Projects/object_detection_data/test_plane/test1/";

    try
    {
        test_net_name = (net_directory + "yj_v4_s4_v4_BEAST_final_net.dat");

        // initialize the network
        init_net(test_net_name.c_str(), &num_classes, det, &num_win);

        dlib::matrix<uint8_t> ti;

        // run through some images to test the code
        for (idx = 0; idx < test_images.size(); ++idx)
        {
            img = cv::imread(image_directory + test_images[idx], cv::IMREAD_ANYCOLOR);
            nr = img.rows;
            nc = img.cols;


            unsigned char* image = new unsigned char[nr * nc * 3]{ 0 };

            uint32_t index = 0;
            for (long r = 0; r < nr; ++r)
            {
                for (long c = 0; c < nc; ++c)
                {
                    image[index++] = img.at<cv::Vec3b>(r, c)[2];
                    image[index++] = img.at<cv::Vec3b>(r, c)[1];
                    image[index++] = img.at<cv::Vec3b>(r, c)[0];
                }
            }
            start_time = std::chrono::system_clock::now();

            // void run_net(unsigned char* image, unsigned int nr, unsigned int nc, unsigned char* &tiled_img, unsigned int *t_nr, unsigned int *t_nc, unsigned char* &det_img, unsigned int *num_dets, struct detection_struct* &dets);
            //unsigned char* img2 = img.ptr<unsigned char>(0);
            run_net(img.ptr<unsigned char>(0), nr, nc, tiled_img, &t_nr, &t_nc, det_img, &num_dets, dets);

            stop_time = std::chrono::system_clock::now();
            elapsed_time = std::chrono::duration_cast<d_sec>(stop_time - start_time);

            std::cout << "Runtime (s): " << elapsed_time.count() << std::endl;
        }

    }
    catch (std::exception & e)
    {
        std::cout << std::endl;
        std::cout << e.what() << std::endl;
    }

    std::cout << std::endl << "Program complete.  Press Enter to close." << std::endl;
    std::cin.ignore();
    return 0;

}   // end of main

#endif  // BUILD_LIB
