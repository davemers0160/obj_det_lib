#ifndef OBJ_DET_DLL_H
#define OBJ_DET_DLL_H

//#define EXTERN_C
//#include <cstdint>
//#include <string>
//#include <vector>

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
#ifdef OBJ_DLL_EXPORTS
#define OBJ_DLL_API __declspec(dllexport)
#else
#define OBJ_DLL_API __declspec(dllimport)
#endif

#else
#define OBJ_DLL_API
//#ifdef MNIST_DLL_API
//#undef MNIST_DLL_API

//#endif

#endif

// ----------------------------------------------------------------------------------------
struct layer_struct
{
    unsigned int k;
    unsigned int n;
    unsigned int nr;
    unsigned int nc;
    unsigned int size;
};

// ----------------------------------------------------------------------------------------
struct detection_struct
{
    unsigned int x;
    unsigned int y;
    unsigned int w;
    unsigned int h;
    char name[256];

    detection_struct()
    {
        x = 0;
        y = 0;
        w = 0;
        h = 0;
        name[0] = 0;
    }

    detection_struct(unsigned int x_, unsigned int y_, unsigned int w_, unsigned int h_, const char name_[])
    {
        x = x_;
        y = y_;
        w = w_;
        h = h_;
        strcpy(name, name_);
    }

};

// ----------------------------------------------------------------------------------------
struct window_struct
{
    unsigned int w;
    unsigned int h;
    char label[256];    
};

//// ----------------------------------------------------------------------------------------
//typedef struct
//{
//    unsigned int nr;
//    unsigned int nc;
//    unsigned char *data;
//} image;


// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
// This function will initialize the network and load the required weights
    OBJ_DLL_API void init_net(const char *net_name, unsigned int *num_classes, struct window_struct* &det_win, unsigned int *num_win);
#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
// This function will take an RGB image in unsigned char row major order [r,g,b, r,g,b,...]
// as an input and produce a resulting classification of the image.  The input must be 28*28
    //OBJ_DLL_API void run_net(unsigned char* image, unsigned int nr, unsigned int nc, unsigned char* &tiled_img, unsigned char* &det_img);
    OBJ_DLL_API void run_net(unsigned char* image, unsigned int nr, unsigned int nc, unsigned char* &tiled_img, unsigned int *t_nr, unsigned int *t_nc, unsigned char* &det_img, unsigned int *num_dets, struct detection_struct* &dets);
#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
    // This function will output a vector of the output layer for the final classification layer
    //MNIST_DLL_API void get_layer_01(struct layer_struct &data, const float* &data_params);
    OBJ_DLL_API void get_layer_01(struct layer_struct *data, const float* &data_params);
#ifdef __cplusplus
}
#endif

//// ----------------------------------------------------------------------------------------
//#ifdef __cplusplus
//extern "C" {
//#endif
//    //MNIST_DLL_API void get_layer_02(struct layer_struct &data, const float* &data_params);
//    OBJ_DLL_API void get_layer_02(layer_struct *data, const float **data_params);
//#ifdef __cplusplus
//}
//#endif
//
//// ----------------------------------------------------------------------------------------
//#ifdef __cplusplus
//extern "C" {
//#endif
//    //MNIST_DLL_API void get_layer_05(struct layer_struct &data, const float* &data_params);
//    OBJ_DLL_API void get_layer_05(layer_struct *data, const float **data_params);
//#ifdef __cplusplus
//}
//#endif
//
//// ----------------------------------------------------------------------------------------
//#ifdef __cplusplus
//extern "C" {
//#endif
//    //MNIST_DLL_API void get_layer_08(struct layer_struct &data, const float* &data_params);
//    OBJ_DLL_API void get_layer_08(layer_struct *data, const float **data_params);
//#ifdef __cplusplus
//}
//#endif
//
//// ----------------------------------------------------------------------------------------
//#ifdef __cplusplus
//extern "C" {
//#endif
//    //MNIST_DLL_API void get_layer_09(struct layer_struct &data, const float* &data_params);
//    OBJ_DLL_API void get_layer_09(layer_struct *data, const float **data_params);
//#ifdef __cplusplus
//}
//#endif
//
//// ----------------------------------------------------------------------------------------
//#ifdef __cplusplus
//extern "C" {
//#endif
//    //MNIST_DLL_API void get_layer_12(struct layer_struct &data, const float* &data_params);
//    OBJ_DLL_API void get_layer_12(layer_struct *data, const float **data_params);
//#ifdef __cplusplus
//}
//#endif

#endif  // OBJ_DET_DLL_H