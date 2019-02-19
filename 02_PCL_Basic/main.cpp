#include <stdlib.h>
#include <iostream>
#include <string>
#include "OpenNI.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <pcl/io/openni2_grabber.h>
#include <pcl/io/grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>

using namespace std;
using namespace cv;
using namespace openni;
using namespace pcl;

Device device;
VideoStream oniDepthStream;
VideoStream oniColorStream;
VideoFrameRef oniDepthImg;
VideoFrameRef oniColorImg;

#define USE_SENSOR_XTION    0
#define USE_SENSOR_KINECT2  1

#define USE_PCL_VIEWER      0
#define USE_CLOUD_VIEWER    1



void initOpenNI()
{
    Status result = STATUS_OK;
    result = OpenNI::initialize();
    if (result != openni::STATUS_OK)
    {
        printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
        exit(0);
    }
}

void initDevice()
{
    Status result = STATUS_OK;
    // open device
    result = device.open(ANY_DEVICE);
    if (result != openni::STATUS_OK)
    {
        printf("Couldn't open device\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }
}

void initDepthStream()
{
    int nResolutionX = 0;
    int nResolutionY = 0;
    int nDepthFormat = 0;
    Status result = STATUS_OK;
    VideoMode modeDepth;
    const SensorInfo* pSensorInfo;

    result = oniDepthStream.create(device, SENSOR_DEPTH);
    pSensorInfo = device.getSensorInfo(SENSOR_DEPTH);
    
    const Array<VideoMode>& videoModeDepthArray = pSensorInfo->getSupportedVideoModes();
    for (int i = 0; i < videoModeDepthArray.getSize(); ++i)
    {
        const VideoMode& vmode = videoModeDepthArray[i];
        nResolutionX = vmode.getResolutionX();
        nResolutionY = vmode.getResolutionY();
        nDepthFormat = vmode.getPixelFormat();
        cout<<"depth nResolutionX = "<<nResolutionX<<endl;
        cout<<"depth nResolutionY = "<<nResolutionY<<endl;
        cout<<endl;
    }
    if(USE_SENSOR_XTION)
    {
        modeDepth.setResolution( 640, 480 );
    }
    else if(USE_SENSOR_KINECT2)
    {
        modeDepth.setResolution( 512, 424 );
    }
    modeDepth.setFps( 30 );
    modeDepth.setPixelFormat( PIXEL_FORMAT_DEPTH_1_MM );
    oniDepthStream.setVideoMode(modeDepth);
    // start depth stream
    result = oniDepthStream.start();
    if (result != openni::STATUS_OK)
    {
        printf("Couldn't start the depth stream\n%s\n", OpenNI::getExtendedError());
        exit(0);
    }
}

void initColorStream()
{
    int nResolutionX = 0;
    int nResolutionY = 0;
    Status result = STATUS_OK;
    const SensorInfo* pSensorInfo;
    pSensorInfo = device.getSensorInfo(SENSOR_COLOR);
    result = oniColorStream.create( device, SENSOR_COLOR );
    const Array<VideoMode>& videoModeColorArray = pSensorInfo->getSupportedVideoModes();
    cout<<"color resolutions:"<<endl<<endl;
    for (int i = 0; i < videoModeColorArray.getSize(); ++i)
    {
        const VideoMode& vmode = videoModeColorArray[i];
        nResolutionX = vmode.getResolutionX();
        nResolutionY = vmode.getResolutionY();
        cout<<"color nResolutionX = "<<nResolutionX<<endl;
        cout<<"color nResolutionY = "<<nResolutionY<<endl;
        cout<<endl;
    }
    
    // set color video mode
    VideoMode modeColor;
    if(USE_SENSOR_XTION)
    {
        modeColor.setResolution( 640, 480 );
        // modeColor.setResolution( 1280, 1024 );
    }
    else if(USE_SENSOR_KINECT2)
    {
        // modeColor.setResolution( 512, 424 );
        modeColor.setResolution( 1920, 1080 );
    }
    modeColor.setFps( 30 );
    modeColor.setPixelFormat( PIXEL_FORMAT_RGB888 );
    oniColorStream.setVideoMode( modeColor);

    // set depth and color imge registration mode
    if( device.isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR ) )
    {
        cout<<"device suppport registration mode"<<endl;
        // device.setImageRegistrationMode( IMAGE_REGISTRATION_DEPTH_TO_COLOR );
    }
    else
    {
        cout<<"device do not suppport registration mode"<<endl;
    }
    // start color stream
    result = oniColorStream.start();
    if (result != STATUS_OK)
    {
        printf("Couldn't start the depth stream\n%s\n", OpenNI::getExtendedError());
        exit(0);
    }
}

Status getVideoFrames()
{
    Mat cvDepthImg;
    Mat cvBGRImg;
    Mat cvFusionImg;
    Status result = STATUS_OK;
    // VideoFrameRef oniDepthImg;
    // VideoFrameRef oniColorImg;
    int depthHeight = 0, depthWidth = 0;
    int colorHeight = 0, colorWidth = 0;

    namedWindow("depth", WINDOW_AUTOSIZE);
    namedWindow("image", WINDOW_AUTOSIZE);

    // read frame
    result = oniColorStream.readFrame( &oniColorImg );
    if( result == STATUS_OK )
    {
        // convert data into OpenCV type
        colorHeight = oniColorImg.getHeight();
        colorWidth = oniColorImg.getWidth();
        // cout<<"colorHeight = "<<colorHeight<<endl;
        // cout<<"colorWidth = "<<colorWidth<<endl;
        Mat cvRGBImg( oniColorImg.getHeight(), oniColorImg.getWidth(), CV_8UC3, (void*)oniColorImg.getData() );
        cvtColor( cvRGBImg, cvBGRImg, CV_RGB2BGR );
        // imshow( "image", cvBGRImg );
        if(USE_SENSOR_KINECT2)
        {
            Mat colorImgResize;
            cv::resize(cvBGRImg, cvBGRImg, Size(512, 424));
            imshow("image", cvBGRImg);
        }
        else if(USE_SENSOR_XTION)
        {
            imshow("image", cvBGRImg);
        }
        
        // Mat grayImgSrc, grayImgDst;
        // cvtColor(cvBGRImg, grayImgSrc, COLOR_BGR2GRAY);
        // imshow("gray img src", grayImgSrc);
        // grayImgSrc.convertTo(grayImgDst, CV_8UC1);
        // imshow("gray img dst", grayImgDst);
    }
    else
    {
        cout<<"color frame read failed!!!"<<endl;
    }

    result = oniDepthStream.readFrame( &oniDepthImg );
    if( result == STATUS_OK )
    {
        depthHeight = oniDepthImg.getHeight();
        depthWidth = oniDepthImg.getWidth();
        // cout<<"depthHeight = "<<depthHeight<<endl;
        // cout<<"depthWidth = "<<depthWidth<<endl;
        Mat cvRawImg16U( oniDepthImg.getHeight(), oniDepthImg.getWidth(), CV_16UC1, (void*)oniDepthImg.getData() );
        cvRawImg16U.convertTo( cvDepthImg, CV_8U, 255.0/(oniDepthStream.getMaxPixelValue()));
        imshow( "depth", cvDepthImg );
    }
    else
    {
        cout<<"depth frame read failed!!!"<<endl;
    }

    return result;
}

//openni图像流转化成点云
bool getCloudXYZCoordinate(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_XYZRGB, int mode)  {
    float fx,fy,fz;
    int i=0;
    //以米为单位
    double fScale = 0.001;
    openni::RGB888Pixel *pColor = (openni::RGB888Pixel*)oniColorImg.getData();

    openni::DepthPixel *pDepthArray = (openni::DepthPixel*)oniDepthImg.getData();
    for(int y = 0; y < oniDepthImg.getHeight(); y++) {
        for(int x = 0; x < oniDepthImg.getWidth(); x++) {
            int idx = x + y*oniDepthImg.getWidth();
            const openni::DepthPixel rDepth = pDepthArray[idx];
            openni::CoordinateConverter::convertDepthToWorld(oniDepthStream,x,y,rDepth,&fx,&fy,&fz);
            if(mode == USE_PCL_VIEWER)
            {
                // fx = -fx;
                fy = -fy;
            }
            else if(mode == USE_CLOUD_VIEWER)
            {
                // fx = -fx;
                fz = -fz;
            }
            
            cloud_XYZRGB->points[i].x = fx * fScale;
            cloud_XYZRGB->points[i].y = fy * fScale;
            cloud_XYZRGB->points[i].z = fz * fScale;
            cloud_XYZRGB->points[i].r = pColor[i].r;
            cloud_XYZRGB->points[i].g = pColor[i].g;
            cloud_XYZRGB->points[i].b = pColor[i].b;
            i++;
        }
    }
    return true;
}

void displayCloudAndImg()
{
    Status result = STATUS_OK;
    char key = 0;
        
    if(result == STATUS_OK)
    {
        //创建pcl云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_XYZRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
        if(USE_SENSOR_XTION)
        {
            cloud_XYZRGB->width = 640;
            cloud_XYZRGB->height = 480;
        }
        else if(USE_SENSOR_KINECT2)
        {
            cloud_XYZRGB->width = 512;
            cloud_XYZRGB->height = 424;
        }

        
        cloud_XYZRGB->points.resize(cloud_XYZRGB->width*cloud_XYZRGB->height);
        //pcl可视化
        int visualMode = USE_PCL_VIEWER;
        if(visualMode == USE_PCL_VIEWER)
        {
            pcl::visualization::PCLVisualizer::Ptr m_pViewer(new pcl::visualization::PCLVisualizer("Viewer"));
            m_pViewer->setCameraPosition(0, 0, -2, 0,-1, 0, 0);
            m_pViewer->addCoordinateSystem(0.3);
            while(!m_pViewer->wasStopped()) 
            {
                getVideoFrames();
                getCloudXYZCoordinate(cloud_XYZRGB, visualMode);
                m_pViewer->addPointCloud<pcl::PointXYZRGB>(cloud_XYZRGB,"cloud");
                m_pViewer->spinOnce();
                m_pViewer->removeAllPointClouds();
                key = cv::waitKey(1);
            }
        }
        else if(visualMode == USE_CLOUD_VIEWER)
        {
            pcl::visualization::CloudViewer m_pViewer("Viewer");

            while(!m_pViewer.wasStopped()) 
            {
                getVideoFrames();
                getCloudXYZCoordinate(cloud_XYZRGB, visualMode);
                m_pViewer.showCloud(cloud_XYZRGB);
                key = cv::waitKey(1);
            }
        }
        
        
    }
    
}

int main( int argc, char** argv )
{
    Status result = STATUS_OK;

    //【1】
    // initialize OpenNI2
    initOpenNI();
    // open device
    initDevice();

    //【2】
    // create depth stream
    initDepthStream();
    //【3】
    // create color stream
    initColorStream();
    //【4】
    //Display cloud depth and rgb image
    displayCloudAndImg();

    return 0;
}


// #include <iostream> 
// #include <OpenNI.h>
// #include <pcl/common/common_headers.h>       // for pcl::PointCloud
// #include <pcl/visualization/pcl_visualizer.h>

// openni::Device mDevice;
// openni::VideoStream mColorStream;
// openni::VideoStream mDepthStream;

// #define USE_SENSOR_XTION    0
// #define USE_SENSOR_KINECT2  1

// bool init(){
//     // Initial OpenNI
//     if(openni::OpenNI::initialize() != openni::STATUS_OK){
//         std::cerr << "OpenNI Initial Error: "  << openni::OpenNI::getExtendedError() << std::endl;
//         return false;
//     }
//     // Open Device
//     if(mDevice.open(openni::ANY_DEVICE) != openni::STATUS_OK) {
//         std::cerr << "Can't Open Device: "  << openni::OpenNI::getExtendedError() << std::endl;
//         return false;
//     }
//     return true;
// }

// bool createColorStream() {
//     if(mDevice.hasSensor(openni::SENSOR_COLOR)) {
//         if(mColorStream.create(mDevice, openni::SENSOR_COLOR) == openni::STATUS_OK) {
//             // set video mode
//             openni::VideoMode mMode;
//             if(USE_SENSOR_XTION)
//                 mMode.setResolution(640, 480);
//             else if(USE_SENSOR_KINECT2)
//                 mMode.setResolution(512, 424);
//             mMode.setFps(30);
//             mMode.setPixelFormat( openni::PIXEL_FORMAT_RGB888 );

//             if(mColorStream.setVideoMode(mMode) != openni::STATUS_OK) {
//                 std::cout << "Can't apply VideoMode: "  << openni::OpenNI::getExtendedError() << std::endl;
//                 return false;
//             }
//         } else {
//             std::cerr << "Can't create color stream on device: " << openni::OpenNI::getExtendedError() << std::endl;
//             return false;
//         }

//         // start color stream
//         mColorStream.start();
//         return true;
//     }
//     return false;
// }

// bool createDepthStream(){
//     if(mDevice.hasSensor(openni::SENSOR_DEPTH)) {
//         if(mDepthStream.create(mDevice, openni::SENSOR_DEPTH) == openni::STATUS_OK) {
//             // set video mode
//             openni::VideoMode mMode;
//             if(USE_SENSOR_XTION)
//                 mMode.setResolution(640, 480);
//             else if(USE_SENSOR_KINECT2)
//                 mMode.setResolution(512, 424);
//             mMode.setFps(30);
//             mMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

//             if(mDepthStream.setVideoMode(mMode) != openni::STATUS_OK) {
//                 std::cout << "Can't apply VideoMode to depth stream: " << openni::OpenNI::getExtendedError() << std::endl;
//                 return false;
//             }
//         } else {
//             std::cerr << "Can't create depth stream on device: " << openni::OpenNI::getExtendedError() << std::endl;
//             return false;
//         }
//         // start depth stream
//         mDepthStream.start();
//         // image registration
//         // if( mDevice.isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR) )
//         //         // mDevice.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
//         // else
//         //     std::cerr << "Don't support registration" << std::endl;
//         return true;
//     } else {
//         std::cerr << "ERROR: This device does not have depth sensor" << std::endl;
//         return false;
//     }
// }

// //openni图像流转化成点云
// bool getCloudXYZCoordinate(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_XYZRGB)  {
//     openni::VideoFrameRef  colorFrame;
//     mColorStream.readFrame(&colorFrame);
//     openni::RGB888Pixel *pColor = (openni::RGB888Pixel*)colorFrame.getData();

//     openni::VideoFrameRef  mDepthFrame;
//     if(mDepthStream.readFrame(&mDepthFrame) == openni::STATUS_OK) {
//         float fx,fy,fz;
//         int i=0;
//         //以米为单位
//         double fScale = 0.001;
//         openni::DepthPixel *pDepthArray = (openni::DepthPixel*)mDepthFrame.getData();
//         for(int y = 0; y < mDepthFrame.getHeight(); y++) {
//             for(int x = 0; x < mDepthFrame.getWidth(); x++) {
//                 int idx = x + y*mDepthFrame.getWidth();
//                 const openni::DepthPixel rDepth = pDepthArray[idx];
//                 openni::CoordinateConverter::convertDepthToWorld(mDepthStream,x,y,rDepth,&fx,&fy,&fz);
//                 fx = -fx;
//                 fy = -fy;
//                 cloud_XYZRGB->points[i].x = fx * fScale;
//                 cloud_XYZRGB->points[i].y = fy * fScale;
//                 cloud_XYZRGB->points[i].z = fz * fScale;
//                 cloud_XYZRGB->points[i].r = pColor[i].r;
//                 cloud_XYZRGB->points[i].g = pColor[i].g;
//                 cloud_XYZRGB->points[i].b = pColor[i].b;
//                 i++;
//             }
//         }
//         return true;
//     } else {
//         std::cout << "getCloudXYZCoordinate: fail to read frame from depth stream" << std::endl;
//         return false;
//     }
// }

// int main(){ 
//     //openni初始化、打开摄像头
//     if(!init()) {
//         std::cout << "Fail to init ..." << std::endl;
//         return -1;
//     }
//     //openni创建图像流
//     if(createColorStream() && createDepthStream())
//         std::cout << "displayPointCloud: create color stream and depth stream ..." << std::endl;
//     else{
//         std::cout << "displayPointCloud: can not create color stream and depth stream ..." << std::endl;
//         return -1;
//     }
//     //创建pcl云
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_XYZRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
//     if(USE_SENSOR_XTION)
//     {
//         cloud_XYZRGB->width = 640;
//         cloud_XYZRGB->height = 480;
//     }
//     else if(USE_SENSOR_KINECT2)
//     {
//         cloud_XYZRGB->width = 512;
//         cloud_XYZRGB->height = 424;
//     }
    
//     cloud_XYZRGB->points.resize(cloud_XYZRGB->width*cloud_XYZRGB->height);
//     //pcl可视化
//     pcl::visualization::PCLVisualizer::Ptr m_pViewer(new pcl::visualization::PCLVisualizer("Viewer"));
//     m_pViewer->setCameraPosition(0, 0, -2, 0,-1, 0, 0);
//     m_pViewer->addCoordinateSystem(0.3);
//     while(!m_pViewer->wasStopped()) {
//         getCloudXYZCoordinate(cloud_XYZRGB);
//         m_pViewer->addPointCloud<pcl::PointXYZRGB>(cloud_XYZRGB,"cloud");
//         m_pViewer->spinOnce();
//         m_pViewer->removeAllPointClouds();
//     }
//     mColorStream.destroy();
//     mDepthStream.destroy();
//     mDevice.close();
//     openni::OpenNI::shutdown();

//     return 0;
// }
