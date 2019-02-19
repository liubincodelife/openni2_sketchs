#include <stdlib.h>
#include <iostream>
#include <string>
#include "OpenNI.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace openni;

Device device;
VideoStream oniDepthStream;
VideoStream oniColorStream;

#define USE_SENSOR_XTION    1
#define USE_SENSOR_KINECT2  0


void CheckOpenNIError( Status result, string status )
{
    if( result != STATUS_OK )
        cerr << status << " Error: " << OpenNI::getExtendedError() << endl;
}

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
        // modeColor.setResolution( 640, 480 );
        modeColor.setResolution( 1280, 1024 );
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
        device.setImageRegistrationMode( IMAGE_REGISTRATION_DEPTH_TO_COLOR );
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

void runLoop()
{
    char key = 0;
    Mat cvDepthImg;
    Mat cvBGRImg;
    Mat cvFusionImg;
    VideoFrameRef oniDepthImg;
    VideoFrameRef oniColorImg;
    int depthHeight = 0, depthWidth = 0;
    int colorHeight = 0, colorWidth = 0;

    namedWindow("depth", WINDOW_AUTOSIZE);
    namedWindow("image", WINDOW_AUTOSIZE);
    // namedWindow("fusion");
    while( key != 27)
    {
        // read frame
        if( oniColorStream.readFrame( &oniColorImg ) == STATUS_OK )
        {
            // convert data into OpenCV type
            colorHeight = oniColorImg.getHeight();
            colorWidth = oniColorImg.getWidth();
            // cout<<"colorHeight = "<<colorHeight<<endl;
            // cout<<"colorWidth = "<<colorWidth<<endl;
            Mat cvRGBImg( oniColorImg.getHeight(), oniColorImg.getWidth(), CV_8UC3, (void*)oniColorImg.getData() );
            cvtColor( cvRGBImg, cvBGRImg, CV_RGB2BGR );
            imshow( "image", cvBGRImg );
        }

        if( oniDepthStream.readFrame( &oniDepthImg ) == STATUS_OK )
        {
            depthHeight = oniDepthImg.getHeight();
            depthWidth = oniDepthImg.getWidth();
            // cout<<"depthHeight = "<<depthHeight<<endl;
            // cout<<"depthWidth = "<<depthWidth<<endl;
            Mat cvRawImg16U( oniDepthImg.getHeight(), oniDepthImg.getWidth(), CV_16UC1, (void*)oniDepthImg.getData() );
            cvRawImg16U.convertTo( cvDepthImg, CV_8U, 255.0/(oniDepthStream.getMaxPixelValue()));
            imshow( "depth", cvDepthImg );
        }
        // convert depth image GRAY to BGR
        // cvtColor(cvDepthImg,cvFusionImg,CV_GRAY2BGR);
        // addWeighted(cvBGRImg,0.7,cvFusionImg,0.3,0,cvFusionImg);
        // imshow( "fusion", cvFusionImg );
        key = cv::waitKey(20);
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
    //Display depth and rgb image
    runLoop();

    return 0;
}

