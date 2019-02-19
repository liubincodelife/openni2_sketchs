#include <stdlib.h>
#include <iostream>
#include <string>
#include "OpenNI.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "NiTE.h"
#include "NiteSampleUtilities.h"

using namespace std;
using namespace cv;
using namespace openni;

#define USE_SENSOR_XTION    0
#define USE_SENSOR_KINECT2  1

Device device;
VideoStream oniDepthStream;
VideoStream oniColorStream;
VideoFrameRef oniDepthImg;
VideoFrameRef oniColorImg;
nite::UserTracker* g_pUserTracker;
nite::UserTrackerFrameRef userTrackerFrame;

bool g_drawSkeleton = true;
bool g_drawBackground = true;
bool g_drawBoundingBox = false;

Mat g_depthRgbImg;


void initOpenNI()
{
    openni::Status result = STATUS_OK;
    result = openni::OpenNI::initialize();
    if (result != STATUS_OK)
    {
        printf("Initialize OpenNI failed\n%s\n", OpenNI::getExtendedError());
        exit(0);
    }
}

void initDevice()
{
    openni::Status result = STATUS_OK;
    // open device
    result = device.open(ANY_DEVICE);
    if (result != STATUS_OK)
    {
        printf("Couldn't open device\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }
}

nite::Status initNite()
{
	nite::Status rc = nite::NiTE::initialize();
    if (rc != nite::STATUS_OK)
	{
		cout<<"Failed to initialize Nite..."<<endl;
		return rc;
	}
    else
    {
        cout<<"Initialize Nite success..."<<endl;
    }
    g_pUserTracker = new nite::UserTracker();
	if (g_pUserTracker->create(&device) != nite::STATUS_OK)
	{
        cout<<"userTracker create failed..."<<endl;
		return nite::STATUS_ERROR;
	}
    else
    {
        cout<<"userTracker create success..."<<endl;
    }

    return nite::STATUS_OK;
}

void getSkeltonPoint(Mat& depthimage, const nite::UserData& userinfo)
{
    const nite::Skeleton& userSkelton = userinfo.getSkeleton();
    if(userSkelton.getState() == nite::SKELETON_TRACKED)
    {
        // const nite::SkeletonJoint& userJointRightHand = userSkelton.getJoint(nite::JOINT_RIGHT_HAND);
        // const nite::Point3f& position = userJointRightHand.getPosition();
        // cout<<"righthand position:"<<"x: "<<position.x<<" y: "<<position.y<<" z: "<<position.z<<endl;
        // 得到所有15个骨骼点坐标
        for(int i = 0; i < 15; i++)
        {
            // 得到骨骼坐标
            const nite::SkeletonJoint& skeletonJoint = userSkelton.getJoint((nite::JointType)i);
            const nite::Point3f& position = skeletonJoint.getPosition();

            float depth_x, depth_y;

            // 将骨骼点坐标映射到深度坐标中
            g_pUserTracker->convertJointCoordinatesToDepth(position.x, position.y, position.z, &depth_x, &depth_y);

            cv::Point point((int)depth_x, (int)depth_y);

            // 将获取的深度图像中相对应的坐标点重新赋值为255.即在深度图像中显示出各个骨骼点。
            depthimage.at<uchar>(point) = 255;
        }
        imshow( "depth", depthimage );
    }
    else if(userSkelton.getState() == nite::SKELETON_CALIBRATING)
    {
        cout<<"skelton calibrating"<<endl;
    }
}

void MyLine( Mat img, Point start, Point end )
{
  int thickness = 1;
  int lineType = LINE_AA;
  line( img,
    start,
    end,
    Scalar( 255, 255, 255 ),
    thickness,
    lineType );
}

void MyFilledCircle( Mat img, Point center )
{
  circle( img,
      center,
      3,
      Scalar( 255, 255, 255 ),
      FILLED,
      LINE_8 );
}

void drawStick(Mat& srcDepthImg, const nite::SkeletonJoint& joint1, const nite::SkeletonJoint& joint2)
{
    if(joint1.getPositionConfidence() < 0.5f || joint2.getPositionConfidence() < 0.5f)
    {
        return;
    }
    Vec3b pixelColor;
    pixelColor[0] = 255;
    pixelColor[1] = 255;
    pixelColor[2] = 255;
    const nite::Point3f& position1 = joint1.getPosition();
    const nite::Point3f& position2 = joint2.getPosition();

    float depth_x, depth_y;
    // 将骨骼点坐标映射到深度坐标中
    g_pUserTracker->convertJointCoordinatesToDepth(position1.x, position1.y, position1.z, &depth_x, &depth_y);
    cv::Point point1((int)depth_x, (int)depth_y);
    g_pUserTracker->convertJointCoordinatesToDepth(position2.x, position2.y, position2.z, &depth_x, &depth_y);
    cv::Point point2((int)depth_x, (int)depth_y);
    MyFilledCircle(srcDepthImg, point1);
    MyFilledCircle(srcDepthImg, point2);
    // 将获取的深度图像中相对应的坐标点重新赋值为255.即在深度图像中显示出各个骨骼点。
    srcDepthImg.at<Vec3b>(point1) = pixelColor;
    srcDepthImg.at<Vec3b>(point2) = pixelColor;
    MyLine(srcDepthImg, point1, point2);
    imshow( "depth", srcDepthImg );
}

void drawSkeleton(Mat& srcDepthImg, const nite::UserData& userInfo)
{
    const nite::Skeleton& userSkelton = userInfo.getSkeleton();
    Vec3b pixelColor;
    pixelColor[0] = 255;
    pixelColor[1] = 255;
    pixelColor[2] = 255;

    const nite::SkeletonJoint& skeletonJointHead = userSkelton.getJoint(nite::JOINT_HEAD);
    const nite::SkeletonJoint& skeletonJointNeck = userSkelton.getJoint(nite::JOINT_NECK);
    const nite::SkeletonJoint& skeletonJointShoulderL = userSkelton.getJoint(nite::JOINT_LEFT_SHOULDER);
    const nite::SkeletonJoint& skeletonJointShoulderR = userSkelton.getJoint(nite::JOINT_RIGHT_SHOULDER);
    const nite::SkeletonJoint& skeletonJointElbowL = userSkelton.getJoint(nite::JOINT_LEFT_ELBOW);
    const nite::SkeletonJoint& skeletonJointElbowR = userSkelton.getJoint(nite::JOINT_RIGHT_ELBOW);
    const nite::SkeletonJoint& skeletonJointHandL = userSkelton.getJoint(nite::JOINT_LEFT_HAND);
    const nite::SkeletonJoint& skeletonJointHandR = userSkelton.getJoint(nite::JOINT_RIGHT_HAND);
    const nite::SkeletonJoint& skeletonJointTorso = userSkelton.getJoint(nite::JOINT_TORSO);
    const nite::SkeletonJoint& skeletonJointHipL = userSkelton.getJoint(nite::JOINT_LEFT_HIP);
    const nite::SkeletonJoint& skeletonJointHipR = userSkelton.getJoint(nite::JOINT_RIGHT_HIP);
    const nite::SkeletonJoint& skeletonJointKneeL = userSkelton.getJoint(nite::JOINT_LEFT_KNEE);
    const nite::SkeletonJoint& skeletonJointKneeR = userSkelton.getJoint(nite::JOINT_RIGHT_KNEE);
    const nite::SkeletonJoint& skeletonJointFootL = userSkelton.getJoint(nite::JOINT_LEFT_FOOT);
    const nite::SkeletonJoint& skeletonJointFootR = userSkelton.getJoint(nite::JOINT_RIGHT_FOOT);
    drawStick(srcDepthImg, skeletonJointHead, skeletonJointNeck);
    drawStick(srcDepthImg, skeletonJointShoulderL, skeletonJointElbowL);
    drawStick(srcDepthImg, skeletonJointElbowL, skeletonJointHandL);
    drawStick(srcDepthImg, skeletonJointShoulderR, skeletonJointElbowR);
    drawStick(srcDepthImg, skeletonJointElbowR, skeletonJointHandR);
    drawStick(srcDepthImg, skeletonJointShoulderL, skeletonJointShoulderR);
    drawStick(srcDepthImg, skeletonJointShoulderL, skeletonJointTorso);
    drawStick(srcDepthImg, skeletonJointShoulderR, skeletonJointTorso);
    drawStick(srcDepthImg, skeletonJointTorso, skeletonJointHipL);
    drawStick(srcDepthImg, skeletonJointTorso, skeletonJointHipR);
    drawStick(srcDepthImg, skeletonJointHipL, skeletonJointHipR);
    drawStick(srcDepthImg, skeletonJointHipL, skeletonJointKneeL);
    drawStick(srcDepthImg, skeletonJointHipR, skeletonJointKneeR);
    drawStick(srcDepthImg, skeletonJointKneeL, skeletonJointFootL);
    drawStick(srcDepthImg, skeletonJointKneeR, skeletonJointFootR);
}

void showSkeltonJoint()
{
    // nite::UserTrackerFrameRef userTrackerFrame;
	openni::VideoFrameRef depthFrame;
    Mat depthImg, depthBinImg;
	nite::Status rc = g_pUserTracker->readFrame(&userTrackerFrame);
	if (rc != nite::STATUS_OK)
	{
		printf("GetNextData failed\n");
		return;
	}

	depthFrame = userTrackerFrame.getDepthFrame();
    Mat cvRawDepthImg16U( depthFrame.getHeight(), depthFrame.getWidth(), CV_16UC1, (void*)depthFrame.getData() );
    // convertScaleAbs(cvRawDepthImg16U, depthImg);
    cvRawDepthImg16U.convertTo( depthImg, CV_8U, 255.0/10000);

    imshow( "depth", depthImg );
    // threshold(depthImg, depthBinImg, 50, 255, THRESH_BINARY_INV);
    // imshow( "depth binary", depthBinImg );
    const nite::Array<nite::UserData>& allUsers = userTrackerFrame.getUsers();
    for(int i = 0; i < allUsers.getSize(); i++)
    {
        const nite::UserData& userInfo = allUsers[i];
        if(userInfo.isNew())
        {
            cout<<"New User ["<<userInfo.getId()<<"] found."<<endl;
            g_pUserTracker->startSkeletonTracking(userInfo.getId());
        }
        getSkeltonPoint(depthImg, userInfo);  
    }
}

void display()
{
    openni::VideoFrameRef depthFrame;
    Mat depthImg, depthRgbImg;
	nite::Status status = g_pUserTracker->readFrame(&userTrackerFrame);
	if (status != nite::STATUS_OK)
	{
		printf("get tracker frame failed\n");
		return;
	}

	depthFrame = userTrackerFrame.getDepthFrame();
    Mat cvRawDepthImg16U( depthFrame.getHeight(), depthFrame.getWidth(), CV_16UC1, (void*)depthFrame.getData() );
    // convertScaleAbs(cvRawDepthImg16U, depthImg);
    // imshow("depthImage", depthImg);
    cvRawDepthImg16U.convertTo( depthImg, CV_8U, 255.0/10000);
    cvtColor(depthImg, depthRgbImg, COLOR_GRAY2BGR);
    imshow( "depth", depthRgbImg );
    Mat depthHistEq;
    equalizeHist(depthImg, depthHistEq);
    imshow("depthHistEq", depthHistEq);

    const nite::Array<nite::UserData>& allUsers = userTrackerFrame.getUsers();
    for(int i = 0; i < allUsers.getSize(); i++)
    {
        const nite::UserData& userInfo = allUsers[i];
        if(userInfo.isNew())
        {
            cout<<"New User ["<<userInfo.getId()<<"] found."<<endl;
            g_pUserTracker->startSkeletonTracking(userInfo.getId());
        }
        else if(!userInfo.isLost())
        {
            if(userInfo.getSkeleton().getState() == nite::SKELETON_TRACKED && g_drawSkeleton)
            {
                drawSkeleton(depthRgbImg, userInfo);
            }
            else if(userInfo.getSkeleton().getState() == nite::SKELETON_CALIBRATING)
            {
                cout<<"skeleton calibrating"<<endl;
            }
        }
        
        
    }
}

int main( int argc, char** argv )
{
    openni::Status result = STATUS_OK;

    //【1】
    // initialize OpenNI2
    initOpenNI();
    //【2】open device
    initDevice();
    //【3】init Nite
    initNite();
    
    char key = 0;
    while(key != 27)
    {
        display();
        key = waitKey(1);
        switch(key)
        {
            case 's':
                g_drawSkeleton = !g_drawSkeleton;
                break;
            case 'b':
                g_drawBackground = !g_drawBackground;
                break;
            case 'x':
                g_drawBoundingBox = !g_drawBoundingBox;
                break;
        }
    }
    
    userTrackerFrame.release();
    g_pUserTracker->destroy();
    nite::NiTE::shutdown();

    return 0;
}
