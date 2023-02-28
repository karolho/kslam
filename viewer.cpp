#include "viewer.hpp"
#include "frame.hpp"
#include <opencv2/opencv.hpp>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/scene/scenehandler.h>
#include <pangolin/gl/gldraw.h>

using namespace std;
using namespace cv;

extern Mat dbg_frame;

extern vector<Frame> frames;

void runViewer()
{
    namedWindow("kslam", 1);

    pangolin::CreateWindowAndBind("Main", 1280, 960);
    glEnable(GL_DEPTH_TEST);
    
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1280, 960, 420, 420, 640, 480, 0.2, 100),
        pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1280.0f/960.0f)
            .SetHandler(&handler);
    
    d_cam.SetDrawFunction([&](pangolin::View& view){
        view.Activate(s_cam);
    }); 

    while (1 && !pangolin::ShouldQuit()) {
        imshow("kslam", dbg_frame);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0, 0.0, 0.0, 1.0);

        glColor3f(0.0, 255.0, 0.0);

        glPushMatrix();

        for (int i = 0; i < frames.size(); i++) {
            Mat pose = frames[i].getPose();
            
            double x_ang = atan2(pose.at<double>(2, 1), pose.at<double>(2, 2));
            double y_ang = atan2(-pose.at<double>(2, 0), sqrt(pow(2, pose.at<double>(2, 1)) + pow(2, pose.at<double>(2, 2))));
            double z_ang = atan2(pose.at<double>(1, 0), pose.at<double>(0, 0));

            glRotated(x_ang, 1, 0, 0);
            glRotated(y_ang, 0, 1, 0);
            glRotated(z_ang, 0, 0, 1);

            glTranslated(pose.at<double>(0, 3), pose.at<double>(1, 3), pose.at<double>(2, 3));

            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(1, 0.75, 0.6);
            glVertex3f(0, 0, 0);
            glVertex3f(1, -0.75, 0.6);
            glVertex3f(0, 0, 0);
            glVertex3f(-1, -0.75, 0.6);
            glVertex3f(0, 0, 0);
            glVertex3f(-1, 0.75, 0.6);

            glVertex3f(1, 0.75, 0.6);
            glVertex3f(1, -0.75, 0.6);

            glVertex3f(-1, 0.75, 0.6);
            glVertex3f(-1, -0.75, 0.6);

            glVertex3f(-1, 0.75, 0.6);
            glVertex3f(1, 0.75, 0.6);

            glVertex3f(-1, -0.75, 0.6);
            glVertex3f(1, -0.75, 0.6);
            glEnd();

        }
        
        glPopMatrix();

        pangolin::FinishFrame();
        waitKey(1);
    }
}
