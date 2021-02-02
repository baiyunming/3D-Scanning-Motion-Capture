#include "Eigen.h"
#include <iostream>
#include <fstream>

class PointCloud {
public:

    // 2D keypoints in the image
    std::vector<cv::KeyPoint> points2d;


    // matching between current frame and next frame
    std::map<int, int> matching_with_next_frame;

    // matching between 2d keypoints and 3d points
    std::map<int, int> keypoints_global_3D_correspondence;

    // Matrices
    Matrix4f cameraExtrinsics;
    Matrix3f cameraIntrinsics;


    // Constructor
    PointCloud(const Matrix3f& intrinsicMatrix, const Matrix4f& extrinsicMatrix) {
        cameraIntrinsics = intrinsicMatrix;
        cameraExtrinsics = extrinsicMatrix;
    }

    //Getters and setters
    
    Matrix4f getCameraExtrinsics() const {
        return cameraExtrinsics;
    }

    void setCameraExtrinsics(const Matrix4f& extrinsicMatrix) {
        cameraExtrinsics = extrinsicMatrix;
    }

    Matrix3f getCameraIntrinsics() const {
        return cameraIntrinsics;
    }

    void setCameraIntrinsics(const Matrix3f& intrinsicMatrix) {
        cameraIntrinsics = intrinsicMatrix;
    }


    void setIndexMatches_Current_Next_Frames(const std::map<int, int>& matches) {
        matching_with_next_frame = matches;
    }

    std::map<int, int> getIndexMatches_Current_Next_Frames() {
        return matching_with_next_frame;
    }

    std::vector<cv::KeyPoint> getPoints2d() {
        return points2d;
    }

    void setPoints2d(const std::vector<cv::KeyPoint> points2dim) {
        points2d = points2dim;
    }

    std::map<int, int> get2d_3d_correspondence() {
        return keypoints_global_3D_correspondence;
    }

    void append2d_3d_correspondence(const int key, const int value) {
        keypoints_global_3D_correspondence[key] = value;
    }


};