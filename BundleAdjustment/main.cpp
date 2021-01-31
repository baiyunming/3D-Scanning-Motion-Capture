#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <VirtualSensor.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <PointCloud.h>
#include <opencv2/calib3d.hpp>
#include <Optimizer.h>
using namespace std;
using namespace cv;


std::string DATA_PATH = "F:/TUM Learning Material/20WS/Informatik/3D Scanning & Motion Capture/bundle_adjustment/src/data/rgbd_dataset_freiburg2_xyz/";
std::string  PROJECT_PATH = "F:/TUM Learning Material/20WS/Informatik/3D Scanning & Motion Capture/bundle_adjustment/src/output/ConsecutiveBA/";
#define SKIP_FRAMES 2
#define NUM_FEATURES_EACH_FRAME 1500
#define NUM_CONSECUTIVE_FRAMES 3

void get_data(std::string file_path, std::vector<cv::Mat>& depth_images, std::vector<cv::Mat>& rgb_images, std::vector<Matrix4f>& transformationMatrices, std::vector<string>& timestamp);
void split(const std::string& s, char delim, std::vector<std::string>& elems);

void find_matches(vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, Mat& descriptor_1, Mat& descriptor_2, vector<DMatch>& good_matches);

int consecutiveFrames(std::vector<PointCloud>& PCs, int currPointCloudIndex, int currKeypointIndex, int currRecursion);

void get_landmark_and_imgpoints(std::vector<PointCloud>& PCs, int currPointCloudIndex, vector<Vector3f>& global_3D_points, std::vector<std::vector<cv::KeyPoint>>& keypointsAllImgs, vector<Point3f>& landmarks_3d, vector<Point2f>& keypoint_2d);

void estimation_trajectory(vector<Point3f>& object_points, vector<Point2f>& image_points, Eigen::Matrix3f& Intrinsic, Eigen::Matrix3f& Rotation_matrix, Eigen::Vector3f& Tranlation_vector, Eigen::Matrix4f& estimated_trajectory);

void generateOffFile(std::string filename, std::vector<Vector3f> points3d, std::vector<cv::Vec3b> colorValues, vector< Eigen::Matrix4f> all_estimated_trajectory, vector< Eigen::Matrix4f> groundtruth_trajectory);

void generateCompareOffFile(std::string filename, std::vector<Vector3f> points3d, std::vector<cv::Vec3b> colorvalues, vector< Eigen::Matrix4f> initialized_trajectory, vector< Eigen::Matrix4f> optimized_trajectory, vector< Eigen::Matrix4f> groundtruth_trajectory);

void generateTxTFile(std::string filename1, vector< Eigen::Matrix4f> all_estimated_trajectory, vector<string> timestamp);

int main(int argc, char** argv) {
    
    std::vector<cv::Mat> depthImages, rgbImages;
    std::vector<Matrix4f> ground_truth_transformationMatrices;
    std::vector<string> timestamp;
    // getting data
    get_data("final_mapping_rgb_depth_trajectory.txt", depthImages, rgbImages, ground_truth_transformationMatrices, timestamp);

    Matrix3f intrinsicMatrix;
    intrinsicMatrix << 517.3f, 0.0f, 318.6f,
        0.0f, 516.5f, 255.3f,
        0.0f, 0.0f, 1.0f;
    
    std::vector<std::vector<cv::KeyPoint>> keypointsAllImgs; //dimension: 158 frames, each frame 1000 keypoints
    std::vector<cv::Mat> descriptorAllImgs; //dimension: 158 frames
    std::vector<std::vector<cv::DMatch>> allMatches;
    std::vector<PointCloud> pointClouds; // dimension: 158 frames

    // Initialize all the frames
    for (int i = 0; i < rgbImages.size(); i++) {

        Matrix4f extrinsicMatrix;

        // We take ground truth poses from the first frame only
        if (i == 0)
            extrinsicMatrix = ground_truth_transformationMatrices[i];
        else
            extrinsicMatrix = Matrix4f::Identity();

        // Create a new point cloud of the current frame
        PointCloud pointCloud = PointCloud(intrinsicMatrix, extrinsicMatrix);
        pointClouds.push_back(pointCloud);
    }

    
        // extract features and descriptors from all frames 
     for (int i = 0; i < depthImages.size(); i++) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            
    
            cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(NUM_FEATURES_EACH_FRAME);
            Ptr<DescriptorExtractor> descriptor = ORB::create();
    
            detector->detect(rgbImages[i], keypoints);
            descriptor->compute(rgbImages[i], keypoints, descriptors);
            
            pointClouds[i].setPoints2d(keypoints);
    
            keypointsAllImgs.push_back(keypoints);
            descriptorAllImgs.push_back(descriptors);
     }
    
    
        // find matches between each two consecutive frames 
     for (int i = 0; i < depthImages.size() - 1; i++) {
            vector<DMatch> matches;
    
            find_matches(keypointsAllImgs[i], keypointsAllImgs[i+1], descriptorAllImgs[i], descriptorAllImgs[i+1], matches);
            allMatches.push_back(matches);
     }
    

     //first and thrid frame ensure consistent feature mapping
     //take the depth value into account 0.1m ---> 1.5m   
     for (int i = 0; i < depthImages.size() - 2 ; i++) {
            vector<DMatch> first_thrid_matches;
            find_matches(keypointsAllImgs[i], keypointsAllImgs[i + 2], descriptorAllImgs[i], descriptorAllImgs[i + 2], first_thrid_matches);
    
            std::map <int, int> match_current_next_frame;
            std::map <int, int> match_current_last_frame;
            
            for (int j = 0; j < allMatches[i].size(); j++) {
                int first_frame_train = allMatches[i][j].trainIdx;
    
                int index = -1;
                for (int k = 0; k < allMatches[i + 1].size(); k++) {
                    if (allMatches[i + 1][k].queryIdx == first_frame_train) {
                        index = k;
                        break;
                    }
                    else continue;
                }
    
                if (index == -1) {
                    continue;
                }
    
                int index_first_thrid = -1;
                for (int k = 0; k < first_thrid_matches.size(); k++) {
                    if (allMatches[i][j].queryIdx == first_thrid_matches[k].queryIdx) {
                        index_first_thrid = k;
                        break;
                    }
                    else continue;
                }
    
                if (index_first_thrid == -1) {
                    continue;
                }
    
             //check consistency of matching result 
            if (allMatches[i + 1][index].trainIdx == first_thrid_matches[index_first_thrid].trainIdx) {
                 // check depth information of 2d points (exclude close and far keypoints)
                 cv::Point2f point2d = keypointsAllImgs[i][allMatches[i][j].queryIdx].pt;
                 cv::Point2f point2d_next_frame = keypointsAllImgs[i + 1][allMatches[i][j].trainIdx].pt;
 
                 float depth1 = depthImages[i].at<uint16_t>(point2d);
                 float depth2 = depthImages[i + 1].at<uint16_t>(point2d_next_frame);
  
                 if (depth1 > 500 && depth2 > 500 && depth1 < 15000 && depth2 < 15000 && abs(depth1 - depth2) < 500) {
                      match_current_next_frame[allMatches[i][j].queryIdx] = allMatches[i][j].trainIdx;
                      match_current_last_frame[allMatches[i][j].trainIdx] = allMatches[i][j].queryIdx;
                  }
               }
            }
    
            pointClouds[i].setIndexMatches_Current_Next_Frames(match_current_next_frame);
           //pointClouds[i + 1].setIndexMatches_Current_Last_Frames(match_current_last_frame);
           //cout << "current frame" << i << "find consistent feature mapping" ;
    }
    
    
     //find strong feature points (remain observed in Consecutive frames)
     //assign each strong feature point with 3d_Point_idx
     //initial 3d_Point_idx using the depth_information from the first frame (ground_truth)
     vector<Vector3f> global_3D_points;
     vector<Vec3b> global_color_points;
        
     for (int i = 0; i < rgbImages.size()-2; i++) {
          std::map <int, int> current_next_match = pointClouds[i].getIndexMatches_Current_Next_Frames();
          std::map <int, int> current_frame_2d3d_correspondence = pointClouds[i].get2d_3d_correspondence();
    
          for (auto& match : current_next_match) {
              int current_frame_idx = match.first;
              int next_frame_idx = match.second;
    
              int consecutive_frames = consecutiveFrames(pointClouds, i, current_frame_idx, 1);
   
              if (consecutive_frames >= NUM_CONSECUTIVE_FRAMES) {
                    //if unregistred keypoint --> assign a new landmark to it
                  if (current_frame_2d3d_correspondence.find(current_frame_idx) == current_frame_2d3d_correspondence.end()) {
                         cv::Point2f keypoint_2D = keypointsAllImgs[i][current_frame_idx].pt;
                         Vector3f Landmark;
                       if (i == 0) {
                              float depth = (depthImages[i].at<uint16_t>(keypoint_2D) * 1.0f) / 5000.0f;
                              Vector3f extended_pixel_coordinate = Vector3f(depth * keypoint_2D.x, depth * keypoint_2D.y, depth);
                              Vector3f camera_coordinate = pointClouds[i].getCameraIntrinsics().inverse() * extended_pixel_coordinate;
                              Vector4f extended_camera_coordinate = Vector4f(camera_coordinate[0], camera_coordinate[1], camera_coordinate[2], 1.0f);
                              Vector4f p_world = pointClouds[i].getCameraExtrinsics().inverse() * extended_camera_coordinate;
                              Landmark = Vector3f(p_world[0], p_world[1], p_world[2]);
                       }
                       else {
                            Landmark = Vector3f(MINF, MINF, MINF);
                       }
   
                  //assign the same keypoint in consecutive frame with the same global_index_3dLandmark
                  for (int j = i; j < i + consecutive_frames; j++) {
                          std::map<int, int> tmp_matches = pointClouds[j].getIndexMatches_Current_Next_Frames();
                          pointClouds[j].append2d_3d_correspondence(current_frame_idx, global_3D_points.size());
                           current_frame_idx = tmp_matches[current_frame_idx];
                  }
    
         
                  global_3D_points.push_back(Landmark);
                  global_color_points.push_back(rgbImages[i].at<cv::Vec3b>(keypoint_2D));
                  }          
              }
           }
     }
        
        for (int i = 1; i < rgbImages.size() - 2; i++) {
    
            vector<Point3f> landmarks_3d;
            vector<Point2f> keypoint_2d;
    
            get_landmark_and_imgpoints(pointClouds, i, global_3D_points, keypointsAllImgs, landmarks_3d, keypoint_2d);
    
            Eigen::Matrix3f Rotation_matrix;
            Eigen::Vector3f Tranlation_vector;
            Eigen::Matrix4f estimated_extrinsic;
    
            //no updtae the landmarks positions 
            if (keypoint_2d.size() < 10) {
                //pointClouds[i].setCameraExtrinsics(ground_truth_transformationMatrices[i]);           
                pointClouds[i].setCameraExtrinsics(pointClouds[i-1].getCameraExtrinsics());
                //pointClouds[i].setCameraExtrinsics(ground_truth_transformationMatrices[i]);
            }
            else {
                estimation_trajectory(landmarks_3d, keypoint_2d, pointClouds[i].getCameraIntrinsics(), Rotation_matrix, Tranlation_vector, estimated_extrinsic);
                pointClouds[i].setCameraExtrinsics(estimated_extrinsic);
            }
    
            //update global 3d landmarks  
            for (auto& match : pointClouds[i].get2d_3d_correspondence()) {
                int keypoint_idx = match.first;
                int landmark_idx = match.second;
                cv::Point2f keypoint_2D = keypointsAllImgs[i][keypoint_idx].pt;
    
                if (global_3D_points[landmark_idx][0] == MINF) {
                    float depth = (depthImages[i].at<uint16_t>(keypoint_2D) * 1.0f) / 5000.0f;
                    Vector3f extended_pixel_coordinate = Vector3f(depth * keypoint_2D.x, depth * keypoint_2D.y, depth);
                    Vector3f camera_coordinate = pointClouds[i].getCameraIntrinsics().inverse() * extended_pixel_coordinate;
                    Vector4f extended_camera_coordinate = Vector4f(camera_coordinate[0], camera_coordinate[1], camera_coordinate[2], 1.0f);
                    Vector4f p_world = pointClouds[i].getCameraExtrinsics().inverse() * extended_camera_coordinate;
                    global_3D_points[landmark_idx] = Vector3f(p_world[0], p_world[1], p_world[2]);
                    global_color_points[landmark_idx] = rgbImages[i].at<cv::Vec3b>(keypoint_2D);
    
                }
            }
    
        }
    
    std::vector<Matrix4f> estimated_extrinsic;
    for (int i = 0; i < rgbImages.size() - 2; i++) {
         estimated_extrinsic.push_back(pointClouds[i].getCameraExtrinsics());
    }

    generateOffFile("initialization_after_PnP.off", global_3D_points, global_color_points, estimated_extrinsic, ground_truth_transformationMatrices);
    generateTxTFile("initialization_after_PnP.txt", estimated_extrinsic, timestamp);


    Optimization optimizer;
    optimizer.setNbOfIterations(10);
    std::cout << "PHASE 2: Optimization" << std::endl;  
    optimizer.solve_Bundle_Adjustment(pointClouds, global_3D_points);

    //consecutivly solve optimization problem
    //for (int i = 0; i < rgbImages.size() - 2 - 4; i++) {
    //    optimizer.solve_Bundle_Adjustment_partially(pointClouds, global_3D_points, i, 4);
    //}
    

    std::vector<Matrix4f> estimated_extrinsic_after_optimization;
    for (int i = 0; i < rgbImages.size() - 2; i++) {
        estimated_extrinsic_after_optimization.push_back(pointClouds[i].getCameraExtrinsics());
    }
    
    generateOffFile("Optimization.off", global_3D_points, global_color_points, estimated_extrinsic_after_optimization, ground_truth_transformationMatrices);
    generateTxTFile("Optimization.txt", estimated_extrinsic_after_optimization, timestamp);
    
    generateCompareOffFile("OptimizationCompare.off", global_3D_points, global_color_points, estimated_extrinsic, estimated_extrinsic_after_optimization, ground_truth_transformationMatrices);
}



int consecutiveFrames(std::vector<PointCloud>& PCs, int currPointCloudIndex, int currKeypointIndex, int currRecursion) {

    std::map<int, int> matchesFrames = PCs[currPointCloudIndex].getIndexMatches_Current_Next_Frames();

    //can't find the feature  
    if (matchesFrames.find(currKeypointIndex) == matchesFrames.end())
        return currRecursion;

    else
        return consecutiveFrames(PCs, currPointCloudIndex + 1, matchesFrames[currKeypointIndex], currRecursion + 1);
}

void split(const std::string& s, char delim, std::vector<std::string>& elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}


void get_data(std::string file_path, std::vector<cv::Mat>& depth_images, std::vector<cv::Mat>& rgb_images, std::vector<Matrix4f>& transformationMatrices, std::vector<string>&timestamp) {

    std::ifstream inFile;
    std::string fileName;

    inFile.open(DATA_PATH + file_path);

    if (!inFile) {
        std::cerr << "Unable to open file.\n" << std::endl;
        exit(1);
    }

    int i = 0;
    while (std::getline(inFile, fileName)) {
        i++;
        if (i % SKIP_FRAMES != 0)
            continue;

        std::vector<std::string> current_line;
        split(fileName, ' ', current_line);

        cv::Mat depthImg = cv::imread(DATA_PATH + current_line[3], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat rgbImg = cv::imread(DATA_PATH + current_line[1], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

        if (!depthImg.data || !rgbImg.data) {
            std::cout << "Image not found.\n" << std::endl;
            exit(1);
        }

        depth_images.push_back(depthImg);
        rgb_images.push_back(rgbImg);

        // Save poses
        //Matrix4f transformationMatrix = getExtrinsicsFromQuaternion(current_line);
        Matrix4f transformationMatrix;
        Vector3f tranlation_vector = Vector3f(std::stof(current_line[5]), std::stof(current_line[6]), std::stof(current_line[7]));
        Eigen::Quaterniond rot(Eigen::Vector4d(std::stod(current_line[8]), std::stod(current_line[9]), std::stod(current_line[10]), std::stod(current_line[11])));
        transformationMatrix.setIdentity();
        Matrix3d rotation = rot.toRotationMatrix();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                transformationMatrix(i, j) = rotation(i, j);
        }

        transformationMatrix.block<3, 1>(0, 3) = tranlation_vector;

        transformationMatrices.push_back(transformationMatrix.inverse());
        timestamp.push_back(current_line[4]);
    }

    inFile.close();
}


void find_matches(
    vector<KeyPoint>& keypoints_1,
    vector<KeyPoint>& keypoints_2,
    cv::Mat& descriptor_1,
    cv::Mat& descriptor_2,
    vector<DMatch>& good_matches) 
{
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    vector<DMatch> matches;
    matcher->match(descriptor_1, descriptor_2, matches);

    auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- max dist : %f \n", max_dist);
    printf("-- min dist : %f \n", min_dist);

    //std::vector<DMatch> good_matches;
    //good_matches.clear();
    for (int i = 0; i < descriptor_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 20.0)) {
            good_matches.push_back(matches[i]);
        }
    }



}


void get_landmark_and_imgpoints(std::vector<PointCloud>& pcs, int currpointcloudindex, vector<Vector3f>& global_3d_points, std::vector<std::vector<cv::KeyPoint>>& keypointsallimgs,vector<Point3f>& landmarks_3d, vector<Point2f>& keypoint_2d)
{
    std::map<int, int> temp_matches = pcs[currpointcloudindex].get2d_3d_correspondence();

    for (auto& match : temp_matches) {
        int idx_keypoint = match.first;
        int idx_landmark = match.second;

        if (global_3d_points[idx_landmark][0] != MINF) {
            Point3f temp_landmark = Point3f(global_3d_points[idx_landmark][0], global_3d_points[idx_landmark][1], global_3d_points[idx_landmark][2]);
            landmarks_3d.push_back(temp_landmark);
            keypoint_2d.push_back(keypointsallimgs[currpointcloudindex][idx_keypoint].pt);
        }
    }
}


void estimation_trajectory(vector<Point3f>& object_points, vector<Point2f>& image_points, Eigen::Matrix3f& intrinsic, Eigen::Matrix3f& rotation_matrix, Eigen::Vector3f & tranlation_vector, Eigen::Matrix4f& estimated_trajectory)
{   
    cv::Mat camera_matrix;
    cv:eigen2cv(intrinsic, camera_matrix);
    cv::Mat r;
    cv::Mat t;
    cout << object_points.size() << endl;
    solvePnPRansac(object_points, image_points, camera_matrix, noArray(), r, t);
    
    Rodrigues(r, r);
    
    //convert cv::mat to eigen 
    
    cv2eigen(r, rotation_matrix);
    cv2eigen(t, tranlation_vector);

    estimated_trajectory.setIdentity();
    estimated_trajectory.block<3, 3>(0, 0) = rotation_matrix;
    estimated_trajectory.block<3, 1>(0, 3) = tranlation_vector;  
}




 //method to generate off file for 3d points
void generateOffFile(std::string filename, std::vector<Vector3f> points3d, std::vector<cv::Vec3b> colorvalues, vector< Eigen::Matrix4f> all_estimated_trajectory, vector< Eigen::Matrix4f> groundtruth_trajectory) {
    std::ofstream outfile(PROJECT_PATH + filename);
    if (!outfile.is_open()) return;

     //write header
    outfile << "COFF" << std::endl;
    outfile << 2*all_estimated_trajectory.size() << " 0 0" << std::endl;

     //save camera position
    for (int i = 0; i < all_estimated_trajectory.size(); i++) {
        Matrix4f cameraextrinsics = all_estimated_trajectory[i];
        Matrix3f rotation = cameraextrinsics.block(0, 0, 3, 3);
        Vector3f translation = cameraextrinsics.block(0, 3, 3, 1);
        Vector3f cameraposition = -rotation.transpose() * translation;
        outfile << cameraposition[0] << " " << cameraposition[1] << " " << cameraposition[2] << " 255 0 0" << std::endl;

        Matrix4f gt_extrinsics = groundtruth_trajectory[i];
        Matrix3f gt_rotation = gt_extrinsics.block(0, 0, 3, 3);
        Vector3f gt_translation = gt_extrinsics.block(0, 3, 3, 1);
        Vector3f gt_cameraposition = -gt_rotation.transpose() * gt_translation;
        outfile << gt_cameraposition[0] << " " << gt_cameraposition[1] << " " << gt_cameraposition[2] << " 0 0 255" << std::endl;    
    }

    // save vertices
    //for (int i = 0; i < points3d.size(); i++) {
    //    if (points3d[i][0] == MINF)
    //        outfile << "0 0 0 0 0 0" << std::endl;
    //     
    //else
    //     outfile << points3d[i][0] << " " << points3d[i][1] << " " << points3d[i][2] << " " <<
    //     static_cast<unsigned>(colorvalues[i][2]) << " " << static_cast<unsigned>(colorvalues[i][1]) << " " << static_cast<unsigned>(colorvalues[i][0]) << std::endl;
    //}

    outfile.close();
}


void generateCompareOffFile(std::string filename, std::vector<Vector3f> points3d, std::vector<cv::Vec3b> colorvalues, vector< Eigen::Matrix4f> initialized_trajectory, vector< Eigen::Matrix4f> optimized_trajectory, vector< Eigen::Matrix4f> groundtruth_trajectory) {
    std::ofstream outfile(PROJECT_PATH + filename);
    if (!outfile.is_open()) return;

    //write header
    outfile << "COFF" << std::endl;
    outfile << 3 * initialized_trajectory.size() << " 0 0" << std::endl;

    //save camera position
    for (int i = 0; i < initialized_trajectory.size(); i++) {
        Matrix4f cameraextrinsics = initialized_trajectory[i];
        Matrix3f rotation = cameraextrinsics.block(0, 0, 3, 3);
        Vector3f translation = cameraextrinsics.block(0, 3, 3, 1);
        Vector3f cameraposition = -rotation.transpose() * translation;
        outfile << cameraposition[0] << " " << cameraposition[1] << " " << cameraposition[2] << " 255 0 0" << std::endl;


        Matrix4f optimized_cameraextrinsics = optimized_trajectory[i];
        Matrix3f optimized_rotation = optimized_cameraextrinsics.block(0, 0, 3, 3);
        Vector3f optimized_translation = optimized_cameraextrinsics.block(0, 3, 3, 1);
        Vector3f optimized_cameraposition = -optimized_rotation.transpose() * optimized_translation;
        outfile << optimized_cameraposition[0] << " " << optimized_cameraposition[1] << " " << optimized_cameraposition[2] << " 0 255 0" << std::endl;


        Matrix4f gt_extrinsics = groundtruth_trajectory[i];
        Matrix3f gt_rotation = gt_extrinsics.block(0, 0, 3, 3);
        Vector3f gt_translation = gt_extrinsics.block(0, 3, 3, 1);
        Vector3f gt_cameraposition = -gt_rotation.transpose() * gt_translation;
        outfile << gt_cameraposition[0] << " " << gt_cameraposition[1] << " " << gt_cameraposition[2] << " 0 0 255" << std::endl;

    }

    outfile.close();
}


void generateTxTFile(std::string filename1, vector< Eigen::Matrix4f> all_estimated_trajectory, vector<string>timestamp) {
    std::ofstream outfile(PROJECT_PATH + filename1);
    if (!outfile.is_open()) return;

    //write header
    outfile << "# estimated trajectory" << std::endl;
    outfile << "# estimated trajectory" << std::endl;
    outfile << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    
    //save ground_truth position
    for (int i = 0; i < all_estimated_trajectory.size(); i++) {
        Matrix4f extrinsics = all_estimated_trajectory[i];
        Matrix3f rotation = extrinsics.block(0, 0, 3, 3);
        Eigen::Quaternionf quaternion1(rotation);
        Vector3f translation = extrinsics.block(0, 3, 3, 1);
        outfile << timestamp[i] <<" " << translation[0] << " " << translation[1] << " " << translation[2] << " " << quaternion1.x() << " " << quaternion1.y() << " " << quaternion1.z() << " " << quaternion1.w() << std::endl;
    }
    outfile.close();

}