#include <iostream>
#include <fstream>
#include <array>

#include "Eigen.h"

#include "VirtualSensor.h"



struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;

	// color stored as 4 unsigned char
	Vector4uc color;
};


bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename)
{
	float edgeThreshold = 0.01f; // 1cm

	// TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
	// - have a look at the "off_sample.off" file to see how to store the vertices and triangles
	// - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
	// - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
	// - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
	// - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
	// - only write triangles with valid vertices and an edge length smaller then edgeThreshold

	// TODO: Get number of vertices
	unsigned int nVertices = width * height;


	// TODO: Determine number of valid faces
	unsigned nFaces = 0;
	std::vector<std::string> valid_face; 


	for (unsigned int y = 0; y < height - 1; y = y + 1) {
		for (unsigned int x = 0; x < width - 1; x = x + 1) {
			unsigned int idx1 = y * width + x;
			unsigned int idx2 = y * width + x + 1;
			unsigned int idx3 = (y + 1) * height + x;
			unsigned int idx4 = (y + 1) * height + x + 1;

			bool edge12 = ((vertices[idx1].position - vertices[idx2].position).norm() < edgeThreshold);
			bool edge13 = ((vertices[idx1].position - vertices[idx3].position).norm() < edgeThreshold);
			bool edge23 = ((vertices[idx2].position - vertices[idx3].position).norm() < edgeThreshold);
			bool edge34 = ((vertices[idx3].position - vertices[idx4].position).norm() < edgeThreshold);
			bool edge24 = ((vertices[idx2].position - vertices[idx4].position).norm() < edgeThreshold);


			if (edge12 && edge13 && edge23) {
				nFaces = nFaces + 1;
				valid_face.push_back(std::to_string(3) + " " + std::to_string(idx1) + " " + std::to_string(idx3) + " " + std::to_string(idx2));
			}

			if (edge23 && edge24 && edge34) {
				nFaces = nFaces + 1;
				valid_face.push_back(std::to_string(3) + " " + std::to_string(idx3) + " " + std::to_string(idx4) + " " + std::to_string(idx2));
			}

		}
	}


	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;
	outFile << "# numVertices numFaces numEdges" << std::endl;
	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// TODO: save vertices
	outFile << "# list of vertices" << std::endl;
	outFile << "# X Y Z R G B A" << std::endl;

	for (unsigned int idx = 0; idx < width * height; idx = idx + 1) {
		if (vertices[idx].position[0] == MINF) {
			outFile << 0 << " " << 0 << " " << 0 << " " <<(int)vertices[idx].color[0] << " " << (int)vertices[idx].color[1] << " " << (int)vertices[idx].color[2] << " " << (int)vertices[idx].color[3]<< std::endl;
		}
		else {
			outFile << vertices[idx].position[0] << " " << vertices[idx].position[1] << " " << vertices[idx].position[2] << " " << int(vertices[idx].color[0]) << " " << int(vertices[idx].color[1]) << " " << int(vertices[idx].color[2]) << " " << int(vertices[idx].color[3])<< std::endl;
		}
	}

	outFile << "# list of faces" << std::endl;
	outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;
	// TODO: save valid faces
	for (unsigned int idx = 0; idx < nFaces; idx = idx + 1) {
		outFile << valid_face[idx] << std::endl;
	}

	// close file
	outFile.close();

	return true;
}

int main()
{
	// Make sure this path points to the data folder
	//std::string filenameIn = "F:/TUM Learning Material/20WS/3D Scanning & Motion Capture/exercise_1/exercise_1_bin/rgbd_dataset_freiburg1_xyz/";
	
	//char* buffer;
	//buffer = getcwd(NULL, 0);
	//printf("%s\n", buffer);
	
	std::string filenameIn = "./data/rgbd_dataset_freiburg1_xyz/";
	//std::string filenameIn = "F:/TUM Learning Material/20WS/Informatik/3D Scanning & Motion Capture/exercise_1/exercise_1_src/data/rgbd_dataset_freiburg1_xyz/";
	std::string filenameBaseOut = "mesh_";

	// load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// convert video to meshes
	while (sensor.ProcessNextFrame())
	{
		// get ptr to the current depth frame
		// depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
		float* depthMap = sensor.GetDepth();

		// get ptr to the current color frame
		// color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
		BYTE* colorMap = sensor.GetColorRGBX();

		// get depth intrinsics
		Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
		float fX = depthIntrinsics(0, 0);
		float fY = depthIntrinsics(1, 1);
		float cX = depthIntrinsics(0, 2);
		float cY = depthIntrinsics(1, 2);
		Matrix3f inverse_Intrinsics = depthIntrinsics.inverse();

		// compute inverse depth extrinsics
		Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();

		Matrix4f trajectory = sensor.GetTrajectory();
		Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();

		MatrixXf extendedId = MatrixXf::Identity(4, 3);
		
		

		// TODO 1: back-projection
		// write result to the vertices array below, keep pixel ordering!
		// if the depth value at idx is invalid (MINF) write the following values to the vertices array
		// vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
		// vertices[idx].color = Vector4uc(0,0,0,0);
		// otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap
		Vertex* vertices = new Vertex[sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight()];
		
		for (unsigned int y = 0; y < sensor.GetDepthImageHeight(); y=y+1) {
			for (unsigned int x = 0; x < sensor.GetDepthImageWidth(); x=x+1) {
				int current_index = y* sensor.GetDepthImageWidth() + x ;
				float depth = depthMap[current_index];
				if (depth == MINF) {
					vertices[current_index].position = Vector4f(MINF, MINF, MINF, MINF);
					vertices[current_index].color = Vector4uc(0,0,0,0);
				}
				else {
					Vector3f extended_pixel_coordinate = Vector3f(depth * x, depth * y, depth);
					Vector3f camera_coordinate = inverse_Intrinsics * extended_pixel_coordinate;
					//Vector4f extended_camera_coordinate = Vector4f(camera_coordinate[0], camera_coordinate[1], camera_coordinate[2], 1.0f);
					Vector4f extended_camera_coordinate = extendedId * camera_coordinate;
					extended_camera_coordinate[3] = 1;
					Vector4f p_world  = trajectoryInv * depthExtrinsicsInv * extended_camera_coordinate;
					vertices[current_index].position = p_world; 
					vertices[current_index].color = Vector4uc(colorMap[4*current_index], colorMap[4 * current_index+1], colorMap[4 * current_index+2], colorMap[4 * current_index+3]);
				}
			}
		}


		// write mesh file
		std::stringstream ss;
		ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
		if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return -1;
		}

		// free mem
		delete[] vertices;
	}

	return 0;
}
