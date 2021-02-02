#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "Eigen.h"
#include <opencv2/core/core.hpp>

template <typename T>
class PoseConverter {
public:

	/**
	 *	Converts 4x4 transformation matrix to 6DOF for optimization 
	 */
	static T* pose6DOF(const Matrix4f& rotationMatrix) {
		T* pose = new T[3];

		T rotMatrix[9] = { rotationMatrix(0,0), rotationMatrix(1,0), rotationMatrix(2,0), rotationMatrix(0,1),
			rotationMatrix(1,1), rotationMatrix(2,1), rotationMatrix(0,2), rotationMatrix(1,2), rotationMatrix(2,2) };
		
		
		ceres::RotationMatrixToAngleAxis(rotMatrix, pose);

		T* pose6DOF = new T[6];
		pose6DOF[0] = pose[0];
		pose6DOF[1] = pose[1];
		pose6DOF[2] = pose[2];
		pose6DOF[3] = rotationMatrix(0, 3);
		pose6DOF[4] = rotationMatrix(1, 3);
		pose6DOF[5] = rotationMatrix(2, 3);

		return pose6DOF;
	}

	static Matrix4f convertToMatrix(T* pose) {
		// pose[0,1,2] is angle-axis rotation.
		// pose[3,4,5] is translation.
		double* rotation = pose;
		double* translation = pose + 3;

		// Convert the rotation from SO3 to matrix notation (with column-major storage).
		double rotationMatrix[9];
		ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

		// Create the 4x4 transformation matrix.
		Matrix4f matrix;
		matrix.setIdentity();
		matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
		matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
		matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);

		return matrix;
	}

	static T* pointVectorToPointer(const Vector3f& m_point) {
		T* point = new T[3];

		point[0] = m_point.x();
		point[1] = m_point.y();
		point[2] = m_point.z();

		return point;
	}
};

/**
 * Optimization constraints.
 */
class Point3DTo2DConstraint {
public:
	Point3DTo2DConstraint(const Vector2f& observation) :
		m_observation{ observation }
	{ }

	template <typename T>
	bool operator()(const T* const pose, const T* const point, T* residuals) const {

		const T* rotation = pose;
		const T* translation = pose + 3;

		// Rotation.
		T pos_proj[3];
		ceres::AngleAxisRotatePoint(rotation, point, pos_proj);

		// Translation.
		pos_proj[0] += translation[0];
		pos_proj[1] += translation[1];
		pos_proj[2] += translation[2];

		T x = pos_proj[0] / pos_proj[2];
		T y = pos_proj[1] / pos_proj[2];

		// Intrinsics
		const T focal_x = T(517.3);
		const T focal_y = T(516.5);
		const T cx = T(318.6);
		const T cy = T(255.3);

		// Compute final projected point position.
		T predicted_x = focal_x * x + cx;
		T predicted_y = focal_y * y + cy;

		residuals[0] = predicted_x - T(m_observation[0]);
		residuals[1] = predicted_y - T(m_observation[1]);

		return true;
	}

	static ceres::CostFunction* create(const Vector2f& observation) {
		return new ceres::AutoDiffCostFunction<Point3DTo2DConstraint, 2, 6, 3>(
			new Point3DTo2DConstraint(observation)
			);
	}

protected:
	const Vector2f m_observation;
};


/**
 * Optimizer - using Ceres for optimization.
 */
 
class Optimization {
public:
	Optimization() :
		m_nIterations{ 10 }
	{ }

	void setNbOfIterations(unsigned nIterations) {
		m_nIterations = nIterations;
	}

	void solve_Bundle_Adjustment(std::vector<PointCloud>& pointClouds, std::vector<Vector3f>& global_3D_points) {

		double** poses = (double**)malloc(sizeof(double*) * (pointClouds.size()-2));
		for (int k = 0; k < (pointClouds.size() - 2); k++)
			poses[k] = (double*)malloc(sizeof(double) * 6);

		double** points3D = (double**)malloc(sizeof(double*) * global_3D_points.size());
		for (int i = 0; i < global_3D_points.size(); i++) {
			points3D[i] = (double*)malloc(sizeof(double) * 3);
			points3D[i] = PoseConverter<double>::pointVectorToPointer(global_3D_points[i]);
		}

		std::cout << "Starting BA"  << std::endl;
		for (int i = 0; i < m_nIterations; ++i) {

			ceres::Problem problem;

			// Configure options for the solver.
			ceres::Solver::Options options;
			configureSolver(options);

			for (int j = 0; j < (pointClouds.size()-2); j++) {

				Matrix4f estimatedPose = pointClouds[j].getCameraExtrinsics();
				poses[j] = PoseConverter<double>::pose6DOF(estimatedPose);

				prepareConstraints(pointClouds[j], points3D, poses[j], problem);
				if (j == 0) {
					problem.SetParameterBlockConstant(poses[j]);
				}
			}

			// Run the solver (for one iteration)
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			//std::cout << summary.BriefReport() << std::endl;
			std::cout << summary.FullReport() << std::endl;

			for (int j = 0; j < (pointClouds.size() - 2); j++) {
				Matrix4f updated_matrix = PoseConverter<double>::convertToMatrix(poses[j]);
				pointClouds[j].setCameraExtrinsics(updated_matrix);
			}

			//Update 3D points
			for (int i = 0; i < global_3D_points.size(); i++) {

				if (global_3D_points[i][0] != MINF)
					global_3D_points[i] = Vector3f(float(points3D[i][0]), float(points3D[i][1]), float(points3D[i][2]));
			}
		}

		// Free memory
		for (int i = 0; i < (pointClouds.size() - 2); i++)
			free(poses[i]);
		free(poses);

		for (int i = 0; i < global_3D_points.size(); i++)
			free(points3D[i]);
		free(points3D);
	}


private:
	unsigned m_nIterations;

	void configureSolver(ceres::Solver::Options& options) {
		// Ceres options.
		options.minimizer_progress_to_stdout = false;
		options.logging_type = ceres::SILENT;
		options.num_threads = 1;
		options.preconditioner_type = ceres::JACOBI;
		options.linear_solver_type = ceres::SPARSE_SCHUR;
		options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
	}

	void prepareConstraints(PointCloud& pointCloud, double** global_3D_points, double* pose, ceres::Problem& problem) const {


		std::vector<cv::KeyPoint> points2D = pointCloud.getPoints2d();

		std::map<int, int> match_2D_to_3D = pointCloud.get2d_3d_correspondence();

		for (std::map<int, int>::iterator it = match_2D_to_3D.begin(); it != match_2D_to_3D.end(); ++it) {

			if (global_3D_points[it->second][0] == MINF)
				continue;

			cv::Point2f tmp = points2D[it->first].pt;
			const auto& targetPoint = Vector2f(tmp.x, tmp.y);

			if (!targetPoint.allFinite()) {
				continue;
			}

			ceres::LossFunction* loss_function = new ceres::HuberLoss(4);
			problem.AddResidualBlock(Point3DTo2DConstraint::create(targetPoint), loss_function, pose, global_3D_points[it->second]);

		}
	}

};
