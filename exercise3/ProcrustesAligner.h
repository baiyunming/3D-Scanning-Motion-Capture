#pragma once
#include "SimpleMesh.h"

class ProcrustesAligner {
public:
	Matrix4f estimatePose(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
		ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");
		
		// We estimate the pose between source and target points using Procrustes algorithm.
		// Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
		// from source points to target points.

		auto sourceMean = computeMean(sourcePoints);
		auto targetMean = computeMean(targetPoints);
		
		Matrix3f rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
		Vector3f translation = computeTranslation(sourceMean, targetMean);

		// To apply the pose to point x on shape X in the case of Procrustes, we execute:
		// 1. Translation of a point to the shape Y: x' = x + t
		// 2. Rotation of the point around the mean of shape Y: 
		//    y = R (x' - yMean) + yMean = R (x + t - yMean) + yMean = R x + (R t - R yMean + yMean)
		
		// TODO: Compute the transformation matrix by using the computed rotation and translation.
		// You can access parts of the matrix with .block(start_row, start_col, num_rows, num_cols) = elements
		Matrix4f estimatedPose = Matrix4f::Identity();
		estimatedPose.block(0, 0, 3, 3) = rotation;
		estimatedPose.block(0, 3, 3, 1) = rotation * translation - rotation * targetMean + targetMean;

		return estimatedPose;
	}

private:
	Vector3f computeMean(const std::vector<Vector3f>& points) {
		// TODO: Compute the mean of input points.
		Vector3f mean = Vector3f::Zero();
		
		for (auto& point : points) {
			mean +=  point;
		}
		mean = mean / points.size();
		return mean;
		
	}

	Matrix3f estimateRotation(const std::vector<Vector3f>& sourcePoints, const Vector3f& sourceMean, const std::vector<Vector3f>& targetPoints, const Vector3f& targetMean) {
		// TODO: Estimate the rotation from source to target points, following the Procrustes algorithm.
		// To compute the singular value decomposition you can use JacobiSVD() from Eigen.
		// Important: The covariance matrices should contain mean-centered source/target points.
		Matrix3f rotation = Matrix3f::Identity();
		Matrix3f Source_matrix(3, sourcePoints.size());
		Matrix3f Target_matrix(3, targetPoints.size());
		
		for (size_t i = 0; i < sourcePoints.size(); i++) {
			Vector3f source = sourcePoints[i]-sourceMean;
			Source_matrix(0, i) = source.x();
			Source_matrix(1, i) = source.y();
			Source_matrix(2, i) = source.z();
		}

		for (size_t j = 0; j < targetPoints.size(); j++) {
	     	Vector3f target = targetPoints[j] - targetMean;
			Target_matrix(0, j) = target.x();
			Target_matrix(1, j) = target.y();
			Target_matrix(2, j) = target.z();
		}

		Matrix3f target_source = Target_matrix * Source_matrix.transpose();
		JacobiSVD<Matrix3f> svd(target_source, ComputeThinU | ComputeThinV);

		const float d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
		Matrix3f D = Matrix3f::Identity();
		D(2, 2) = d;

		Matrix3f rotation = svd.matrixU() * D * svd.matrixV().transpose();

		return rotation;
	}

	Vector3f computeTranslation(const Vector3f& sourceMean, const Vector3f& targetMean) {
		// TODO: Compute the translation vector from source to target points.
		Vector3f translation = targetMean - sourceMean;

		return translation;
	}
};