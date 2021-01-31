#include "utils/io.h"
#include "utils/points.h"

#include "ceres/ceres.h"
#include <math.h>


// TODO: Implement the cost function
struct RegistrationCostFunction
{
	RegistrationCostFunction(const Point2D& point1_, const Point2D& point2_, const Weight& weight_)
		: point1(point1_), point2(point2_), weight(weight_)
	{
	}

	template<typename T>
	bool operator()(const T* const degree, const T* const tx, const T* const ty, T* residual) const
	{
		// TODO: Implement the cost function 
		// convert radian to degree
		T angle_rad = degree[0] * T(0.0174533);
		residual[0] = sqrt(weight.w)*sqrt(pow(cos(angle_rad) * point1.x - sin(angle_rad) * point1.y + tx[0] - point2.x, 2) + pow(sin(angle_rad) * point1.x + cos(angle_rad) * point1.y + ty[0] - point2.y, 2));		
		
		return true;
	}

private:
	const Point2D point1;
	const Point2D point2;
	const Weight weight;
};

int main(int argc, char** argv)
{
	google::InitGoogleLogging(argv[0]);

	// Read data points and the weights, and define the parameters of the problem
	//const std::string file_path_1 = "F:/TUM Learning Material/20WS/Informatik/3D Scanning & Motion Capture/exercise_4/exercise_4_src/data/points_dragon_1.txt";
	const std::string file_path_1 = "../data/points_dragon_1.txt";
	const auto points1 = read_points_from_file<Point2D>(file_path_1);
	
	//const std::string file_path_2 = "F:/TUM Learning Material/20WS/Informatik/3D Scanning & Motion Capture/exercise_4/exercise_4_src/data/points_dragon_2.txt";
	const std::string file_path_2 = "../data/points_dragon_2.txt";
	const auto points2 = read_points_from_file<Point2D>(file_path_2);
	
	//const std::string file_path_weights = "F:/TUM Learning Material/20WS/Informatik/3D Scanning & Motion Capture/exercise_4/exercise_4_src/data/weights_dragon.txt";
	const std::string file_path_weights = "../data/weights_dragon.txt";
	const auto weights = read_points_from_file<Weight>(file_path_weights);
	
	const double angle_initial = 0.0;
	const double tx_initial = 0.0;
	const double ty_initial = 0.0;
	
	double angle = angle_initial;
	double tx = tx_initial;
	double ty = ty_initial;

	ceres::Problem problem;

	// TODO: For each weighted correspondence create one residual block
	size_t num_points = points1.size();
	for (int idx = 0; idx<num_points; idx++)
	{
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<RegistrationCostFunction, 1, 1, 1, 1>(
				new RegistrationCostFunction(points1[idx], points2[idx], weights[idx])
				),
			nullptr, &angle, &tx, &ty
		);
	}



	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;

	// Output the final values of the translation and rotation (in degree)
	std::cout << "Initial angle: " << angle_initial << "\ttx: " << tx_initial << "\tty: " << ty_initial << std::endl;
	std::cout << "Final angle: " << std::fmod(angle * 180 / M_PI, 360.0) << "\ttx: " << tx << "\tty: " << ty << std::endl;
	//std::cout << "Final angle: " << angle << "\ttx: " << tx << "\tty: " << ty << std::endl;
	system("pause");
	return 0;
}
