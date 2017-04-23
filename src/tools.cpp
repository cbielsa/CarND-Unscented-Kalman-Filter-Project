#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{

  if( estimations.size() != ground_truth.size() || estimations.size()==0 )
  {
		std::cerr << "Tools::CalculateRMSE: invalid input dimensions.\n" << std::endl;
		exit(EXIT_FAILURE);  	
  }

  VectorXd rmse(estimations[0].size());
  rmse.fill(0.);

  //accumulate squared residuals
  for(std::size_t i=0; i<estimations.size(); ++i)
  {
  	VectorXd residual = estimations[i] - ground_truth[i];  
  	//coefficient-wise multiplication
  	residual = residual.array() * residual.array();
  	rmse += residual;
  }

  return ( rmse/estimations.size() ).array().sqrt();
}
