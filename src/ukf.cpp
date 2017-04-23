#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF( double std_a, double std_yawdd, bool use_laser, bool use_radar  )
:

  is_initialized_(false),

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_(use_laser),

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_(use_radar),

  // State dimension
  n_x_(5),

  // Augmented state dimension
  n_aug_(7),

  // Sigma point spreading parameter
  lambda_(3-n_aug_),

  // Weights of sigma points
  weights_(2*n_aug_+1),

  // initial state vector
  x_(n_x_),

  // initial covariance matrix
  P_(n_x_, n_x_),

  // Predicted sigma points matrix
  Xsig_pred_(n_x_, 2*n_aug_+1),

  // Time when the state is true, in us
  time_us_(0),

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_(std_a),

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_(std_yawdd),

  // Laser measurement noise standard deviation position1 in m
  std_laspx_(0.15),

  // Laser measurement noise standard deviation position2 in m
  std_laspy_(0.15),

  // Radar measurement noise standard deviation radius in m
  std_radr_(0.3),

  // Radar measurement noise standard deviation angle in rad
  std_radphi_(0.03),

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_(0.3),

  // Current NIS for radar
  NIS_radar_(0.),

  // Current NIS for laser
  NIS_laser_(0.),

  // Radar measurement noise covariance matrix
  R_radar_(3, 3),

  // Laser measurement noise covariance matrix
  R_laser_(2, 2)

{
  
  // calculate weights of sigma points
  weights_(0) = lambda_/(lambda_+n_aug_);
  double ww = 0.5/(lambda_+n_aug_);
  for( size_t i=1; i<2*n_aug_+1; ++i )
    weights_(i) = ww;

  // calculate radar measurement noise covariance matrix
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;

  // calculate laser measurement noise covariance matrix
  R_laser_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  // initialize covariance matrix to zeros
  // diagonal elements are later set by ProcessMeasurement()
  // in the initialization cycle
  P_.fill(0.);

}


UKF::~UKF() {}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement( MeasurementPackage meas_package )
{

  // if sensor is disabled, skip processing cycle
  if( (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
      || (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
      || (meas_package.raw_measurements_.array().abs().maxCoeff() == 0) )
    return;


  // initialization cycle
  if( !is_initialized_ )
  {

    // case laser measurement
    if( meas_package.sensor_type_ == MeasurementPackage::LASER )
    {

      std::cout << "Initialization with lidar measurement" << std::endl;

      // assume velocity, yaw and yaw rate are zero
      x_ << meas_package.raw_measurements_(0),  // px
            meas_package.raw_measurements_(1),  // py
            0., 0., 0.;                         // v, yaw, yawd

      // initialize P_ based on sensor stdev for px and py,
      // and based on max expected values for v, yaw and yawd
      // (off-diagonal elements were set to 0. by constructor)
      P_(0,0) = std_laspx_*std_laspx_;   // px
      P_(1,1) = std_laspy_*std_laspy_;   // py
      P_(2,2) = 5.;                      // v
      P_(3,3) = (M_PI/2)*(M_PI/2);       // yaw
      P_(4,4) = (M_PI/5)*(M_PI/5);       // yawd

    }

    // case radar measurement
    else
    {

      std::cout << "Initialization with radar measurement" << std::endl;

      double ro = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rod = meas_package.raw_measurements_(2);

      // assume non-radial component of velocity is zero
      // assume yaw rate is zero
      x_ << ro*cos(phi),  // px
            ro*sin(phi),  // py
            rod,          // v
            phi,          // yaw
            0.;           // yawd

      // initialize P_ based on sensor stdevs for px and py,
      // and based on max expected values for v, yaw and yawd
      // (off-diagonal elements were set to 0. by constructor)
      P_(0,0) = ro*ro*sin(phi)*sin(phi)*std_radphi_*std_radphi_ + cos(phi)*cos(phi)*std_radr_*std_radr_;   // px
      P_(1,1) = ro*ro*cos(phi)*cos(phi)*std_radphi_*std_radphi_ + sin(phi)*sin(phi)*std_radr_*std_radr_;   // py
      P_(2,2) = 5.;                      // v
      P_(3,3) = (M_PI/2)*(M_PI/2);       // yaw
      P_(4,4) = (M_PI/5)*(M_PI/5);       // yawd
    }

    is_initialized_ = true;

  }

  // subsequent cycles
  else
  {

    // prediction step

    double delta_t = static_cast<double>(meas_package.timestamp_-time_us_)/1e6;

    Prediction(delta_t);
    std::cout << "predicted state : " << x_ << std::endl;
  
    // update step
    if( meas_package.sensor_type_ == MeasurementPackage::LASER )
      UpdateLidar(meas_package);
    else
      UpdateRadar(meas_package);

    std::cout << "corrected state : " << x_ << std::endl;

  }

  // update time for next cycle
  time_us_ = meas_package.timestamp_;

  return;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{

 // Augment state and calculate sigma points ---------------------

  // augmented mean state
  VectorXd x_aug = VectorXd(7);
  x_aug.head(n_x_) = x_;
  x_aug(n_aug_-2) = 0.; x_aug(n_aug_-1) = 0.;

  // augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_aug_-2,n_aug_-2) = std_a_*std_a_;
  P_aug(n_aug_-1,n_aug_-1) = std_yawdd_*std_yawdd_;

  // augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);

  // square root matrix of P_aug
  MatrixXd A = P_aug.llt().matrixL();
  A.array() = A.array()*sqrt(lambda_+n_aug_);
  
  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for( size_t i=0; i<n_aug_; ++i )
  {
      Xsig_aug.col(1+i) = x_aug + A.col(i);
      Xsig_aug.col(1+n_aug_+i) = x_aug - A.col(i);
  }

  //std::cout << "augmented sigma points Xsig_aug:" << std::endl;
  //for( size_t ii=0; ii<15; ++ii )
  //  std::cout << "sigp " << ii << " : " << Xsig_aug.col(ii) << std::endl;


  // Predict sigma points -------------------------------------------------

  for( size_t i=0; i<2*n_aug_+1; ++i )
  {
    // extract values for better readability
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if ( fabs(yawd) > 0.001 )
    {
        px_p = px + v/yawd * ( sin(yaw + yawd*delta_t) - sin(yaw) );
        py_p = py + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else
    {
        px_p = px + v*delta_t*cos(yaw);
        py_p = py + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p += 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p += 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p += nu_a*delta_t;

    yaw_p += 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p += nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;

    //std::cout << "predicted sigma points Xsig_pred_:" << std::endl;
    //for( size_t ii=0; ii<15; ++ii )
    //  std::cout << "sigp " << ii << " : " << Xsig_pred_.col(ii) << std::endl;

  }


  // Calculated mean and covariance of predicted sigma points -----------

  // predict state mean
  x_.fill(0.);
  for( size_t i=0; i<2*n_aug_+1; ++i )
    x_ += weights_(i)*Xsig_pred_.col(i);
  
  // predict state covariance matrix
  P_.fill(0.);
  for( size_t i=0; i<2*n_aug_+1; ++i )
  {
      VectorXd residual = Xsig_pred_.col(i)-x_;
      P_ += weights_(i)*residual*(residual.transpose());
  }

  return;
}



/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{

  std::cout << "Update lidar measurement" << std::endl;

  // Predict measurement ---------------------------------------------

  // dimension of measurement space
  int n_z = 2;

  // construct matrix for sigma points in measurement space
  MatrixXd Zsig(n_z, 2*n_aug_+1);

  // construct vector for mean predicted measurement
  VectorXd z_pred(n_z);
  
  // construct matrix for measurement covariance matrix S
  MatrixXd S(n_z, n_z);

  // transform sigma points into measurement space
  Zsig = Xsig_pred_.topRows(2);
  //for( size_t i=0; i<2*n_aug_+1; ++i )
  //{
  //    VectorXd x = Xsig_pred_.col(i);  // x, y, v, phi, phid
  //    Zsig.col(i) << x(0), x(1);
  //}

  // calculate mean predicted measurement
  z_pred.fill(0.);
  for( size_t i=0; i<2*n_aug_+1; ++i )
    z_pred += weights_(i)*Zsig.col(i);
  
  // calculate measurement covariance matrix S
  S.fill(0.);
  for( size_t i=0; i<2*n_aug_+1; ++i )
  {
    VectorXd residual = Zsig.col(i)-z_pred;
    S += weights_(i)*residual*residual.transpose();
  }

  // add sensor noise contribution
  S += R_laser_;


  // Update state --------------------------------------------

  // construct matrix for cross correlation Tc
  MatrixXd Tc(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.);

//std::cout << "Calculate cross-correlation matrix Laser..." << std::endl;

  for( size_t i=0; i<2*n_aug_+1; ++i )
  {
    VectorXd dx = Xsig_pred_.col(i)-x_;
    VectorXd dz = Zsig.col(i)-z_pred;

    //std::cout << "sigma point " << i << std::endl;
    //std::cout << "values : " << Xsig_pred_.col(i) << std::endl;
    //std::cout << "x_ : " << x_ << std::endl;

    // normalize angles
    normalizeDang(dx(3));

    Tc += weights_(i)*dx*dz.transpose();
  }

  //std::cout << "Cross-correlation matrix Laser : " << Tc << std::endl;

  // calculate Kalman gain K
  MatrixXd Sinv = S.inverse();
  MatrixXd K = Tc*Sinv; 
  
  // update state mean and covariance matrix
  VectorXd zresidual = meas_package.raw_measurements_-z_pred;
  x_ += K*zresidual;
  P_ -= K*S*K.transpose();


  // Calculate Normalized Innovation Squared --------------------------

  NIS_laser_ = zresidual.transpose()*Sinv*zresidual;


  return;
}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar( MeasurementPackage meas_package )
{

  std::cout << "Update radar measurement" << std::endl;

  // Predict measurement ---------------------------------------------

  // dimension of measurement space
  int n_z = 3;

  // construct matrix for sigma points in measurement space
  MatrixXd Zsig(n_z, 2*n_aug_+1);

  // construct vector for mean predicted measurement
  VectorXd z_pred(n_z);
  
  // construct matrix for measurement covariance matrix S
  MatrixXd S(n_z, n_z);

  // transform sigma points into measurement space
  for( size_t i=0; i<2*n_aug_+1; ++i )
  {
      VectorXd x = Xsig_pred_.col(i);  // x, y, v, yaw, yawd
      double ro = sqrt(x(0)*x(0) + x(1)*x(1));
      if( ro < 1e-8 )
      {
        std::cout << "ERROR: division by zero" << std::endl;
        std::exit(1);
      }
      Zsig.col(i) << ro, atan2(x(1),x(0)), (x(0)*cos(x(3))+x(1)*sin(x(3)))*x(2)/ro;
  }

  // calculate mean predicted measurement
  z_pred.fill(0.);
  for( size_t i=0; i<2*n_aug_+1; ++i )
    z_pred += weights_(i)*Zsig.col(i);
  
  // calculate measurement covariance matrix S
  S.fill(0.);
  for( size_t i=0; i<2*n_aug_+1; ++i )
  {
    VectorXd residual = Zsig.col(i)-z_pred;

    // angle normalization
    normalizeDang(residual(1));

    S += weights_(i)*residual*residual.transpose();
  }
  
  // add sensor noise contribution
  S += R_radar_;


  // Update state --------------------------------------------

  // construct matrix for cross correlation Tc
  MatrixXd Tc(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.);
  for( size_t i=0; i<2*n_aug_+1; ++i )
  {
    VectorXd dx = Xsig_pred_.col(i)-x_;
    VectorXd dz = Zsig.col(i)-z_pred;
    
    // normalize angles
    normalizeDang(dx(3));
    normalizeDang(dz(1));

    Tc += weights_(i)*dx*dz.transpose();
  }

  // calculate Kalman gain K
  MatrixXd Sinv = S.inverse();
  MatrixXd K = Tc*Sinv; 
  
  // update state mean and covariance matrix
  VectorXd zresidual = meas_package.raw_measurements_-z_pred;
  normalizeDang(zresidual(1));

  x_ += K*zresidual;
  P_ -= K*S*K.transpose();


  // Calculate Normalized Innovation Squared --------------------------
  
  NIS_radar_ = zresidual.transpose()*Sinv*zresidual;

  return;
}


void UKF::normalizeDang( double& dang )
{

  //std::cout << "Entering normalizeDang with dang = " << dang << "..." << std::endl;
  while (dang> M_PI) dang-=2.*M_PI;
  while (dang<-M_PI) dang+=2.*M_PI;
  //std::cout << "Leaving normalizeDang with dang = " << dang << "..." << std::endl;

  return;
}
