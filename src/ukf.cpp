#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);
  P_(2, 2) = 0.5;
  P_(3, 3) = 0.5;
  P_(4, 4) = 0.5;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;

  x_aug_ = VectorXd::Zero(7);
  P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i = 1; i <= 2* n_aug_; i++)
  {
    weights_(i) = 0.5/(lambda_ + n_aug_);
  }

  R_Lidar = MatrixXd::Zero(2, 2);
  R_Lidar(0, 0) = std_laspx_ * std_laspx_;
  R_Lidar(1, 1) = std_laspy_ * std_laspy_;

  R_Radar = MatrixXd::Zero(3, 3);
  R_Radar(0, 0) = std_radr_ * std_radr_;
  R_Radar(1, 1) = std_radphi_ * std_radphi_;
  R_Radar(2, 2) = std_radrd_ * std_radrd_;
}

UKF::~UKF() {}

double UKF::Normalize(double angle)
{
  while (angle > M_PI) angle -= 2*M_PI;
  while (angle <-M_PI) angle += 2*M_PI;
  return angle;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_(0) = meas_package.raw_measurements_[0];
      x_(1) = meas_package.raw_measurements_[1];
      is_initialized_ = true;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      double rho = meas_package.raw_measurements_[0];
      double angle = meas_package.raw_measurements_[1];
      double rho_d = meas_package.raw_measurements_[2];
      x_(0) = rho * cos(angle);
      x_(1) = rho * sin(angle);
      x_(2) = rho_d;
      is_initialized_ = true;
    }
    time_us_ = meas_package.timestamp_;

    return;
  }
  double delta_t = (meas_package.timestamp_ - time_us_)/ 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {    
    UpdateLidar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  {
    UpdateRadar(meas_package);
  }
  
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  x_aug_.fill(0);
  x_aug_.head(5) = x_;

  std::cout << "====================Prediction Started============================"<<std::endl;
  std::cout << "X Before " << std::endl << x_.transpose() << std::endl;

  //create augmented covariance matrix
  P_aug_.fill(0);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(n_x_, n_x_) = std_a_ * std_a_;
  P_aug_(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
  MatrixXd L = P_aug_.llt().matrixL();

  //calculate sigma points for the augmented state
  MatrixXd Xsig_aug_ = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug_.col(0) = x_aug_;

  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug_.col(i + 1)          = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(n_aug_ + i + 1) = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  //predict the sigma values
  for (int i = 0; i <= 2 * n_aug_; i++)
  {
    double px_sigma       = Xsig_aug_(0, i);
    double py_sigma       = Xsig_aug_(1, i);
    double v_sigma        = Xsig_aug_(2, i);
    double yaw_sigma      = Xsig_aug_(3, i);
    double yawrate_sigma  = Xsig_aug_(4, i);
    double accel_sigma    = Xsig_aug_(5, i);
    double yawaccel_sigma = Xsig_aug_(6, i);

    double px_pred, py_pred, v_pred, yaw_pred, yawrate_pred;

    if (abs(yawrate_sigma) > 0.001)
    {
      px_pred = px_sigma + (v_sigma / yawrate_sigma) * (sin(yaw_sigma + yawrate_sigma * delta_t) - sin(yaw_sigma)) + 0.5 * delta_t * delta_t * cos(yaw_sigma) * accel_sigma;
      py_pred = py_sigma + (v_sigma / yawrate_sigma) * (-cos(yaw_sigma + yawrate_sigma * delta_t) + cos(yaw_sigma)) + 0.5 * delta_t * delta_t * sin(yaw_sigma) * accel_sigma;
    }
    else
    {
      px_pred = px_sigma + v_sigma * cos(yaw_sigma) * delta_t + 0.5 * delta_t * delta_t * cos(yaw_sigma) * accel_sigma;
      py_pred = py_sigma + v_sigma * sin(yaw_sigma) * delta_t + 0.5 * delta_t * delta_t * sin(yaw_sigma) * accel_sigma;
    }
    v_pred       = v_sigma       + delta_t * accel_sigma;
    yaw_pred     = yaw_sigma     + yawrate_sigma * delta_t + 0.5 * delta_t * delta_t * yawaccel_sigma;
    yawrate_pred = yawrate_sigma + delta_t * yawaccel_sigma;

    Xsig_pred_(0, i) = px_pred;
    Xsig_pred_(1, i) = py_pred;
    Xsig_pred_(2, i) = v_pred;
    Xsig_pred_(3, i) = yaw_pred;
    Xsig_pred_(4, i) = yawrate_pred;
  }

  //predict state mean and covariance matrix
  x_.fill(0);
  P_.fill(0);
  for (int i = 0; i <= 2 * n_aug_; i++)
  {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  for (int i = 0; i <= 2 * n_aug_; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) >  M_PI) x_diff(3) -= 2 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2 * M_PI;
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

  std::cout << "X after" << std::endl << x_.transpose() << std::endl;

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  std::cout << "========================= Measurement Update ===============================" << std::endl;
  std::cout << "X Before " << std::endl << x_.transpose() << std::endl;
  MatrixXd H_ = MatrixXd(2, n_x_);
  H_ << 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0;

  VectorXd z  = VectorXd(2);
  z(0) = meas_package.raw_measurements_[0];
  z(1) = meas_package.raw_measurements_[1];

  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_Lidar;

  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  x_ = x_ + K * y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
  std::cout << "X after " << std::endl << x_.transpose() << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  MatrixXd H_ = MatrixXd(3, n_x_);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0;
  
  //create vector of measured values of radar rho, angle, rho_d
  VectorXd z = VectorXd(3);
  z(0) = meas_package.raw_measurements_[0];
  z(1) = meas_package.raw_measurements_[1];
  z(2) = meas_package.raw_measurements_[2];

  //create the matrix of Sigma Z values corresponding to the Sigma X values
  MatrixXd z_sigma_pred = MatrixXd::Zero(3, 2 * n_aug_ + 1);
  for (int i = 0; i <= 2 * n_aug_; i++)
  {
    double px_sigma_pred      = Xsig_pred_(0, i);
    double py_sigma_pred      = Xsig_pred_(1, i);
    double v_sigma_pred       = Xsig_pred_(2, i);
    double yaw_sigma_pred     = Xsig_pred_(3, i);
    double yawrate_sigma_pred = Xsig_pred_(4, i);

    double rad_sigma_pred = sqrt(px_sigma_pred * px_sigma_pred + py_sigma_pred * py_sigma_pred);
    double angle_sigma_pred  = atan2(py_sigma_pred, px_sigma_pred);
    double vrad_sigma_pred   = (px_sigma_pred * cos(yaw_sigma_pred) * v_sigma_pred + py_sigma_pred + sin(yaw_sigma_pred) * v_sigma_pred)/ (rad_sigma_pred);

    z_sigma_pred(0, i) = rad_sigma_pred;
    z_sigma_pred(1, i) = angle_sigma_pred;
    z_sigma_pred(2, i) = vrad_sigma_pred;
  }


  //predicted mean z
  VectorXd z_pred = VectorXd::Zero(3);
  for (int i = 0; i <= 2*n_aug_; i++)
  {
    z_pred = z_pred + weights_(i) * z_sigma_pred.col(i);
  }

  //predicted covariance 
  MatrixXd S = MatrixXd::Zero(3, 3);
  for (int i = 0; i <= 2 * n_aug_; i++)
  {
    VectorXd z_diff = z_sigma_pred.col(i) - z_pred;
    z_diff(1) = Normalize(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose(); 
  }

  S = S + R_Radar;

  MatrixXd Tc = MatrixXd::Zero(n_x_, 3);
  for (int i = 0; i <= 2 * n_aug_; i++)
  {
    VectorXd z_diff = z_sigma_pred.col(i) - z_pred;
    z_diff(1) = Normalize(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = Normalize(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  MatrixXd K = Tc * S.inverse();
  
  VectorXd z_diff = z - z_pred;
  z_diff(1) = Normalize(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}