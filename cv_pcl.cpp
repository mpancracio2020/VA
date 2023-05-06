/*
Autor: Marvin Pancracio Manso.
Partes implementadas:
- Detección de pelota en 2D y proyección 3D
- Detección de pelota en 3D y proyección 2D  (problemas a la hora de proyectar el punto en 2D)
- Proyección líneas
- Proyección de cubos en función del slider distance.
- Funcionalidad extra:
- 
*/

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <image_transport/image_transport.hpp>

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/core.hpp"


#include "image_geometry/pinhole_camera_model.h"


#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types_conversion.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud);
cv::Mat image_processing(const cv::Mat in_image);
const char * window_name = "PRACTICA_FINAL";
const char * trackbar_name_1 = "Option";
const char * trackbar_name_2 = "Distance ";
const int init_value_t1= 2;
const int init_value_t2= 8;
const int min_set_value = 3;

const int op_KEY_0 = 0;
const int op_KEY_1 = 1;
const int op_KEY_2 = 2;

const int KEY_esc = 27;
cv::Matx<double, 3, 3>  K_;
int distance_trackbar;
int option_trackbar;
geometry_msgs::msg::TransformStamped t;
cv::Mat depth_img;
std::vector<cv::Point2f> point;
std::vector<pcl::PointXYZ> black_square_pos;
std::vector<pcl::PointXYZ> square_pos;
std::vector<pcl::PointXYZ> square_pos_2;
std::vector<cv::Point2f> point_line; // puntos de las lineas.
std::vector<cv::Point2f> point_2D; // puntos de las lineas.
int person = 0;
cv::dnn::Net net;
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
cv::Mat clone_img;
cv::Mat circles_img;
std::vector<cv::Point2f> k_center;
std::vector<int> k_radius;

const int MAX_CLUSTERS = 5;
cv::Scalar colorTab[] =
{
  cv::Scalar(0, 0, 255),
  cv::Scalar(0, 255, 0),
  cv::Scalar(255, 100, 100),
  cv::Scalar(255, 0, 255),
  cv::Scalar(0, 255, 255)
};


class ComputerVisionSubscriber : public rclcpp::Node
{
  public:
    ComputerVisionSubscriber()
    : Node("opencv_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/rgb/image_raw", qos, std::bind(&ComputerVisionSubscriber::topic_callback, this, std::placeholders::_1));
    
      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "cv_image", qos);

      intri_subs_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/head_front_camera/rgb/camera_info", qos, std::bind(
        &ComputerVisionSubscriber::intri_callback, this, std::placeholders::_1));
      
      depth_subs_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/depth_registered/image_raw", qos, std::bind(
        &ComputerVisionSubscriber::depth_callback, this, std::placeholders::_1));

      tf_buffer_ =
        std::make_unique<tf2_ros::Buffer>(this->get_clock());
      tf_listener_ =
        std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
      timer_ = this->create_wall_timer(
        std::chrono::seconds(1), std::bind(&ComputerVisionSubscriber::on_timer, this));
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw =  cv_ptr->image;

      // Image processing
      cv::Mat cv_image = image_processing(image_raw);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; // >> message to be sent
      img_bridge.toImageMsg(out_image); // from cv_bridge to sensor_msgs::Image

      // Publish the data
      publisher_ -> publish(out_image);
    }
    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
      depth_img = cv_ptr->image;
    }
    void intri_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) const
    {
      image_geometry::PinholeCameraModel model;
      model.fromCameraInfo(msg);
      K_ = model.intrinsicMatrix();

      //std::cout << "Matrix k: " << K_ << std::endl;
    }
      
    void on_timer() 
    {
      try {
          t = tf_buffer_->lookupTransform(
            "head_front_camera_rgb_optical_frame", "base_footprint",
            tf2::TimePointZero);
        } catch (const tf2::TransformException & ex) {
      
          return;
        }


    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_{nullptr};
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr intri_subs_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subs_;
};

class PCLSubscriber : public rclcpp::Node
{
  public:
    PCLSubscriber()
    : Node("pcl_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      subscription_3d_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/head_front_camera/depth_registered/points", qos, std::bind(&PCLSubscriber::topic_callback_3d, this, std::placeholders::_1));
    
      publisher_3d_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pcl_points", qos);
    }

  private:
    void topic_callback_3d(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {    
      // Convert to PCL data type
      pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
      pcl::fromROSMsg(*msg, point_cloud);     

      pcl::PointCloud<pcl::PointXYZRGB> pcl_pointcloud = pcl_processing(point_cloud);
      
      // Convert to ROS data type
      sensor_msgs::msg::PointCloud2 output;
      pcl::toROSMsg(pcl_pointcloud, output);
      output.header = msg->header;

      // Publish the data
      publisher_3d_ -> publish(output);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_3d_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_3d_;
};


/**
  TO-DO
*/


cv::Mat pink_filter_cv(const cv::Mat input_image) {
    
    cv::Mat  out_image, filtered_img;
    
    // change space color to HSV to use inRange function that keep blue colors.

    cvtColor(input_image, out_image, CV_BGR2HSV);
    inRange(out_image, cv::Scalar(145,75,35), cv::Scalar(165,255,255), filtered_img);
    return filtered_img;
}

cv::Mat balls_cv(const cv::Mat input_image) {

  cv::Mat gray,out_image, pink_img;
  out_image = input_image.clone();
  pink_img = pink_filter_cv(input_image);
  cvtColor(out_image, gray, CV_BGR2GRAY);
  
  cv::medianBlur(gray, gray, 5);
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles( gray, circles, cv::HOUGH_GRADIENT, 1,
                gray.rows/16,  // change this value to detect circles with different distances to each other
                175, 30, 1, 100 // change the last two parameters
          // (min_radius & max_radius) to detect larger circles
  );
  //circles_img = gray;
  for( size_t i = 0; i < circles.size(); i++ )
  {
      cv::Vec3i c = circles[i];
      int radius = c[2];
      cv::Point2f center = cv::Point2f(c[0], c[1]);
      k_center.push_back(center);
      k_radius.push_back(radius);
      //point.push_back(center);
      // circle center
      if (pink_img.at<uchar>(center) > 0) {
        point.push_back(center);
        cv::circle( out_image, center, 1, cv::Scalar(0,0,0), 3, cv::LINE_AA);
        cv::circle( out_image, center, radius, cv::Scalar(0,0,255), 3, cv::LINE_AA);
      }
      // circle outline  
  }
  return out_image;
}

cv::Mat project_lines(cv::Mat input_image){

  cv::Mat img_clone = input_image.clone();
  std::vector<cv::Point3f> point_3D;
  cv::Mat K, R, T;
  K = cv::Mat(3,3,CV_64F,K_.val);

  for (int i = 0; i < distance_trackbar; i++) {

    point_3D.push_back(cv::Point3f(i+1,-1.4,0));
    point_3D.push_back(cv::Point3f(i+1,1.4,0));
    
    R = (cv::Mat_<double>(3,3) << 0,1,0,0,0,1,1,0,0);
    T = (cv::Mat_<double>(3,1) << t.transform.translation.x,t.transform.translation.y,t.transform.translation.z);

    projectPoints(point_3D,R,T,K,cv::noArray(),point_line);
  }

  for (int i = 1; i < distance_trackbar; i++) {
    cv::circle(img_clone,point_line[i*2], 1, cv::Scalar(0,i*42,255-i*42), 5, cv::LINE_AA);
    cv::circle(img_clone,point_line[i*2+1], 1, cv::Scalar(0,i*42,255-i*42), 5, cv::LINE_AA);
    cv::line(img_clone,point_line[i*2], point_line[i*2+1], cv::Scalar(0,i*42,255-i*42),2,cv::LINE_AA);
    std::stringstream text;
    text << i+1;
    cv::putText(img_clone,text.str(),point_line[i*2+1] +  cv::Point2f(20,5),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,i*42,255-i*42));
  
  }
  return img_clone;
}

cv::Mat project_point_3D_2D(cv::Mat input_image){

  cv::Mat img_clone = input_image.clone();
  std::vector<cv::Point3f> point_3D;
  cv::Mat K, R, T;
  K = cv::Mat(3,3,CV_64F,K_.val);
  for (int i = 0; i < (int)square_pos_2.size(); i++) {

    point_3D.push_back(cv::Point3f(square_pos_2[i].x,square_pos_2[i].y,0));
    //point_3D.push_back(cv::Point3f(i+1,1.4,0));
    
    R = (cv::Mat_<double>(3,3) << 0,1,0,0,0,1,1,0,0);
    T = (cv::Mat_<double>(3,1) << t.transform.translation.x,t.transform.translation.y,t.transform.translation.z);

    projectPoints(point_3D,R,T,K,cv::noArray(),point_2D);
    std::cout << "x: " << point_2D[i].x << " y: " << point_2D[i].y << std::endl;
  }

  for (int i = 1; i < (int)point_2D.size(); i++) {
    cv::circle(img_clone,point_2D[i], 3, cv::Scalar(255,255,255), 5, cv::LINE_AA);
  }
  return img_clone;
}

pcl::PointCloud<pcl::PointXYZRGB> draw_square_black(pcl::PointCloud<pcl::PointXYZRGB> cloud_in, float x_, float y_, float z_) 
{ // metodo para dibujar cuadrados en una nube de puntos cloud_in. 

  float size = 0.1;
  float step = 0.01;
  std::vector<pcl::PointXYZRGB> vertices;
  for (float i = -size/2; i <= size/2; i += step) {
      for (float j = -size/2; j <= size/2; j += step) {
          for (float k = -size/2; k <= size/2; k += step) {
              pcl::PointXYZRGB vertex(x_+i, y_+j, z_+k, 0, 0, 0);
              cloud_in.push_back(vertex);
          }
      }
  }
  return cloud_in;
}

void project_points(pcl::PointCloud<pcl::PointXYZRGB> cloud_in) {
// metodo para proyectar los puntos de la imagen de profundidad en cubos negros en pcl.
  cv::Mat img_depth = depth_img.clone();
  pcl::PointXYZ position;

  std::vector<cv::Point3f> point3D;

  for (int i = 0; i < img_depth.rows; i++) // eliminamos los valores infinitos.
  {
    for (int j = 0; j < img_depth.cols; j++)
    {
      if(isnan(img_depth.at<float>(i, j)) || isinf(img_depth.at<float>(i, j)))
      {
        img_depth.at<float>(i, j) = 0.0;
      }
    }
  }

  int size = point.size();
  for (int i = 0; i < size; i++)
  {
    float d = img_depth.at<float>(point[i].y,point[i].x);
    float x = point[i].x;  float cx = img_depth.rows/2;
    float y = point[i].y; float cy = img_depth.cols/2;
  
    black_square_pos.push_back(pcl::PointXYZ((x - cx)*d/522,(y - cy)*d/522,d));
    position.x = (x - cx)*d/522;
    position.y = (y - cy)*d/522;
    position.z = d;
    //std::cout << "centro x: " << black_square_pos[i].x << "centro y: " << black_square_pos[i].y << "centro z: " << black_square_pos[i].z<< std::endl;
    
  }
}

void postprocess(cv::Mat & frame, const std::vector<cv::Mat> & outs)
{

  for (size_t i = 0; i < outs.size(); ++i) {
    // Scan through all the bounding boxes output from the network and keep only the
    // ones with high confidence scores. Assign the box's class label as the class
    // with the highest score for the box.
    float * data = (float *)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
      cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      cv::Point classIdPoint;
      double confidence;
      // Get the value and location of the maximum score
      cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confThreshold) {
        if (classIdPoint.x == 0)
        {
          person = 1;
        }
      }
    }
  }
}

std::vector<cv::String> getOutputsNames(const cv::dnn::Net & net)
{
  static std::vector<cv::String> names;
  if (names.empty()) {
    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    std::vector<int> outLayers = net.getUnconnectedOutLayers();

    //get the names of all the layers in the network
    std::vector<cv::String> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
      names[i] = layersNames[outLayers[i] - 1];
    }
  }
  return names;
}
void person_detected(cv::Mat input_img) {

  
  std::vector<cv::String> classes;
  cv::Mat frame, blob;

  frame = input_img;
  cv::dnn::blobFromImage(
      frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(
        0, 0,
        0), true, false);

  //Sets the input to the network
  net.setInput(blob);

  // Runs the forward pass to get output of the output layers
  std::vector<cv::Mat> outs;
  net.forward(outs, getOutputsNames(net));

  // Remove the bounding boxes with low confidence
  postprocess(frame, outs);
}

// EXTRA K-MEANS.

void k_means(cv::Mat input_img) {

  cv::RNG rng(k_center.size());

  for (;; ) {
    int k, clusterCount = rng.uniform(2, k_center.size() + 1);
    int i, sampleCount = rng.uniform(1, 1001);
    cv::Mat points(sampleCount, 1, CV_32FC2), labels;

    clusterCount = MIN(clusterCount, sampleCount);
    std::vector<cv::Point2f> centers;

    /* generate random sample from multigaussian distribution */
    for (k = 0; k < clusterCount; k++) {
      cv::Point center;
      center.x = k_center[k].x;
      center.y = k_center[k].y;
      cv::Mat pointChunk = points.rowRange(
        k * sampleCount / clusterCount,
        k == clusterCount - 1 ? sampleCount :
        (k + 1) * sampleCount / clusterCount);
      rng.fill(
        pointChunk, cv::RNG::RNG::NORMAL, cv::Scalar(center.x, center.y),
        cv::Scalar(input_img.cols * 0.05, input_img.rows * 0.05));
    }

    cv::randShuffle(points, 1, &rng);

    double compactness = cv::kmeans(
      points, clusterCount, labels,
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
      3, cv::KMEANS_PP_CENTERS, centers);

    input_img = cv::Scalar::all(0);

    for (i = 0; i < sampleCount; i++) {
      int clusterIdx = labels.at<int>(i);
      cv::Point ipt = points.at<cv::Point2f>(i);
      circle(input_img, ipt, 2, colorTab[clusterIdx], cv::FILLED, cv::LINE_AA);
    }

    for (i = 0; i < (int)k_center.size(); ++i) {
      cv::Point2f c = k_center[i];
      circle(input_img, c, k_radius[i], colorTab[i], 1, cv::LINE_AA);
    }

    std::cout << "Compactness: " << compactness << std::endl;
    imshow("clusters", input_img);

    char key = (char)cv::waitKey();
    if (key == 27 || key == 'q' || key == 'Q') {    // 'ESC'
      break;
    }
  }

}


cv::Mat image_processing(const cv::Mat in_image) 
{
  // Create output image
  cv::Mat out_image = in_image;
  clone_img = in_image.clone();
  option_trackbar = cv::getTrackbarPos(trackbar_name_1, window_name);

  // You must to return a 3-channels image to show it in ROS, so do it with 1-channel images
  //cv::cvtColor(out_image, out_image, cv::COLOR_GRAY2BGR);
  

  if ( option_trackbar == op_KEY_1) {
    person_detected(in_image);
    if(person == 1) 
    {
      out_image = balls_cv(out_image);
      out_image = project_lines(out_image);
      out_image = project_point_3D_2D(out_image);
      person_detected(in_image);
    }
  }

  if (option_trackbar == op_KEY_2) {
    out_image = balls_cv(out_image);
    k_means(clone_img);
    for( int i = 0; i < k_center.size(); i++){
      std::cout << "cennter x: "<< k_center[i].x << " center y: " << k_center[i].y << std::endl;
    }
    out_image = clone_img;
  }
  
  int key_pressed = cv::waitKey(200);
  person = 0;
  if (key_pressed == KEY_esc) {
    // Pulse Esc to close the windows.
    rclcpp::shutdown();
  }
  k_center.clear();
  k_radius.clear();
  cv::imshow(window_name, out_image);
  return out_image;
}

/**
  TO-DO
*/



pcl::PointXYZHSV rgb2hsv(const pcl::PointXYZRGB rgb_point)
{ // metodo para convertir un pointrgb a pointhsv.
    pcl::PointXYZHSV hsv_point;
    float r = rgb_point.r / 255.0;
    float g = rgb_point.g / 255.0;
    float b = rgb_point.b / 255.0;
    float cmax = std::max(r, std::max(g, b));
    float cmin = std::min(r, std::min(g, b));
    float delta = cmax - cmin;

    if (delta == 0)
    {
        hsv_point.h = 0;
    }
    else if (cmax == r)
    {
        hsv_point.h = fmod((g - b) / delta, 6.0);
    }
    else if (cmax == g)
    {
        hsv_point.h = (b - r) / delta + 2.0;
    }
    else
    {
        hsv_point.h = (r - g) / delta + 4.0;
    }

    hsv_point.h *= 60.0;
    if (hsv_point.h < 0)
    {
        hsv_point.h += 360.0;
    }

    if (cmax == 0)
    {
        hsv_point.s = 0;
    }
    else
    {
        hsv_point.s = delta / cmax;
    }

    hsv_point.v = cmax;

    hsv_point.x = rgb_point.x;
    hsv_point.y = rgb_point.y;
    hsv_point.z = rgb_point.z;

    return hsv_point;
}

pcl::PointCloud<pcl::PointXYZRGB> filterPink(const pcl::PointCloud<pcl::PointXYZRGB> cloud_in)
{ // filtramos el color de la esfera.
    pcl::PointCloud<pcl::PointXYZRGB> cloud_out_hsv;
    pcl::PointXYZHSV hsv_point;
    for (size_t i = 0; i < cloud_in.size(); ++i)
    {
        pcl::PointXYZRGB rgb_point = cloud_in.at(i);
        hsv_point = rgb2hsv(rgb_point);
        if (hsv_point.h >= 260 && hsv_point.h <= 330 && hsv_point.s > 0.1 && hsv_point.s > 0.1)
        {
            cloud_out_hsv.push_back(rgb_point);
        }
    }
    return cloud_out_hsv;
}

pcl::PointCloud<pcl::PointXYZRGB> outliers_filter(pcl::PointCloud<pcl::PointXYZRGB> cloud_in)
{ // Eliminanos los outliers.
    // Create the filtering object
    pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud (std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(cloud_in));
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (filtered_cloud);
    return filtered_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> draw_square(pcl::PointCloud<pcl::PointXYZRGB> cloud_in, float x_, float y_, float z_) 
{ // metodo para dibujar cuadrados en una nube de puntos cloud_in. 

  float size = 0.1;
  float step = 0.01;
  std::vector<pcl::PointXYZRGB> vertices;
  for (float i = -size/2; i <= size/2; i += step) {
      for (float j = -size/2; j <= size/2; j += step) {
          for (float k = -size/2; k <= size/2; k += step) {
              pcl::PointXYZRGB vertex(x_+i, y_+j, z_+k, 0, 0, 255);
              cloud_in.push_back(vertex);
          }
      }
  }
  return cloud_in;
}
pcl::PointCloud<pcl::PointXYZRGB> draw_square_dregaded(pcl::PointCloud<pcl::PointXYZRGB> cloud_in, float x_, float y_, float z_,int r, int g, int b) 
{ // metodo para dibujar cuadrados en el suelo del escenario. 

  float size = 0.1;
  float step = 0.01;
  std::vector<pcl::PointXYZRGB> vertices;
  for (float i = -size/2; i <= size/2; i += step) {
      for (float j = -size/2; j <= size/2; j += step) {
          for (float k = -size/2; k <= size/2; k += step) {
              pcl::PointXYZRGB vertex(x_+i, y_+j, z_+k, r, g, b);
              cloud_in.push_back(vertex);
          }
      }
  }
  return cloud_in;
}

void get_points(pcl::PointCloud<pcl::PointXYZRGB> cloud_in)
{
    // Create the filtering object
    pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud, cloud_f;
    pcl::PointXYZ position;
    

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_SPHERE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.01);

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;

    int i = 0, nr_points = (int) cloud_in.size ();
    // While 30% of the original cloud is still there
    while (cloud_in.size () > 0.01 * nr_points)
    {
      // Segment the largest planar component from the remaining cloud
      seg.setInputCloud (std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(cloud_in));
      seg.segment (*inliers, *coefficients);
      if (inliers->indices.size () == 0)
      {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        break;
      }

      // Extract the inliers
      extract.setInputCloud (std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(cloud_in));
      extract.setIndices (inliers);
      extract.setNegative (false);
      extract.filter (filtered_cloud);
      
      float sphere_center_x = coefficients->values[0];
      float sphere_center_y = coefficients->values[1];
      float sphere_center_z = coefficients->values[2];

      position.x = sphere_center_x;
      position.y = sphere_center_y;
      position.z = sphere_center_z;
      
      //std::cerr << "x: " << sphere_center_x  << ",y: "<< sphere_center_y << " ,z: "<< sphere_center_z << std::endl;
      
      square_pos.push_back(position);

      // Create the filtering object
      extract.setNegative (true);
      extract.filter (cloud_f);
      cloud_in.swap (cloud_f);

      i++;
    }
}

pcl::PointCloud<pcl::PointXYZRGB> calculate_cube_pos(pcl::PointCloud<pcl::PointXYZRGB> cloud_in, int distance)
{
  cv::Mat rot_tras;
  int r, g, b;
  rot_tras = (cv::Mat_<double>(4, 4) << 
    0.0, 1.0, 0.0, t.transform.translation.x,
    0.0, 0.0, 1.0, t.transform.translation.y,
    1.0, 0.0, 0.0, t.transform.translation.z,
    0.0, 0.0, 0.0, 1.0);

  if(distance  > 2) {

    for(int i = 0; i < distance - 2; i++){
    // guardo los puntos de cada distancia
    cv::Mat point1 = (cv::Mat_<double>(4,1) << i+3, 1, 0, 1.0);
    cv::Mat point2 = (cv::Mat_<double>(4,1) << i+3, -1, 0, 1.0);
    // realizo la roatcion y translación
    cv::Mat t_point1 = rot_tras * point1;
    cv::Mat t_point2 = rot_tras * point2;
    // obtengo los valores normalizados
    double x1 = t_point1.at<double>(0,0) / t_point1.at<double>(3,0);
    double y1 = t_point1.at<double>(1,0) / t_point1.at<double>(3,0);
    double z1 = t_point1.at<double>(2,0) / t_point1.at<double>(3,0);
    
    double x2 = t_point2.at<double>(0,0) / t_point2.at<double>(3,0);
    double y2 = t_point2.at<double>(1,0) / t_point2.at<double>(3,0);
    double z2 = t_point2.at<double>(2,0) / t_point2.at<double>(3,0);
    // calculo el color
    r = 255 - i * 42, g = i * 42, b = 0;
    // pinto el cubo
    cloud_in = draw_square_dregaded(cloud_in, x1, y1, z1, r, g ,b);
    cloud_in = draw_square_dregaded(cloud_in, x2, y2, z2, r, g, b);
    }
  }
  

  return cloud_in;
}



pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud = in_pointcloud;
  pcl::PointCloud<pcl::PointXYZRGB> filtered_pointcloud;
  pcl::PointCloud<pcl::PointXYZRGB> test;
  pcl::PointCloud<pcl::PointXYZRGB> temp_pointcloud = in_pointcloud;
  option_trackbar = cv::getTrackbarPos(trackbar_name_1, window_name);

  if (option_trackbar == op_KEY_0) {
    return temp_pointcloud;
  }
  if(option_trackbar == op_KEY_1) 
  {person_detected(clone_img);
    if (person == 1) 
    { 
      distance_trackbar = cv::getTrackbarPos(trackbar_name_2, window_name);
      out_pointcloud = filterPink(in_pointcloud);
      filtered_pointcloud = outliers_filter(out_pointcloud);
      get_points(filtered_pointcloud);

      for(size_t i = 0; i < square_pos.size(); i++) {

        filtered_pointcloud = draw_square(filtered_pointcloud, square_pos[i].x ,square_pos[i].y , square_pos[i].z);
        //std::cout <<"x: " << square_pos[i].x<<" y: " << square_pos[i].y << " z: "<< square_pos[i].z << std::endl;
      }
      filtered_pointcloud = calculate_cube_pos(filtered_pointcloud, distance_trackbar);
      project_points(filtered_pointcloud);
      for(size_t i = 0; i < black_square_pos.size(); i++) {

        filtered_pointcloud = draw_square_black(filtered_pointcloud, square_pos[i].x ,square_pos[i].y , black_square_pos[i].z);
      }
      
    }
    
    //std::cout <<"x: " << t.transform.translation.x <<"y: " << t.transform.translation.y << "z: "<< t.transform.translation.z << std::endl;
  }

  if ( option_trackbar == op_KEY_2){
    return temp_pointcloud;
  }
  square_pos_2 = square_pos;
  // limpiamos los vectores de puntos para evitar detecciones antiguas.
  square_pos.clear();
  black_square_pos.clear();
  point.clear();
  person = 0;
  return filtered_pointcloud;
}


int main(int argc, char * argv[])
{
  
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);   // Create window
  cv::createTrackbar(trackbar_name_1, window_name, nullptr,init_value_t1);
  cv::createTrackbar(trackbar_name_2, window_name, nullptr, init_value_t2);
  cv::setTrackbarPos(trackbar_name_2, window_name, min_set_value); 

  
  
  // Give the configuration and weight files for the model
  cv::String modelConfiguration = "/home/keist/vision_ws/src/vision/src/cfg/yolov3.cfg";
  cv::String modelWeights = "/home/keist/vision_ws/src/vision/src/cfg/yolov3.weights";
  net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
  net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
  
  
  
  rclcpp::init(argc, argv);

  rclcpp::executors::SingleThreadedExecutor exec;

  auto cv_node = std::make_shared<ComputerVisionSubscriber>();
  auto pcl_node = std::make_shared<PCLSubscriber>();
  exec.add_node(cv_node);
  exec.add_node(pcl_node);
  exec.spin();
  
  rclcpp::shutdown();
  return 0;
}