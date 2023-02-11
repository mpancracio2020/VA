// Copyright (c) 2022 Marvin Pancracio Manso
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


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


cv::Mat image_processing(const cv::Mat in_image);
cv::Mat image_processing_RGB(const cv::Mat in_image);
cv::Mat image_processing_CMY(const cv::Mat in_image);
cv::Mat image_processing_HSI_p2p(const cv::Mat in_image);
cv::Mat image_processing_HSV_max(const cv::Mat in_image);
cv::Mat image_processing_HSV(const cv::Mat in_image);
cv::Mat image_processing_HSI(const cv::Mat in_image);


class ComputerVisionSubscriber : public rclcpp::Node
{
public:
  ComputerVisionSubscriber()
  : Node("opencv_subscriber")
  {
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5));
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/rgb/image_raw", qos, std::bind(
        &ComputerVisionSubscriber::topic_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "cv_image", qos);
  }

private:
  void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
  {
    // Convert ROS Image to CV Image
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_raw = cv_ptr->image;

    // Image processing
    cv::Mat cv_image = image_processing(image_raw);

    // Convert OpenCV Image to ROS Image
    cv_bridge::CvImage img_bridge = cv_bridge::CvImage(
      msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
    sensor_msgs::msg::Image out_image;  // >> message to be sent
    img_bridge.toImageMsg(out_image);  // from cv_bridge to sensor_msgs::Image

    // Publish the data
    publisher_->publish(out_image);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

/**
  TO-DO
*/

cv::Mat image_processing_RGB(const cv::Mat in_image)
{  // Option 1 -- RGB.
  cv::Mat out_image = in_image;
  cv::imshow("out_image", in_image);
  return out_image;
}

cv::Mat image_processing_CMY(const cv::Mat in_image)
{
  cv::Mat out_image = in_image;
  for (int i = 0; i < out_image.rows; i++) {
    for (int j = 0; j < out_image.cols; j++) {
      out_image.at<cv::Vec3b>(i, j)[0] = 255 - (uint)out_image.at<cv::Vec3b>(i, j)[0];
      out_image.at<cv::Vec3b>(i, j)[1] = 255 - (uint)out_image.at<cv::Vec3b>(i, j)[1];
      out_image.at<cv::Vec3b>(i, j)[2] = 255 - (uint)out_image.at<cv::Vec3b>(i, j)[2];
    }
  }
  cv::imshow("out_image", out_image);
  return out_image;
}

cv::Mat image_processing_HSI_p2p(const cv::Mat in_image)
{
  cv::Mat out_image = in_image;
  double b, g, r, s, in, h;
  for (int i = 0; i < out_image.rows; i++) {
    for (int j = 0; j < out_image.cols; j++) {
      // You can now access the pixel value with cv::Vec3b

      b = out_image.at<cv::Vec3b>(i, j)[0];
      g = out_image.at<cv::Vec3b>(i, j)[1];
      r = out_image.at<cv::Vec3b>(i, j)[2];

      double b_n = b / 255; double g_n = g / 255; double r_n = r / 255;

      in = ((b_n + g_n + r_n) / (3.0));
      s = 1.0 - ((3.0 / (b_n + g_n + r_n)) * (std::min(r_n, std::min(b_n, g_n))));
      double num = 0.5 * ((r_n - g_n) + (r_n - b_n));
      double den = sqrt((r_n - b_n) * (r_n - b_n) + (r_n - b_n) * (g_n - b_n));

      h = num / den;
      h = acos(h);
      h = 180 * h / 3.14159265;

      if (b > g) {
        h = 360 - h;
      }

      out_image.at<cv::Vec3b>(i, j)[0] = (h / 360.0) * 255.0;
      out_image.at<cv::Vec3b>(i, j)[1] = s * 255.0;
      out_image.at<cv::Vec3b>(i, j)[2] = in * 255.0;
    }
  }
  cv::imshow("out_image", out_image);
  return out_image;
}
cv::Mat image_processing_HSV_max(const cv::Mat in_image)
{
  cv::Mat out_image = in_image;
  double b, g, r, s, v, h;
  for (int i = 0; i < out_image.rows; i++) {
    for (int j = 0; j < out_image.cols; j++) {
      b = out_image.at<cv::Vec3b>(i, j)[0];
      g = out_image.at<cv::Vec3b>(i, j)[1];
      r = out_image.at<cv::Vec3b>(i, j)[2];

      double b_n = b / 255; double g_n = g / 255; double r_n = r / 255;

      v = std::max(r_n, std::max(g_n, b_n));
      s = 1.0 - ((3.0 / (b_n + g_n + r_n)) * (std::min(r_n, std::min(b_n, g_n))));
      double num = 0.5 * ((r_n - g_n) + (r_n - b_n));
      double den = sqrt((r_n - g_n) * (r_n - g_n) + (r_n - b_n) * g_n - r_n);
      h = num / den;
      h = acos(h);
      h = 180 * h / 3.14159265;

      if (b > g) {
        h = 360 - h;
      }
      out_image.at<cv::Vec3b>(i, j)[0] = (h / 360.0) * 255.0;
      out_image.at<cv::Vec3b>(i, j)[1] = s * 255.0;
      out_image.at<cv::Vec3b>(i, j)[2] = v * 255.0;
    }
  }
  cv::imshow("out_image", out_image);
  return out_image;
}

cv::Mat image_processing_HSV(const cv::Mat in_image)
{
  cv::Mat out_image = in_image;
  cvtColor(in_image, out_image, CV_BGR2HSV);
  cv::imshow("out_image", out_image);
  return out_image;
}
cv::Mat image_processing_HSI(const cv::Mat in_image)
{
  std::vector<cv::Mat> three_channels;  // canales BGR
  cv::split(in_image, three_channels);

  cv::Mat temp_image;
  cv::Mat out_image = in_image;

  double b, g, r, in;

  for (int i = 0; i < in_image.rows; i++) {
    for (int j = 0; j < in_image.cols; j++) {
      // You can now access the pixel value with cv::Vec3b

      b = three_channels[0].at<uchar>(i, j);
      g = three_channels[1].at<uchar>(i, j);
      r = three_channels[2].at<uchar>(i, j);

      double b_n = b / 255; double g_n = g / 255; double r_n = r / 255;

      in = ((b_n + g_n + r_n) / (3.0));

      three_channels[2].at<uchar>(i, j) = in * 255.0;
    }
  }

  cvtColor(in_image, temp_image, CV_BGR2HSV);
  std::vector<cv::Mat> three_channels_HSI;  // canales HSI
  cv::split(temp_image, three_channels_HSI);

  std::vector<cv::Mat> channels;
  channels.push_back(three_channels_HSI[0]);
  channels.push_back(three_channels_HSI[1]);
  channels.push_back(three_channels[2]);

  merge(channels, out_image);
  cv::imshow("out_image", out_image);
  return out_image;
}

cv::Mat image_processing(const cv::Mat in_image)
{
  // Create output image
  cv::Mat out_image;
  // Processing
  out_image = in_image;
  std::string text = (
    "1: RGB, 2: CMY, 3: HSI, 4: HSV, 5: HSV OpenCV, 6: HSI OpenCv");
  cv::putText(
    out_image, text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

  int c = cv::waitKey(0);

  std::cout << c << std::endl;

  if (c == 49) {
    // Option 1 -- BGR.
    image_processing_RGB(in_image);
  } else if (c == 50) {
    // Option 2 -- CMY filter.
    image_processing_CMY(in_image);
  } else if (c == 51) {
    // Option 3 --- HSI pix2pix.
    image_processing_HSI_p2p(in_image);
  } else if (c == 52) {
    // Option 4 --- HSV pix2pix.
    image_processing_HSV_max(in_image);
  } else if (c == 53 || c == -75) {
    // Option 5 -- HSV OpenCV.
    image_processing_HSV(in_image);
  } else if (c == 54) {
    // Option 6.
    image_processing_HSI(in_image);
  } else if (c == 27) {
    // Pulse Esc to close the windows.
    rclcpp::shutdown();
  }

  // Show image in a different window
  // You must to return a 3-channels image to show it in ROS, so do it with 1-channel images
  cv::imshow("out_image", out_image);
  // cv::cvtColor(out_image, out_image, cv::COLOR_GRAY2BGR);
  return out_image;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}
