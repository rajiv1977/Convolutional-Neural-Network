#pragma once

#include <array>
#include <cmath>              // For mathematical constants like M_PI
#include <filesystem>         // Include filesystem header for directory handling
#include <numbers>            // For std::numbers::pi in C++20
#include <opencv2/opencv.hpp> // Include OpenCV header for image loading
#include <random>             // For generating random numbers
#include <string>             // Include string header for using std::string
#include <vector>             // Include vector header for using std::vector

/// @brief Struct to hold settings for image processing.
struct ImageProcessingSettings
{
    bool                  ConvertToGray; // Flag to convert images to grayscale
    bool                  Resize;        // Flag to indicate if resizing is needed
    std::array<size_t, 2> ResizeFactor;  // Downscale factors for width and height
    bool                  Rotate;        // Flag to indicate if rotation is needed
    size_t                numClasses;    // Number of classes or features

    /// @brief Constructor to initialize default settings.
    ImageProcessingSettings()
        : ConvertToGray(false)
        , Resize(false)
        , ResizeFactor({1, 1})
        , Rotate(false)
        , numClasses(10)
    {
    }
};

// @brief Struct to hold formatted image data and corresponding labels.
struct FormattedData
{
    std::vector<std::vector<std::vector<float>>> images; // 3D vector for image data
    std::vector<float>                           labels; // 1D vector for labels
};

// @brief Struct to hold formatted CNN image data and corresponding labels.
struct ALLFormattedData
{
    std::vector<std::vector<std::vector<std::vector<float>>>> images; // 4D vector for image data
    std::vector<std::vector<float>>                           labels; // 2D vector for labels
};

/// @brief Class to load animal images for classification.
class ImageLoader
{
  public:
    /// @brief Constructor to initialize the dataset path.
    /// @param datasetPath Path to the dataset folder.
    explicit ImageLoader(const std::string& datasetPath)
        : mDatasetPath(datasetPath)
        , mSettings{}
        , mNumSamples(0)
    {
    }

    /// @brief Updates the settings based on the provided values.
    /// @param convertToGray New value for ConvertToGray setting.
    /// @param resize New value for Resize setting.
    /// @param resizeFactor New values for ResizeFactor setting.
    /// @param rotate New value for Rotate setting.
    void UpdateSettings(bool                         convertToGray,
                        bool                         resize,
                        const std::array<size_t, 2>& resizeFactor,
                        bool                         rotate,
                        size_t                       numClasses)
    {
        mSettings.ConvertToGray = convertToGray;
        mSettings.Resize        = resize;
        mSettings.ResizeFactor  = resizeFactor;
        mSettings.Rotate        = rotate;
        mSettings.numClasses    = numClasses;
    }

    /// @brief Function to get all folder names in a specific directory.
    /// @return Vector of folder names.
    std::vector<std::string> GetFolderNames()
    {
        std::vector<std::string> folderNames; // Vector to store folder names

        // Iterate through the directory
        for (const auto& entry : std::filesystem::directory_iterator(mDatasetPath))
        {
            if (entry.is_directory())
            {
                // Check if the entry is a directory
                folderNames.push_back(entry.path().filename().string()); // Get the folder name and add to the vector
                mNumSamples += CountImageFiles(mDatasetPath + "//" + entry.path().filename().string());
            }
        }

        return folderNames; // Return the list of folder names
    }

    void PrintMat(const cv::Mat& mat)
    {
        std::cout << "Image Size: " << mat.rows << "x" << mat.cols << std::endl;
        for (int i = 0; i < mat.rows; i++)
        {
            for (int j = 0; j < mat.cols; j++)
            {
                cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j); // Get BGR pixel values
                std::cout << "[B: " << (int) pixel[0] << ", G: " << (int) pixel[1] << ", R: " << (int) pixel[2] << "] ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    // @brief Function to format a single image for input into a CNN model.
    // @param image The image to be formatted.
    // @param height The height of the output image.
    // @param width The width of the output image.
    // @param numChannels The number of channels in the output image (1 for grayscale, 3 for color).
    // @param className The class name (label) for the image.
    // @return A FormattedData struct containing the formatted image data and its label.
    FormattedData
        FormatImagesForCNN(const cv::Mat& image, int height, int width, int numChannels, const std::string& className)
    {
        // Since it is RGB 
        height = height / 3;
        width  = width / 3;

        // Initialize the 3D vector to hold formatted image data for one sample
        std::vector<std::vector<std::vector<float>>> data(
            numChannels, std::vector<std::vector<float>>(height, std::vector<float>(width)));

        // Initialize the label vector with zeros (one-hot encoding for multiple classes)
        std::vector<float> label(mSettings.numClasses, 0.0f); // classes: Buffalo, Elephant, Rhino, Zebra

        // Assign the appropriate label based on the class name
        if (className == "Bikes")
        {
            label[0] = 1;
        }
        else if (className == "Buffalo")
        {
            label[1] = 1;
        }
        else if (className == "Cars")
        {
            label[2] = 1;
        }
        else if (className == "Elephant")
        {
            label[3] = 1;
        }
        else if (className == "Motorcycles")
        {
            label[4] = 1;
        }
        else if (className == "Planes")
        {
            label[5] = 1;
        }
        else if (className == "Rhino")
        {
            label[6] = 1;
        }
        else if (className == "Ships")
        {
            label[7] = 1;
        }
        else if (className == "Trains")
        {
            label[8] = 1;
        }
        else if (className == "Zebra")
        {
            label[9] = 1;
        }

        // ViewImage(image, "Classifications", 2000);

        for (int y = 0; y < static_cast<int>(image.rows / 3); ++y)
        {
            for (int x = 0; x < static_cast<int>(image.cols / 3); ++x)
            {
                cv::Vec3b pixel = image.at<cv::Vec3b>(y, x); // Get pixel value
                if (numChannels == 1)
                {
                    // If grayscale, use the average of RGB values
                    float grayValue = (pixel[0] + pixel[1] + pixel[2]) / 3.0f / 255.0f; // Normalize to [0, 1]
                    data[0][y][x]   = grayValue;                                        // Assign to the first channel
                }
                else
                {
                    // If color, assign R, G, B channels separately
                    data[0][y][x] = pixel[2] / 255.0f; // Red channel
                    data[1][y][x] = pixel[1] / 255.0f; // Green channel
                    data[2][y][x] = pixel[0] / 255.0f; // Blue channel
                }
            }
        }

        // Create the formatted data struct to hold images and labels
        FormattedData formattedData;
        formattedData.images = data;  // Assign formatted image data
        formattedData.labels = label; // Assign label

        return formattedData; // Return the FormattedData struct
    }

    /// @brief Function to get a specified number of image file names from a specific folder and load the images.
    /// @param folderPath Path to the folder containing image files.
    /// @param numImages The maximum number of images to load.
    /// @return Vector of loaded images.
    ALLFormattedData GetImagesFromFolder(size_t numImages)
    {
        size_t     index       = 0;
        const auto height      = mSettings.ResizeFactor[0];
        const auto width       = mSettings.ResizeFactor[1];
        const auto numChannels = (mSettings.ConvertToGray == true) ? 1 : 3;

        // standard needed by CNN Model
        std::vector<std::vector<std::vector<std::vector<float>>>> data = {};

        // Generate dummy labels (one-hot encoded for number of classes)
        std::vector<std::vector<float>> labels = {};

        for (const auto names : GetFolderNames())
        {
            std::string folderPath = mDatasetPath + "//" + names;

            size_t id = 0;
            // Iterate through files in the folder
            for (const auto& entry : std::filesystem::directory_iterator(folderPath))
            {
                if (entry.is_regular_file()) // Check if entry is a file (not a directory)
                {
                    std::string filePath = entry.path().string(); // Get the file path as a string
                    cv::Mat     image    = cv::imread(filePath);  // Load the image using OpenCV
                    image                = PreprocessingData(image);

                    if (!image.empty()) // Check if the image was successfully loaded
                    {
                        FormattedData cnnData = FormatImagesForCNN(image, height, width, numChannels, names);
                        data.push_back(cnnData.images);
                        labels.push_back(cnnData.labels);
                    }

                    // Stop loading if the desired number of images is reached
                    if (id >= numImages)
                    {
                        break;
                    }
                    id++;
                }
            }
        }

        return {data, labels}; // Return the vector of loaded images
    }

    /// @brief Function to count the number of image files with a specific extension in a folder.
    /// @param folderPath Path to the folder.
    /// @param fileExtension Desired image file extension (e.g., ".jpg", ".png").
    /// @return Number of image files with the specified extension in the folder.
    int CountImageFiles(const std::string& folderPath, const std::string& fileExtension = ".jpg")
    {
        int count = 0; // Initialize file count

        // Iterate through the directory
        for (const auto& entry : std::filesystem::directory_iterator(folderPath))
        {
            if (entry.is_regular_file() &&
                entry.path().extension() == fileExtension) // Check if the file has the desired extension
            {
                ++count; // Increment count if it's a file with the specified extension
            }
        }

        return count; // Return the total number of image files
    }

    /// @brief Function to display an image in a window.
    /// @param image The image to display.
    /// @param windowName The name of the window to create for displaying the image.
    /// @param displayTime The time in milliseconds to display the image before moving to the next one.
    void ViewImage(const cv::Mat& image, const std::string& windowName, int displayTime)
    {
        if (image.empty()) // Check if the image is empty
        {
            std::cerr << "Error: Unable to display the image. The image is empty." << std::endl;
            return;
        }

        cv::imshow(windowName, image); // Show the image in a window
        cv::waitKey(displayTime);      // Wait for the specified time (milliseconds)
    }

  private:
    std::string             mDatasetPath; // Path to the dataset folder
    ImageProcessingSettings mSettings;    // Setting for input data processing
    size_t                  mNumSamples;

    /// @brief Processes the image based on settings.
    /// @param image The original image.
    /// @return The processed image.
    cv::Mat PreprocessingData(const cv::Mat& image)
    {
        cv::Mat processedImage = image; // Start with the original image

        if (mSettings.ConvertToGray)
        {
            processedImage = ConvertToGrayscale(processedImage); // Convert to grayscale
        }

        if (mSettings.Resize)
        {
            processedImage =
                ResizeImage(processedImage, mSettings.ResizeFactor[0], mSettings.ResizeFactor[1]); // Resize image
        }

        if (mSettings.Rotate)
        {
            processedImage = RotateImage(processedImage); // Rotate image
        }

        return processedImage; // Return processed image
    }

    /// @brief Converts the image to grayscale.
    /// @param image The original image.
    /// @return The grayscale image.
    cv::Mat ConvertToGrayscale(const cv::Mat& image)
    {
        cv::Mat grayImage;                                  // Matrix to store the grayscale image
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY); // Convert to grayscale
        return grayImage;                                   // Return grayscale image
    }

    /// @brief Resizes the image to the specified width and height.
    /// @param image The image to resize.
    /// @param width The target width for the resized image.
    /// @param height The target height for the resized image.
    /// @return The resized image.
    cv::Mat ResizeImage(const cv::Mat& image, int width, int height)
    {
        cv::Mat resizedImage; // Matrix to store the resized image

        // Resize image to the specified width and height
        cv::resize(image, resizedImage, cv::Size(width, height));

        return resizedImage; // Return resized image
    }

    /// @brief Rotates the image by a random angle between 0 and 2π.
    /// @param image The image to rotate.
    /// @return The rotated image.
    cv::Mat RotateImage(const cv::Mat& image)
    {
        // Generate a random rotation angle between 0 and 2π (0 to 360 degrees)
        std::random_device               rd;                            // Initialize random device
        std::mt19937                     gen(rd());                     // Seed the generator
        std::uniform_real_distribution<> dist(0, 2 * std::numbers::pi); // Uniform distribution between 0 and 2π
        float                            angle = dist(gen) * (180.0 / std::numbers::pi); // Convert radians to degrees

        cv::Mat   rotatedImage;                                                 // Matrix to store the rotated image
        cv::Point center(image.cols / 2, image.rows / 2);                       // Compute the center of the image
        cv::Mat   rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0); // Get rotation matrix
        cv::warpAffine(image, rotatedImage, rotationMatrix, image.size());      // Apply rotation
        return rotatedImage;                                                    // Return rotated image
    }
};

