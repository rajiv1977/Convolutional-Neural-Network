#pragma once

#include <algorithm>
#include <array>
#include <any>
#include <cmath>              // For mathematical constants like M_PI
#include <filesystem>         // Include filesystem header for directory handling
#include <iterator>
#include <iomanip> // Include this for std::setprecision
#include <map>
#include <numbers>            // For std::numbers::pi in C++20
#include <opencv2/opencv.hpp> // Include OpenCV header for image loading
#include <random>             // For generating random numbers
#include <string>             // Include string header for using std::string
#include <unordered_set>
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

    /// @brief Prints detailed information of a cv::Mat, including image dimensions, channels, and pixel values.
    /// @param mat The cv::Mat object to be printed.
    void PrintMat(const cv::Mat& mat)
    {
        // Print matrix dimensions and channel count
        std::cout << "Image Size: " << mat.rows << "x" << mat.cols << "\t"
                  << "Channels: " << mat.channels() << "\nMatrix Depth: ";

        // Identify and print the matrix depth type
        switch (mat.depth())
        {
            case CV_8U:
                std::cout << "8-bit unsigned integer (CV_8U)";
                break;
            case CV_8S:
                std::cout << "8-bit signed integer (CV_8S)";
                break;
            case CV_16U:
                std::cout << "16-bit unsigned integer (CV_16U)";
                break;
            case CV_16S:
                std::cout << "16-bit signed integer (CV_16S)";
                break;
            case CV_32S:
                std::cout << "32-bit signed integer (CV_32S)";
                break;
            case CV_32F:
                std::cout << "32-bit floating point (CV_32F)";
                break;
            case CV_64F:
                std::cout << "64-bit floating point (CV_64F)";
                break;
            default:
                std::cout << "Unknown";
                break;
        }
        std::cout << "\n" << std::endl;

        // Check if the matrix has 3 channels (assumed as BGR)
        if (mat.channels() == 3)
        {
            // Print each pixel's BGR values for color images
            for (int i = 0; i < mat.rows; i++)
            {
                for (int j = 0; j < mat.cols; j++)
                {
                    cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j); // Retrieve BGR pixel values
                    std::cout << "[B:" << (int) pixel[0] << ", G:" << (int) pixel[1] << ", R:" << (int) pixel[2]
                              << "] ";
                }
                std::cout << "\n"; // New line after each row
            }
        }
        else
        {
            // Print each pixel value for grayscale or single-channel images
            for (int i = 0; i < mat.rows; ++i)
            {
                for (int j = 0; j < mat.cols; ++j)
                {
                    // Retrieve and print each pixel based on matrix depth
                    switch (mat.depth())
                    {
                        case CV_8U:
                            std::cout << static_cast<int>(mat.at<uchar>(i, j)) << " ";
                            break;
                        case CV_8S:
                            std::cout << static_cast<int>(mat.at<char>(i, j)) << " ";
                            break;
                        case CV_16U:
                            std::cout << mat.at<ushort>(i, j) << " ";
                            break;
                        case CV_16S:
                            std::cout << mat.at<short>(i, j) << " ";
                            break;
                        case CV_32S:
                            std::cout << mat.at<int>(i, j) << " ";
                            break;
                        case CV_32F:
                            std::cout << mat.at<float>(i, j) << " ";
                            break;
                        case CV_64F:
                            std::cout << mat.at<double>(i, j) << " ";
                            break;
                        default:
                            std::cout << "Unsupported type!" << std::endl;
                            return;
                    }
                }
                std::cout << "\n"; // New line after each row
            }
        }
        std::cout << std::endl; // Final newline for spacing
    }

    /// @brief Helper function to scale pixel values based on the cv::Mat data type.
    /// @param image The input image in OpenCV Mat format.
    /// @param x The x-coordinate of the pixel.
    /// @param y The y-coordinate of the pixel.
    /// @param channel The color channel to access (0 for first channel, 1 for second channel, etc.).
    /// @return A float representing the scaled pixel value.
    float scalePixelValue(const cv::Mat& image, int x, int y, int channel)
    {
        // Check if the pixel coordinates are within the image boundaries
        if (x < 0 || x >= image.cols || y < 0 || y >= image.rows)
        {
            throw std::out_of_range("Pixel coordinates are out of bounds.");
        }

        // Check if the channel index is valid
        if (channel < 0 || channel >= image.channels())
        {
            throw std::out_of_range("Invalid channel index.");
        }

        switch (image.depth())
        {
            case CV_8U: // unsigned 8-bit (0 to 255)
            {
                if (image.channels() == 1) // Grayscale image
                    return image.at<uchar>(y, x) / 255.0f;
                else if (image.channels() == 3) // Color image
                    return image.at<cv::Vec3b>(y, x)[channel] / 255.0f;
                break;
            }
            case CV_8S: // signed 8-bit (-128 to 127)
            {
                if (image.channels() == 1) // Grayscale image
                    return (image.at<char>(y, x) + 128) / 255.0f;
                else if (image.channels() == 3) // Color image
                    return (image.at<cv::Vec3b>(y, x)[channel] + 128) / 255.0f;
                break;
            }
            case CV_16U: // unsigned 16-bit (0 to 65535)
            {
                if (image.channels() == 1) // Grayscale image
                    return image.at<ushort>(y, x) / 65535.0f;
                else if (image.channels() == 3) // Color image
                    return image.at<cv::Vec3w>(y, x)[channel] / 65535.0f;
                break;
            }
            case CV_16S: // signed 16-bit (-32768 to 32767)
            {
                if (image.channels() == 1) // Grayscale image
                    return (image.at<short>(y, x) + 32768) / 65535.0f;
                else if (image.channels() == 3) // Color image
                    return (image.at<cv::Vec3s>(y, x)[channel] + 32768) / 65535.0f;
                break;
            }
            case CV_32S: // signed 32-bit
            {
                if (image.channels() == 1) // Grayscale image
                    return image.at<int>(y, x) / static_cast<float>(std::numeric_limits<int>::max());
                else if (image.channels() == 3) // Color image
                    return image.at<cv::Vec3i>(y, x)[channel] / static_cast<float>(std::numeric_limits<int>::max());
                break;
            }
            case CV_32F: // 32-bit floating point
            {
                if (image.channels() == 1) // Grayscale image
                    return image.at<float>(y, x);
                else if (image.channels() == 3) // Color image
                    return image.at<cv::Vec3f>(y, x)[channel];
                break;
            }
            case CV_64F: // 64-bit floating point
            {
                if (image.channels() == 1) // Grayscale image
                    return static_cast<float>(image.at<double>(y, x));
                else if (image.channels() == 3) // Color image
                    return static_cast<float>(image.at<cv::Vec3d>(y, x)[channel]);
                break;
            }
            default:
                throw std::runtime_error("Unsupported image depth.");
        }

        // If no return was made, throw an error
        throw std::runtime_error("Failed to scale pixel value.");
    }

    /// @brief Function to format a single image for input into a CNN model.
    /// @param image The image to be formatted.
    /// @param height The desired height for the output image.
    /// @param width The desired width for the output image.
    /// @param numChannels The number of channels in the output image (1 for grayscale, 3 for color).
    /// @param className The label of the image's class.
    /// @param verbose If true, prints image type and dimensions for debugging.
    /// @return A FormattedData struct containing the formatted image data and label.
    FormattedData FormatImagesForCNN(const cv::Mat&     image,
                                     int                height,
                                     int                width,
                                     int                numChannels,
                                     const std::string& className,
                                     bool               verbose = false)
    {
        if (verbose)
        {
            PrintMat(image);
        }

        // Initialize a 3D vector to store the formatted image data with the desired dimensions
        std::vector<std::vector<std::vector<float>>> data(
            numChannels, std::vector<std::vector<float>>(height, std::vector<float>(width)));

        // Initialize label vector for one-hot encoding
        std::vector<float>         label(10, 0.0f); // 10 classes
        std::map<std::string, int> classMap = {{"Bikes", 0},
                                               {"Buffalo", 1},
                                               {"Cars", 2},
                                               {"Elephant", 3},
                                               {"Motorcycles", 4},
                                               {"Planes", 5},
                                               {"Rhino", 6},
                                               {"Ships", 7},
                                               {"Trains", 8},
                                               {"Zebra", 9}};

        auto it = classMap.find(className);
        if (it != classMap.end())
        {
            label[it->second] = 1.0f;
        }

        // Iterate over pixels and scale them according to their data type
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (numChannels == 1)
                {
                    // For grayscale images, use the first channel
                    data[0][y][x] = scalePixelValue(image, x, y, 0); // x, y for grayscale
                }
                else
                {
                    // For RGB images, store the R, G, and B channels separately
                    data[0][y][x] = scalePixelValue(image, x, y, 2); // Red
                    data[1][y][x] = scalePixelValue(image, x, y, 1); // Green
                    data[2][y][x] = scalePixelValue(image, x, y, 0); // Blue
                }
            }
        }

        // Store the image data and label in the FormattedData struct
        return {data, label};
    }

    /// @brief Function to get a specified number of image file names from a specific folder and load the images.
    /// @return Vector of loaded images and corresponding labels.
    ALLFormattedData GetImagesFromFolder()
    {
        // Load necessary settings
        const auto height      = mSettings.ResizeFactor[0];
        const auto width       = mSettings.ResizeFactor[1];
        const auto numChannels = (mSettings.ConvertToGray == true) ? 1 : 3;

        // Initialize data containers
        std::vector<std::vector<std::vector<std::vector<float>>>> data;
        std::vector<std::vector<float>>                           labels;

        // Set to track loaded file paths to avoid duplicates
        std::unordered_set<std::string> loadedFiles;

        for (const auto& folderName : GetFolderNames())
        {
            std::string folderPath = mDatasetPath + "//" + folderName;

            // Check if folder exists
            if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath))
            {
                std::cerr << "Warning: Folder " << folderPath << " does not exist or is not a directory.\n";
                continue;
            }

            // Iterate through files in the folder
            for (const auto& entry : std::filesystem::directory_iterator(folderPath))
            {
                // Ensure it's a regular file (not a directory or special file)
                if (!entry.is_regular_file())
                {
                    continue;
                }

                // Get the file path as a string
                std::string filePath = entry.path().string();

                // Skip the file if it has already been loaded
                if (loadedFiles.find(filePath) != loadedFiles.end())
                {
                    continue;
                }

                // Attempt to load the image
                cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
                if (image.empty()) // Check if image was successfully loaded
                {
                    std::cerr << "Warning: Failed to load image at " << filePath << "\n";
                    continue;
                }

                // Preprocess the image
                image = PreprocessingData(image);

                ViewImage(image, "Cross Hair", 3000);

                // Format image data for CNN model if preprocessing succeeded
                FormattedData cnnData = FormatImagesForCNN(image, height, width, numChannels, folderName, false);
                data.push_back(cnnData.images);
                labels.push_back(cnnData.labels);

                // Mark the file as loaded
                loadedFiles.insert(filePath);
            }
        }
        return {data, labels};
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

