#include <opencv2/opencv.hpp>

using namespace cv;

bool isInside(const Mat &img, int i, int j) {
    return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

uchar linear_interpolation(int val, int in_min, int in_max, int out_min, int out_max) {
    return (uchar) (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

Mat_<uchar> median_filter(const Mat_<uchar> &img, const int &kernel_size) {
    Mat_<uchar> result_img(img.rows, img.cols);
    int k = kernel_size / 2;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            std::vector<uchar> temp;

            for (int l = 0; l < kernel_size; ++l) {
                for (int m = 0; m < kernel_size; ++m) {
                    if (isInside(img, i + l - k, j + m - k)) {
                        temp.push_back(img(i + l - k, j + m - k));
                    }
                }
            }

            std::sort(temp.begin(), temp.end());
            result_img(i, j) = temp[temp.size() / 2];
        }
    }

    return result_img;
}

uint64 **census_transform(const Mat_<uchar> &img, const int &window_size) {
    auto **result_matrix = new uint64 *[img.rows];
    for (int i = 0; i < img.rows; ++i) {
        result_matrix[i] = new uint64[img.cols];
    }

    int k = window_size / 2;
    uint64 bit;

    for (int i = k; i < img.rows - k; ++i) {
        for (int j = k; j < img.cols - k; ++j) {
            bit = 0;
            for (int l = 0; l < window_size; ++l) {
                for (int m = 0; m < window_size; ++m) {
                    if (l == i && m == j) {
                        continue;
                    }

                    bit <<= 1;
                    bit += img(i, j) >= img(i + l - k, j + m - k);
                }
            }
            result_matrix[i][j] = bit;
        }
    }

    return result_matrix;
}

int hamming_distance(uint64 n1, uint64 n2) {
    uint64 x = n1 ^n2;
    int setBits = 0;

    while (x > 0) {
        setBits += x & 1;
        x >>= 1;
    }

    return setBits;
}

int ***precompute_hamming(const Mat_<uchar> &img1, uint64 **census_left, uint64 **census_right) {
    int ***result_matrix = new int **[img1.rows];
    for (int i = 0; i < img1.rows; ++i) {
        result_matrix[i] = new int *[img1.cols];
        for (int j = 0; j < img1.cols; ++j) {
            result_matrix[i][j] = new int[51];
        }
    }

    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.cols; ++j) {
            for (int k = 0; k < 50; ++k) {
                if (j - k >= 0) {
                    result_matrix[i][j][k] = hamming_distance(census_left[i][j], census_right[i][j - k]);
                } else {
                    result_matrix[i][j][k] = 0;
                }
            }
        }
    }

    return result_matrix;
}

int ***sum_hamming_table(const Mat_<uchar> &img1, int ***hamming_table) {
    int ***result_matrix = new int **[img1.rows];
    int w = 5, sum;
    for (int i = 0; i < img1.rows; ++i) {
        result_matrix[i] = new int *[img1.cols];
        for (int j = 0; j < img1.cols; ++j) {
            result_matrix[i][j] = new int[51];
        }
    }

    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.cols; ++j) {
            for (int k = 0; k < 50; ++k) {
                if (j - k >= 0) {
                    sum = 0;

                    for (int l = i - w; l < i + w; ++l) {
                        if (l >= 0 && l < img1.rows) {
                            for (int m = j - w; m < j + w; ++m) {
                                if (m >= 0 && m < img1.cols) {
                                    sum += hamming_table[l][m][k];
                                }
                            }
                        }
                    }

                    result_matrix[i][j][k] = sum;
                }
            }
        }
    }

    return result_matrix;
}

Mat_<uchar> compute_disparity(const Mat_<uchar> &img1, uint64 **census_left, uint64 **census_right) {
    auto hamming_table = precompute_hamming(img1, census_left, census_right);
    auto aggregated_hamming_table = sum_hamming_table(img1, hamming_table);

    Mat_<uchar> result_img(img1.rows, img1.cols);
    int min_hamming_dist;
    uchar disparity;

    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.cols; ++j) {
            min_hamming_dist = INT_MAX;

            for (int k = 0; j - k >= 0 && k < 50; ++k) {
                int hamming_dist = aggregated_hamming_table[i][j][k];

                if (hamming_dist < min_hamming_dist) {
                    min_hamming_dist = hamming_dist;
                    disparity = k;
                }
            }

            result_img(i, j) = disparity;
        }
    }

    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.rows; ++j) {
            delete[] hamming_table[i][j];
        }

        delete[] hamming_table[i];
    }
    delete[] hamming_table;

    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.rows; ++j) {
            delete[] aggregated_hamming_table[i][j];
        }

        delete[] aggregated_hamming_table[i];
    }
    delete[] aggregated_hamming_table;

    return median_filter(result_img, 10);
}

float check_error(const Mat_<uchar> &result, const Mat_<uchar> &ground_truth) {
    int sum = 0, threshold = 2;
    ground_truth /= 4;

    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            if (abs(result(i, j) - ground_truth(i, j)) > threshold) {
                ++sum;
            }
        }
    }

    return (float) sum / (result.rows * result.cols);
}

int main() {
    Mat_<uchar> img1 = imread("Images/teddy2.png", IMREAD_GRAYSCALE);
    Mat_<uchar> img2 = imread("Images/teddy6.png", IMREAD_GRAYSCALE);
    Mat_<uchar> ground_truth = imread("Images/teddyDispL.png", IMREAD_GRAYSCALE);
    auto census_img1 = census_transform(img1, 8);
    auto census_img2 = census_transform(img2, 8);
    imshow("Original Image", img1);
    const Mat_<uchar> &disparity_map = compute_disparity(img1, census_img1, census_img2);
    imshow("Disparity Image", disparity_map);
    auto error = check_error(disparity_map, ground_truth);
    std::cout << "Error rate: " << error * 100 << "%";
    waitKey();
    for (int i = 0; i < img1.rows; ++i)
        delete[] census_img1[i];
    delete[] census_img1;
    for (int i = 0; i < img2.rows; ++i)
        delete[] census_img2[i];
    delete[] census_img2;
}
