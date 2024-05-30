#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "bm3d.h" //need to include after opencv
#include <stdio.h>

namespace py = pybind11;

py::array_t<unsigned char> denoise_wrapper(py::array_t<unsigned char> src_image){
    py::buffer_info src_buf = src_image.request();

    if (src_buf.ndim != 2) {
        throw std::runtime_error("Input should be a 2D NumPy array");
    }

    // Create a cv::Mat with the same shape and data as the NumPy array
    cv::Mat src_mat(src_buf.shape[0], src_buf.shape[1], CV_8UC1, (unsigned char*)src_buf.ptr);
    cv::Mat dst_mat = cv::Mat::zeros(src_buf.shape[0], src_buf.shape[1], CV_8UC1);
    Bm3d bm3d;

    std::cout << "hello from denoise wrapper"<<std::endl;

    // convert cv mat to numpy array for output
     py::buffer_info dst_buf(
        (unsigned char*)dst_mat.data,                        // Pointer to buffer
        sizeof(unsigned char),                  // Size of one scalar
        py::format_descriptor<unsigned char>::format(), // Python struct-style format descriptor
        2,                                      // Number of dimensions
        { dst_mat.rows, dst_mat.cols },                 // Buffer dimensions
        { dst_mat.step[0], dst_mat.step[1] }            // Strides (in bytes) for each index
    );
    
    return py::array_t<unsigned char>(dst_buf);
}

PYBIND11_MODULE(pyGpuBM3D, m) {
    m.doc() = "Python GPU BM3D denoising";
    // py::class_<Bm3d>(m, "Bm3d")
    // .def(py::init<>())
    m.def("denoise", &denoise_wrapper,
        R"pbdoc(
          GPU BM3D denoise function.

          Parameters
          ----------
          input : numpy.ndarray
              2D NumPy array of unsigned char (dtype=np.uint8).

          Returns
          -------
          numpy.ndarray
              Denoised 2D NumPy array with the same shape and dtype.
          )pbdoc"
    );
}