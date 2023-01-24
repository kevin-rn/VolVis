#include "volume.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype> // isspace
#include <chrono>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <gsl/span>
#include <iostream>
#include <string>

struct Header {
    glm::ivec3 dim;
    size_t elementSize;
};
static Header readHeader(std::ifstream& ifs);
static float computeMinimum(gsl::span<const uint16_t> data);
static float computeMaximum(gsl::span<const uint16_t> data);
static std::vector<int> computeHistogram(gsl::span<const uint16_t> data);

namespace volume {

Volume::Volume(const std::filesystem::path& file)
    : m_fileName(file.string())
{
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    loadFile(file);
    auto end = clock::now();
    std::cout << "Time to load: " << std::chrono::duration<double, std::milli>(end - start).count() << "ms" << std::endl;

    if (m_data.size() > 0) {
        m_minimum = computeMinimum(m_data);
        m_maximum = computeMaximum(m_data);
        m_histogram = computeHistogram(m_data);
    }
}

Volume::Volume(std::vector<uint16_t> data, const glm::ivec3& dim)
    : m_fileName()
    , m_elementSize(2)
    , m_dim(dim)
    , m_data(std::move(data))
    , m_minimum(computeMinimum(m_data))
    , m_maximum(computeMaximum(m_data))
    , m_histogram(computeHistogram(m_data))
{
}

float Volume::minimum() const
{
    return m_minimum;
}

float Volume::maximum() const
{
    return m_maximum;
}

std::vector<int> Volume::histogram() const
{
    return m_histogram;
}

glm::ivec3 Volume::dims() const
{
    return m_dim;
}

std::string_view Volume::fileName() const
{
    return m_fileName;
}

float Volume::getVoxel(int x, int y, int z) const
{
    const size_t i = size_t(x + m_dim.x * (y + m_dim.y * z));
    return static_cast<float>(m_data[i]);
}

// This function returns a value based on the current interpolation mode
float Volume::getSampleInterpolate(const glm::vec3& coord) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getSampleNearestNeighbourInterpolation(coord);
    }
    case InterpolationMode::Linear: {
        return getSampleTriLinearInterpolation(coord);
    }
    case InterpolationMode::Cubic: {
        return getSampleTriCubicInterpolation(coord);
    }
    default: {
        throw std::exception();
    }
    }
}

// This function returns the nearest neighbour value at the continuous 3D position given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
float Volume::getSampleNearestNeighbourInterpolation(const glm::vec3& coord) const
{
    // check if the coordinate is within volume boundaries, since we only look at direct neighbours we only need to check within 0.5
    if (glm::any(glm::lessThan(coord + 0.5f, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord + 0.5f, glm::vec3(m_dim))))
        return 0.0f;

    // nearest neighbour simply rounds to the closest voxel positions
    auto roundToPositiveInt = [](float f) {
        // rounding is equal to adding 0.5 and cutting off the fractional part
        return static_cast<int>(f + 0.5f);
    };

    return getVoxel(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// This function returns the trilinear interpolated value at the continuous 3D position given by coord.
// A trilinear interpolation is identical to performing a linear interpolation on the outcome of two bilinear interpolations. 
float Volume::getSampleTriLinearInterpolation(const glm::vec3& coord) const
{
    // check if the coordinate is within volume boundaries, we only need to check within distance 1
    // as we use the box coordinates in bilinear interpolation that are at most distance 1.
    if (glm::any(glm::lessThan(coord - 1.0f, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord + 1.0f, glm::vec3(m_dim))))
        return 0.0f;

    // Create new vector coordinate for X and Y.
    const glm::vec2 xyCoord = glm::vec2(coord.x, coord.y); 
    
    // As to interpolate between two z-planes we floor z and for the other plane add 1 to it.
    // Floor the value for Z (use static_cast<int> instead of floor as it already floors the value and changes it to an int).
    const int zfloor = static_cast<int>(coord.z);

    // Interpolate X and Y coordinates for each of the two z-planes.
    float z0 = biLinearInterpolate(xyCoord, zfloor);
    float z1 = biLinearInterpolate(xyCoord, zfloor + 1);

    // Interpolate two resulting values along Z (Use the decimal part of z value as factor). 
    return linearInterpolate(z0, z1, coord.z - static_cast<float>(zfloor));
}

// This function linearly interpolates the value at X using incoming values g0 and g1 given a factor (equal to the positon of x in 1D)
//
// g0--X--------g1
//   factor
float Volume::linearInterpolate(float g0, float g1, float factor)
{
        return (1.0f - factor) * g0 + factor * g1;
}

// This function bi-linearly interpolates the value at the given continuous 2D XY coordinate for a fixed integer z coordinate.
float Volume::biLinearInterpolate(const glm::vec2& xyCoord, int z) const
{
    // Calculate the 'box' coordinates that contain a given coordinate for separating interpolation per axis.
    // e.g.     (x1, y2)-----------(x2, y2)
    //          -----------(x, y)----------
    //          (x1, y1)-----------(x2, y1)
    const int x1 = static_cast<int>(xyCoord.x), y1 = static_cast<int>(xyCoord.y);
    const int x2 = x1 + 1, y2 = y1 + 1;

    // Get the voxels for the four coordinates.
    const float v00 = getVoxel(x1, y1, z); // bottomleft
    const float v01 = getVoxel(x2, y1, z); // bottomright
    const float v10 = getVoxel(x1, y2, z); // topleft
    const float v11 = getVoxel(x2, y2, z); // topright

    // Interpolate points with the same Y-coordinate on the sides of the cell along x (so linearly interpolate between x values with x as factor).
    const float Xfactor = xyCoord.x - static_cast<float>(x1);
    const float v0 = linearInterpolate(v00, v01, Xfactor);
    const float v1 = linearInterpolate(v10, v11, Xfactor);

    // Interpolate resulting samples along y
    return linearInterpolate(v0, v1, xyCoord.y - static_cast<float>(y1));
}


// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This function represents the h(x) function, which returns the weight of the cubic interpolation kernel for a given position x
float Volume::weight(float x)
{
    const float abs_x = std::abs(x);
    const float abs_x2 = abs_x * abs_x;
    const float abs_x3 = abs_x * abs_x2;
    const int a = -1.0f;

    if (0 <= abs_x && abs_x < 1) {
        return (a + 2) * abs_x3 - (a + 3) * abs_x2 + 1;
    } else if (1 <= abs_x && abs_x < 2) {
        return a * abs_x3 - 5 * a * abs_x2 + 8 * a * abs_x - 4 * a;
    } else {
        return 0.0f;
    }
}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This functions returns the results of a cubic interpolation using 4 values and a factor
float Volume::cubicInterpolate(float g0, float g1, float g2, float g3, float factor)
{
    // Check if factor within the range of [0, 1].
    if (0.0f > factor || factor > 1.0f) {
        return 0.0f;
    } else {
        return g0 * weight(factor + 1.0f) + g1 * weight(factor) + g2 * weight(1.0f - factor) + g3 * weight(2.0f - factor);
    }
}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This function returns the value of a bicubic interpolation
float Volume::biCubicInterpolate(const glm::vec2& xyCoord, int z) const
{
    const int p = static_cast<int>(xyCoord.x), q = static_cast<int>(xyCoord.y);
    const float Xfactor = xyCoord.x - static_cast<float>(p), Yfactor = xyCoord.y - static_cast<float>(q);

    /** 
    // Alternative way of computing
    const float v01 = getVoxel(p - 1, q - 1, z);
    const float v02 = getVoxel(p - 1, q, z);
    const float v03 = getVoxel(p - 1, q + 1, z);
    const float v04 = getVoxel(p - 1, q + 2, z);
    const float g0 = cubicInterpolate(v01, v02, v03, v04, Yfactor);

    const float v11 = getVoxel(p, q - 1, z);
    const float v12 = getVoxel(p, q, z);
    const float v13 = getVoxel(p, q + 1, z);
    const float v14 = getVoxel(p, q + 2, z);
    const float g1 = cubicInterpolate(v11, v12, v13, v14, Yfactor);

    const float v21 = getVoxel(p + 1, q - 1, z);
    const float v22 = getVoxel(p + 1, q, z);
    const float v23 = getVoxel(p + 1, q + 1, z);
    const float v24 = getVoxel(p + 1, q + 2, z);
    const float g2 = cubicInterpolate(v21, v22, v23, v24, Yfactor);

    const float v31 = getVoxel(p + 2, q - 1, z);
    const float v32 = getVoxel(p + 2, q, z);
    const float v33 = getVoxel(p + 2, q + 1, z);
    const float v34 = getVoxel(p + 2, q + 2, z);
    const float g3 = cubicInterpolate(v31, v32, v33, v34, Yfactor);

    return cubicInterpolate(g0, g1, g2, g3, Xfactor);
    **/

    // Store intermediate voxels for each row and cubic interpolation of it.
    float vxl[4], g_vals[4];

    // For given coordinates (p,q), loop over the total 16 pixels (4 x 4) covering the pixels from (p-1, q-1) to (p+2, q+2).
    for (int i = -1; i <= 2; i++) {
        for (int j = -1; j <= 2; j++) {
            // Get the voxel for the given pixel coordinates
            vxl[j + 1] = getVoxel(p + static_cast<float>(i), q + static_cast<float>(j), z);
        }
        // Perform cubic interpolation over each row
        g_vals[i + 1] = cubicInterpolate(vxl[0], vxl[1], vxl[2], vxl[3], Yfactor);
    }
    // Perform cubic interpolation over the intermediate computed cubic interpolation row results.
    return cubicInterpolate(g_vals[0], g_vals[1], g_vals[2], g_vals[3], Xfactor);

}

// ======= OPTIONAL : This functions can be used to implement cubic interpolation ========
// This function computes the tricubic interpolation at coord
float Volume::getSampleTriCubicInterpolation(const glm::vec3& coord) const
{
    // Check if coordinates are within distance 2 as we want to later loop over 16 pixels which goes at most distance 2 from a given coordinate.
    if (glm::any(glm::lessThan(coord - 2.0f, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord + 2.0f, glm::vec3(m_dim))))
        return 0.0f;

    // Create new vector coordinate for X and Y. 
    const glm::vec2 xyCoord = glm::vec2(coord.x, coord.y);
    // Calculate floor of z value.
    const int u = static_cast<int>(coord.z);

    // Perform bicubic interpolations for each of the different z-panes.
    const float g0 = biCubicInterpolate(xyCoord, u - 1);
    const float g1 = biCubicInterpolate(xyCoord, u);
    const float g2 = biCubicInterpolate(xyCoord, u + 1);
    const float g3 = biCubicInterpolate(xyCoord, u + 2);

    // Perform final cubic interpolation of the intermediate results.
    return cubicInterpolate(g0, g1, g2, g3, coord.z - static_cast<float>(u));
}

// Load an fld volume data file
// First read and parse the header, then the volume data can be directly converted from bytes to uint16_ts
void Volume::loadFile(const std::filesystem::path& file)
{
    assert(std::filesystem::exists(file));
    std::ifstream ifs(file, std::ios::binary);
    assert(ifs.is_open());

    const auto header = readHeader(ifs);
    m_dim = header.dim;
    m_elementSize = header.elementSize;

    const size_t voxelCount = static_cast<size_t>(header.dim.x * header.dim.y * header.dim.z);
    const size_t byteCount = voxelCount * header.elementSize;
    std::vector<char> buffer(byteCount);
    // Data section is separated from header by two /f characters.
    ifs.seekg(2, std::ios::cur);
    ifs.read(buffer.data(), std::streamsize(byteCount));

    m_data.resize(voxelCount);
    if (header.elementSize == 1) { // Bytes.
        for (size_t i = 0; i < byteCount; i++) {
            m_data[i] = static_cast<uint16_t>(buffer[i] & 0xFF);
        }
    } else if (header.elementSize == 2) { // uint16_ts.
        for (size_t i = 0; i < byteCount; i += 2) {
            m_data[i / 2] = static_cast<uint16_t>((buffer[i] & 0xFF) + (buffer[i + 1] & 0xFF) * 256);
        }
    }
}
}

static Header readHeader(std::ifstream& ifs)
{
    Header out {};

    // Read input until the data section starts.
    std::string line;
    while (ifs.peek() != '\f' && !ifs.eof()) {
        std::getline(ifs, line);
        // Remove comments.
        line = line.substr(0, line.find('#'));
        // Remove any spaces from the string.
        // https://stackoverflow.com/questions/83439/remove-spaces-from-stdstring-in-c
        line.erase(std::remove_if(std::begin(line), std::end(line), ::isspace), std::end(line));
        if (line.empty())
            continue;

        const auto separator = line.find('=');
        const auto key = line.substr(0, separator);
        const auto value = line.substr(separator + 1);

        if (key == "ndim") {
            if (std::stoi(value) != 3) {
                std::cout << "Only 3D files supported\n";
            }
        } else if (key == "dim1") {
            out.dim.x = std::stoi(value);
        } else if (key == "dim2") {
            out.dim.y = std::stoi(value);
        } else if (key == "dim3") {
            out.dim.z = std::stoi(value);
        } else if (key == "nspace") {
        } else if (key == "veclen") {
            if (std::stoi(value) != 1)
                std::cerr << "Only scalar m_data are supported" << std::endl;
        } else if (key == "data") {
            if (value == "byte") {
                out.elementSize = 1;
            } else if (value == "short") {
                out.elementSize = 2;
            } else {
                std::cerr << "Data type " << value << " not recognized" << std::endl;
            }
        } else if (key == "field") {
            if (value != "uniform")
                std::cerr << "Only uniform m_data are supported" << std::endl;
        } else if (key == "#") {
            // Comment.
        } else {
            std::cerr << "Invalid AVS keyword " << key << " in file" << std::endl;
        }
    }
    return out;
}

static float computeMinimum(gsl::span<const uint16_t> data)
{
    return float(*std::min_element(std::begin(data), std::end(data)));
}

static float computeMaximum(gsl::span<const uint16_t> data)
{
    return float(*std::max_element(std::begin(data), std::end(data)));
}

static std::vector<int> computeHistogram(gsl::span<const uint16_t> data)
{
    std::vector<int> histogram(size_t(*std::max_element(std::begin(data), std::end(data)) + 1), 0);
    for (const auto v : data)
        histogram[v]++;
    return histogram;
}
