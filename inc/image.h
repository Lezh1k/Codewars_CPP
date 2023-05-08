#ifndef IMAGE_H
#define IMAGE_H

#include <cstddef>
#include <cstdint>

class Image {
private:
  size_t m_width;
  size_t m_height;

public:
  Image() = delete;
  Image(size_t width,
        size_t height);
};

#endif // IMAGE_H
