#ifndef BMP_H
#define BMP_H

#include <cstdint>

static const uint16_t BMP_SIGNATURE = 0x4d42;

/* 1 = monochrome palette. NumColors = 1
 * 4 = 4bit palletized. NumColors = 16
 * 8 = 8bit palletized. NumColors = 256
 * 16 = 16bit RGB. NumColors = 65536
 * 24 = 24bit RGB. NumColors = 16M */
enum bits_per_pixel {
  BPP_1 = 1,
  BPP_4 = 4,
  BPP_8 = 8,
  BPP_16 = 16,
  BPP_24 = 24,
  BPP_32 = 32
};

/* 0 = BI_RGB   no compression
 * 1 = BI_RLE8 8bit RLE encoding
 * 2 = BI_RLE4 4bit RLE encoding */
enum compression_type {
  CT_RGB = 0,
  CT_RLE8 = 1,
  CT_RLE4 = 2,
};

#pragma pack(push)
#pragma pack(1)

typedef struct bmp_file_hdr {
  uint16_t signature;
  uint32_t file_size; // in bytes
  uint32_t reserved;
  uint32_t data_offset;
} bmp_file_hdr_t;
//////////////////////////////////////////////////////////////

typedef union tagRGBQUAD {
  struct {
    uint8_t B;
    uint8_t G;
    uint8_t R;
    uint8_t _R;
  } rgb;
  uint32_t i32_val;
} RGBQUAD;
//////////////////////////////////////////////////////////////

typedef struct bmp_info_hdr {
  uint32_t size_hdr;         // of info header = 40
  uint32_t width;            // in pixels;
  uint32_t height;           // in pixels
  uint16_t plains;           // (=1) ??
  uint16_t bpp;              // bits per pixel
  uint32_t compression_type; // CT_RGB
  uint32_t size_img;         // compressed
  uint32_t x_pix_per_meter;  //
  uint32_t y_pix_per_meter;  //
  uint32_t colors_used;      // 0 - all
  uint32_t important_colors; // 0 = all
} bmp_info_hdr_t;
#pragma pack(pop)

#endif // BMP_H
