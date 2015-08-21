#include <stdexcept>
#include <iostream>
#include <vector>

#include "lodepng.h"

float get_float(const bool text_format) {
    if (text_format) {
        float val;
        if (std::cin >> val) {
            return val;
        } else {
            throw std::runtime_error("end of stream");
        }
    } else {
        union { float data; unsigned char bytes[4]; } val;
        static_assert(sizeof(float) == 4, "Fix your architecture");

        // read the float
        for (unsigned j = 0; j < sizeof(float); ++j) {
            char c;
            if (std::cin.get(c)) {
                val.bytes[j] = c;
            } else {
                throw std::runtime_error("end of stream");
            }
        }

        return val.data;
    }
}

int main(int argc, char **) {
    bool text_format = false;
    if (argc == 2) {
        text_format = true;
    }

    std::vector<unsigned char> png(28 * 28 * 4);

    int counter = 0;
    try {
        while (true) {
            for (unsigned i = 0; i < 28 * 28; ++i) {
                const float data = get_float(text_format);
                // write the png
                png[i * 4 + 0] = static_cast<unsigned char>(data * 255.0f);
                png[i * 4 + 1] = static_cast<unsigned char>(data * 255.0f);
                png[i * 4 + 2] = static_cast<unsigned char>(data * 255.0f);
                png[i * 4 + 3] = 255;
            }

            // kill the label
            get_float(text_format);

            std::vector<unsigned char> data;
            lodepng::encode(data, png, 28, 28);

            const std::string filename = std::to_string(counter++) + ".png";
            std::cout << "Saved " << filename << std::endl;
            lodepng::save_file(data, filename);
        }
    } catch (const std::runtime_error &e) {
    }
}
