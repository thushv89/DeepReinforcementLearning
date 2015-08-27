#include <iostream>
#include <random>
#include <vector>
#include <string>

int main(int argc, char **argv) {
    unsigned multiply = 16;
    if (argc == 2) {
        multiply = std::stoi(argv[1]);
    }

    std::cin.sync_with_stdio(false);

    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // each input should have 16 variations
    while (std::cin.good()) {
        std::vector<float> image;
        float val;
        for (unsigned i = 0; i < 28 * 28 + 1; ++i) {
            if (std::cin >> val) {
                image.push_back(val);
            } else {
                return 0;
            }
        }

        for (unsigned i = 0; i < multiply; ++i) {
            for (unsigned j = 0; j < 28 * 28; ++j) {
                std::cout << std::min(image[j] + dist(gen), 1.0f) << " ";
            }

            // print the label
            std::cout << image.back() << "\n";
        }
    }
    return 0;
}
