#include <iostream>
#include <vector>

int main() {
    std::vector<float> buffer(28 * 28);

    std::cin.sync_with_stdio(false);

    while (std::cin.good()) {
        std::cin.read(reinterpret_cast<char *>(&buffer[0]), buffer.size() * sizeof(float));
    }
}
