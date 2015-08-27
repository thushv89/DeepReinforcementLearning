#include <iostream>
#include <vector>

int main() {
    std::cin.sync_with_stdio(false);

    // type pun the raw character array out
    union { float data; unsigned char raw[4]; } val;
    static_assert(sizeof(val) == 4, "Some weird architecture you have there");

    while (std::cin >> val.data) {
        std::cout << val.raw[0] << val.raw[1] << val.raw[2] << val.raw[3];
    }
}
