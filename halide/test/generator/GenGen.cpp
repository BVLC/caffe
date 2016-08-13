#include "Halide.h"

int main(int argc, char **argv) {
  return Halide::Internal::generate_filter_main(argc, argv, std::cerr);
}

