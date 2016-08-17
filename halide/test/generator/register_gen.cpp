#include <Halide.h>

using namespace Halide;

// Generators compile halide objects ahead-of-time, to use them we
// define a class that inherits from Halide::Generator.
class GenPlip : public Halide::Generator<GenPlip> {
  public:
  ImageParam inp{type_of<float>(), 4,"inp"};
  Var x, y, dx, dy;

  Func build() {
    Expr h = (inp.extent(2) / 2);

    Func plip;
    plip(x, y, dx, dy) = inp(x, y, dx, dy) + x + y +float(0.5);

    return plip;
  }
};

RegisterGenerator<GenPlip> gen_plip{"plip"};
