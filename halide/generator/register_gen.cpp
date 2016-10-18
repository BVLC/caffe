#include <Halide.h>

using namespace Halide;

// Generators compile halide objects ahead-of-time, to use them we
// define a class that inherits from Halide::Generator.
class GenBlur : public Halide::Generator<GenBlur> {
  public:

  ImageParam inp{type_of<float>(), 4,"inp"};
  Var b, c, x, y;
  Var xi, yi;

  Func build() {
    Func blur_x, blur_y;

    Func clamped("clamped");
    clamped = BoundaryConditions::repeat_edge(inp);

    Func inp_c("inp_c");
    inp_c(x,y, c, b) = clamped(x,y,c, b);


    // The algorithm - no storage or order
    blur_x(x, y, c, b) = (inp_c(x-1, y, c, b) + inp_c(x, y, c, b) + inp_c(x+1, y, c, b))/3;
    blur_y(x, y, c, b) = (blur_x(x, y-1, c, b) + blur_x(x, y, c, b) + blur_x(x, y+1,c ,b))/3;

    // The schedule - defines order, locality; implies storage
    //blur_y.tile(x, y, xi, yi, 256, 32)
    //      .vectorize(xi, 8).parallel(y);
    //blur_x.compute_at(blur_y, x).vectorize(x, 8);

    return blur_y;
  }

};

RegisterGenerator<GenBlur> gen_blurfunc{"blur_y"};
