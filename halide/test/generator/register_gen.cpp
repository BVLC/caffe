#include <Halide.h>

using namespace Halide;

//Target target = get_host_target();

// upgrade this?
//target.set_feature(Target::CUDA);
//target.set_feature(Target::CUDACapability35);


// Generators are a more structured way to do ahead-of-time
// compilation of Halide pipelines. Instead of writing an int main()
// with an ad-hoc command-line interface like we did in lesson 10, we
// define a class that inherits from Halide::Generator.
class GenPlip : public Halide::Generator<GenPlip> {
  public:
  ImageParam inp{type_of<float>(), 4,"inp"};
  Var x, y, dx, dy;

  Func build() {
    Expr h = (inp.extent(2) / 2);

    Func plip;
    plip(x, y, dx, dy) = inp(x, y, dx, dy) + x + y +float(0.5);

    //Var s;
    //plip.fuse(x, y, s);
    //plip.gpu_tile(s, dx, dy, 8, 8, 8);
    //plip.compile_to_file("plip_Forward_gpu", {inp}, target);
    //plip.compile_to_file("plip_Backward_gpu", {inp}, target);
    return plip;
  }
};

RegisterGenerator<GenPlip> gen_plip{"plip"};

/*
template< Forward_cpu, Backward_cpu, Forward_gpu, Backward_gpu,
          cpu_target, gpu_target>
class Wrapper {
*/




