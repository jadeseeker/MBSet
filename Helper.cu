class Complex 
{
public:
  float   r;
  float   i;
  __host__ __device__ Complex() : r(0), i(0) {}
  __host__ __device__ Complex( float a, float b ) : r(a), i(b)  {}
  __host__ __device__ Complex(const Complex& x) : r(x.r), i(x.i) {}

  __host__ __device__ float magnitude2( void ) {
    return r * r + i * i;
  }

  __host__ __device__ Complex operator*(const Complex& a) {
    return Complex(r*a.r - i*a.i, i*a.r + r*a.i);
  }

  __host__ __device__ Complex operator+(const Complex& a) {
    return Complex(r+a.r, i+a.i);
  }

  void Print();
};

class RGB       // RGB class to define R, G, B coordinates
{
public:
  RGB() : r(0), g(0), b(0) {}

  RGB(double r0, double g0, double b0) : r(r0), g(g0), b(b0) {}

public:
  double r;
  double g;
  double b;
};


class Memory     // Class memory to store minC, maxC values after zooming in. Used in back button
{
public:
  float minC_r, minC_i, maxC_r, maxC_i;

  Memory(float a, float b, float c, float d) : minC_r(a), minC_i(b), maxC_r(c), maxC_i(d) {}
};


struct Position  // Structure for using mouse click
{
  Position() : x(0), y(0) {}
  float x, y;                                                                                  // X and Y coordinates of the mouse click
};