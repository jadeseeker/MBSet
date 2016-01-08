// Mandelbrot Set program
// ECE8893, Georgia Tech, Fall 2014
// AKSHAY SAWANT

#include <iostream>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <GL/freeglut.h>

#ifdef LINUX
// Linux Headers
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#ifdef OSX
// MAC Headers
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#endif

#ifdef WINDOWS
// Windows Headers
#include <Windows.h>
#include <gl/GL.h>
#include <gl/glut.h>
#endif

#include <fstream>
#include <vector>

using namespace std;

#define WINDOW_DIM 512                                                                         // Size of window in pixels, both width and height
#define NUM_THREADS 32                                                                         // Define the number of threads

/*************************************************************************************************************************
 *                                     CLASSES, STRUCTURES AND GLOBAL VARIABLES                                          *
 *************************************************************************************************************************/

struct Position                                                                                // Structure for using mouse click
{
  Position() : x(0), y(0) {}
  float x, y;                                                                                  // X and Y coordinates of the mouse click
};
Position start, end;                                                                           // Start and end position of mouse click

class Complex 
{
public:
  float   r;
  float   i;
  __host__ __device__ Complex() : r(0), i(0) {}
  __host__ __device__ Complex( float a, float b ) : r(a), i(b)  {}
  __host__ __device__ Complex(const Complex& x) : r(x.r), i(x.i) {}
  __host__ __device__ float magnitude2( void ) 
  {
    return r * r + i * i;
  }
  __host__ __device__ Complex operator*(const Complex& a) 
  {
    return Complex(r*a.r - i*a.i, i*a.r + r*a.i);
  }
  __host__ __device__ Complex operator+(const Complex& a) 
  {
    return Complex(r+a.r, i+a.i);
  }
  void Print();
};

void Complex::Print()                                                                           // Function to print the Complex number (used for debugging)
{
  if (i == 0)
  { 
    cout << r;
  }
  else
  {
    cout << '(' << r << "," << i << ')';
  }
}

std::ostream& operator << (std::ostream &os, const Complex& c)                                  // Function to print the Complex number (used for debugging)
{
  if (c.i == 0)
  { 
    os << c.r;
  }
  else
  {     
    os << '(' << c.r << "," << c.i << ')';
  }
  return os;
}

//Complex minC(-2.0, -1.2);
//Complex maxC(1.0, 1.8);
Complex minC(-2.25, -1.25);                                                                      // Changed the screen coordinates to display MB Set in the center of the window
Complex maxC(0.75, 1.25);

Complex* dev_minC;                                                                               // Complex pointer for minC (used in device)
Complex* dev_maxC;                                                                               // Complex pointer for maxC (used in device)
Complex* dev_c;                                                                                  // Complex pointer for c array (used in device)
int* dev_computation;                                                                            // Integer pointer for array of number of iterations (used in device)

const int maxIt = 2000;                                                                          // Maximum Iterations

Complex* c = new Complex[WINDOW_DIM * WINDOW_DIM];                                               // Complex pointer for c array
int computation[WINDOW_DIM * WINDOW_DIM];                                                        // Array to store number of iterations for a particular pixel

float dx, dy, dz;                                                                                // Global variables used in mouse selection
bool dispSelector;                                                                               // Will display square when mouse is clicked and clear the square when mouse is released
bool cudaSelect = 1;                                                                             // To select between CUDA and non-CUDA mode

class Memory                                                                                     // Class memory to store minC, maxC values after zooming in. Used in back button
{
public:
  float minC_r, minC_i, maxC_r, maxC_i;
  Memory(float a, float b, float c, float d)
    : minC_r(a), minC_i(b), maxC_r(c), maxC_i(d) {}
};

vector<Memory> memory_vec;                                                                       // Vector of class Memory, used to store minC, maxC

class RGB                                                                                        // RGB class to define R, G, B coordinates
{
public:
  RGB()
    : r(0), g(0), b(0) {}
  RGB(double r0, double g0, double b0)
    : r(r0), g(g0), b(b0) {}
public:
  double r;
  double g;
  double b;
};

RGB* colors = 0;                                                                                 // Array of color values

/*****************************************************************************************************************
 *                                            FUNCTION DECLARATION                                               *
 *****************************************************************************************************************/

void init(void);                                                                                 // Function to initialize OpenGL
void drawMandelbrot();                                                                           // Function to compute MB Set
void compute_c_array();                                                                          // Function to generate c array (used in non-CUDA mode)
Complex generate_c(int i, int j);                                                                // Function to generate c per pixel (used in non-CUDA mode)
void display(void);                                                                              // Display function in OpenGL 
void displayMandelbrot();                                                                        // Function to display MB Set
void drawSquare();                                                                               // Function to display square
void keyboard (unsigned char key, int x, int y);                                                 // Function for keystroke processing 
void mouse(int button, int state, int x, int y);                                                 // Function for mouse selection
void motion(int x, int y);                                                                       // Function for mouse motion

void InitializeColors()                                                                          // Function to initialize colors, Color scheme is modelled on the sun
{
  colors = new RGB[maxIt + 1];
  for (int i = 0; i < maxIt; ++i)
  {
    if(i < 1)
      colors[i] = RGB(1, 0, 0);
    if(i >=1 && i < 2)
      colors[i] = RGB(1, 0.1, 0);
    if(i >= 2 && i < 3)
      colors[i] = RGB(1, 0.15, 0);
    if(i >= 3 && i < 4) 
      colors[i] = RGB(1, 0.2, 0);
    if(i >= 4 && i < 5) 
      colors[i] = RGB(1, 0.25, 0);
    if(i >= 5 && i < 6) 
      colors[i] = RGB(1, 0.3, 0);
    if(i >= 6 && i < 7) 
      colors[i] = RGB(1, 0.35, 0);
    if(i >= 7 && i < 8) 
      colors[i] = RGB(1, 0.4, 0);
    if(i >= 8 && i < 9) 
      colors[i] = RGB(1, 0.45, 0);
    if(i >= 9 && i < 10) 
      colors[i] = RGB(1, 0.5, 0);
    if(i >= 10 && i < 20) 
      colors[i] = RGB(1, 0.55, 0);
    if(i >= 20 && i < 40) 
      colors[i] = RGB(1, 0.6, 0);
    if(i >= 40 && i < 80) 
      colors[i] = RGB(1, 0.65, 0);
    if(i >= 80 && i < 160) 
      colors[i] = RGB(1, 0.7, 0);
    if(i >= 160 && i < 240) 
      colors[i] = RGB(1, 0.75, 0);
    if(i >= 240 && i < 480) 
      colors[i] = RGB(1, 0.8, 0);
    if(i >= 480 && i < 720) 
      colors[i] = RGB(1, 0.85, 0);
    if(i >= 720 && i < 960) 
      colors[i] = RGB(1, 0.9, 0);
    if(i >= 960 && i < 1200) 
      colors[i] = RGB(1, 0.95, 0);
    if(i >= 1200 && i < 1400)
      colors[i] = RGB(1, 1, 0);
    if(i >= 1400 && i < 1600)
      colors[i] = RGB(1, 1, 0.25);
    if(i >= 1600 && i < 1800)
      colors[i] = RGB(1, 1, 0.5);
    if(i >= 1800 && i < maxIt)
      colors[i] = RGB(1, 1, 1);
  }
  colors[maxIt] = RGB();                                                                                    // Black color for points in the MB Set
}  

/**********************************************************************************************************************
 *                                                   CUDA FUNCTION                                                    *
 **********************************************************************************************************************/

__global__ void computeMB(Complex* dev_minC, Complex* dev_maxC, int* dev_computation, Complex* dev_c)       // CUDA function to compute MB Set
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;                                                           // ID is calculated first
  int i = id / WINDOW_DIM;                                                                                  // Used in linear interpolation
  int j = id % WINDOW_DIM;                                                                                  // And accessing c array 
  
  double dX = dev_maxC->r - dev_minC->r;
  double dY = dev_maxC->i - dev_minC->i;
  double nX = (double) i / WINDOW_DIM;
  double nY = (double) j / WINDOW_DIM;
  
  dev_c[id].r = dev_minC->r + nX * dX;
  dev_c[id].i = dev_minC->i + nY * dY;

  Complex Z (0,0);
  Z.r = dev_c[id].r;
  Z.i = dev_c[id].i;
  dev_computation[id] = 0;
      
  while(dev_computation[id] < 2000 && Z.magnitude2() < 4.0)                                                 // Iterations is less than 2000 or z exceeds 2
  {
    dev_computation[id]++;
    Z = (Z*Z) + dev_c[id];
  }
}

/*************************************************************************************************************************
 *                                                 MAIN PROGRAM BEGINS                                                   *
 *************************************************************************************************************************/

int main(int argc, char* argv[])                                                                      // Main function
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);
    glutInitWindowPosition(700, 250);
    glutCreateWindow("Mandelbrot Set");
  
    init();                                                                                           // Initialization function
    
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutKeyboardFunc (keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    glutMainLoop();
    return 0;
}
  
void init(void)                                                                                             // Function to initialize OpenGL
{   
  InitializeColors();
  drawMandelbrot();

  glViewport(0, 0, WINDOW_DIM, WINDOW_DIM);                                            
  glMatrixMode(GL_PROJECTION); 
  glLoadIdentity();

  gluOrtho2D(0, WINDOW_DIM, WINDOW_DIM, 0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();  
}

/*************************************************************************************************************************
 *                                            FUNCTIONS FOR COMPUTING MB SET                                             *
 *************************************************************************************************************************/                               

void drawMandelbrot()                                                                     // Function to compute MB Set
{

  if(cudaSelect)
  {
    cudaMalloc((void**)&dev_computation, WINDOW_DIM * WINDOW_DIM * sizeof(int));
    cudaMalloc((void**)&dev_minC, sizeof(Complex));
    cudaMalloc((void**)&dev_maxC, sizeof(Complex));
    cudaMalloc((void**)&dev_c, WINDOW_DIM * WINDOW_DIM * sizeof(Complex));
    cudaMemcpy(dev_minC, &minC, sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_maxC, &maxC, sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_computation, computation, WINDOW_DIM * WINDOW_DIM * sizeof(int), cudaMemcpyHostToDevice);  
    cudaMemcpy(dev_c, c, WINDOW_DIM * WINDOW_DIM * sizeof(Complex), cudaMemcpyHostToDevice);  
    
    computeMB<<< WINDOW_DIM * WINDOW_DIM / NUM_THREADS, NUM_THREADS >>>(dev_minC, dev_maxC, dev_computation, dev_c);
  
    cudaMemcpy(computation, dev_computation, WINDOW_DIM * WINDOW_DIM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dev_c, WINDOW_DIM * WINDOW_DIM * sizeof(Complex), cudaMemcpyDeviceToHost);  
  }     

  else
  {
    compute_c_array();
    for(int i = 0; i < WINDOW_DIM; i++)
    {
      for(int j = 0; j < WINDOW_DIM; j++)
      {
        Complex Z = c[i*WINDOW_DIM + j];
        computation[i*WINDOW_DIM + j] = 0;
     
        while(computation[i*WINDOW_DIM + j] < maxIt && Z.magnitude2() < 4.0)                  // Iterations is less than 2000 or z exceeds 2
        {
          computation[i*WINDOW_DIM + j]++;
          Z = (Z*Z) + c[i*WINDOW_DIM + j];
        }   
      }
    }
  }
}

void compute_c_array()                                                                    // Function to generate c array (used in non-CUDA mode)
{
  for(int i = 0; i < WINDOW_DIM; i++)
  {
    for(int j = 0; j < WINDOW_DIM; j++)
    {
      c[i*WINDOW_DIM + j] = generate_c(i, j);  
    }
  }
}

Complex generate_c(int i, int j)                                                          // Function to generate c per pixel (used in non-CUDA mode)
{
  double dX = maxC.r - minC.r;
  double dY = maxC.i - minC.i;
  double nX = (double)i / WINDOW_DIM;
  double nY = (double)j / WINDOW_DIM;
  return minC + Complex(nX*dX, nY*dY); 
}

/**************************************************************************************************************************
 *                                           FUNCTIONS FOR DISPLAYING MB SET                                              *
 **************************************************************************************************************************/

void display(void)                                                                        // Display function in OpenGL 
{
  bool background = 1;                                                                    // Background is set to white by default
  if(background)
    glClearColor(1.0, 1.0, 1.0, 1.0);                                                     // White Background      
  else  
    glClearColor(0.0, 0.0, 0.0, 1.0);                                                     // Black Background
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();

  displayMandelbrot();                                                                    // Display MB Set

  if(dispSelector)
    drawSquare();                                                                         // Draw Square
  glutSwapBuffers();                                                                      // Swap the double buffers
}

void displayMandelbrot()                                                                  // Function to display MB Set
{
  glBegin(GL_POINTS);
  for(int i = 0; i < WINDOW_DIM; i++)
  {
    for(int j = 0; j < WINDOW_DIM; j++)
    {
      glColor3f(colors[computation[i*WINDOW_DIM + j]].r, colors[computation[i*WINDOW_DIM + j]].g, colors[computation[i*WINDOW_DIM + j]].b);
      glVertex2d(i, j);
    }
  }
  glEnd();
}

void drawSquare()                                                                        // Function to draw a square
{
  glColor3f(1, 1, 1);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glBegin(GL_POLYGON);
  glVertex2f(start.x, start.y);
  glVertex2f(end.x, start.y);
  glVertex2f(end.x, end.y);
  glVertex2f(start.x, end.y);
  glEnd(); 
}

/*************************************************************************************************************************
 *                                     FUNCTIONS FOR KEYBOARD, MOUSE AND MOTION                                          *
 *************************************************************************************************************************/

void keyboard (unsigned char key, int x, int y)                                          // Function for keystroke processing 
{
  if(key == 'q' || key == 'Q')                                                           // If press q, quit the program
  { 
    exit(0);  
  }

  if(key == 'b' || key == 'B')                                                           // If press b, zoom out 
  {
    if(memory_vec.size() > 0)
    {
      Memory temp = memory_vec.back();                                                   // Pop the contents from the back of the memory vector
      memory_vec.pop_back();
      cout<<"Memory vector size = "<<memory_vec.size()<<endl;                            // Print the vector size
      minC.r = temp.minC_r;
      minC.i = temp.minC_i;
      maxC.r = temp.maxC_r;
      maxC.i = temp.maxC_i;
      drawMandelbrot();                                                                  // Recompute the MB Set
      glutPostRedisplay();                                                               // Redisplay rhe MB Set
    }
    else
      cout<<"Cannot zoom out"<<endl;                                                     // Cannot zoom out if vector is empty
  }
}

void mouse(int button, int state, int x, int y)                                          // Function for mouse selection
{
  if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)                                   // If mouse button is pressed
  {
    start.x = end.x = x;
    start.y = end.y = y;
    dispSelector = 1;
  }

  if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)                                     // If mouse button is released                         
  {
    memory_vec.push_back(Memory(minC.r, minC.i, maxC.r, maxC.i));                        // Push minC, maxC contents into the memory
    cout<<"Memory vector size = "<<memory_vec.size()<<endl;                              // Print the vector size 

    if(x > start.x && y > start.y)
    {
      end.x = start.x + dz;
      end.y = start.y + dz;

      for(int i = 0; i < WINDOW_DIM; i++)
      {
        for(int j = 0; j < WINDOW_DIM; j++)
        {
          if(i == start.x && j  == start.y)
          {
            minC.r = c[i*WINDOW_DIM + j].r;
            minC.i = c[i*WINDOW_DIM + j].i;
          }

          if(i == end.x && j  == end.y)
          {
            maxC.r = c[i*WINDOW_DIM + j].r;
            maxC.i = c[i*WINDOW_DIM + j].i;
          }
        }
      }
    }

    if(x < start.x && y < start.y)
    {
      end.x = start.x - dz;
      end.y = start.y - dz;

      for(int i = 0; i < WINDOW_DIM; i++)
      {
        for(int j = 0; j < WINDOW_DIM; j++)
        {
          if(i == end.x && j  == end.y)
          {
            minC.r = c[i*WINDOW_DIM + j].r;
            minC.i = c[i*WINDOW_DIM + j].i;
          }

          if(i == start.x && j  == start.y)
          {
            maxC.r = c[i*WINDOW_DIM + j].r;
            maxC.i = c[i*WINDOW_DIM + j].i;
          }
        }
      }
    }

    if(x > start.x && y < start.y)
    {
      end.x = start.x + dz;
      end.y = start.y - dz;

      for(int i = 0; i < WINDOW_DIM; i++)
      {
        for(int j = 0; j < WINDOW_DIM; j++)
        {
          if(i == start.x && j  == end.y)
          {
            minC.r = c[i*WINDOW_DIM + j].r;
            minC.i = c[i*WINDOW_DIM + j].i;
          }

          if(i == end.x && j  == start.y)
          {
            maxC.r = c[i*WINDOW_DIM + j].r;
            maxC.i = c[i*WINDOW_DIM + j].i;
          }
        }
      }
    }

    if(x < start.x && y > start.y)
    {
      end.x = start.x - dz;
      end.y = start.y + dz;

      for(int i = 0; i < WINDOW_DIM; i++)
      {
        for(int j = 0; j < WINDOW_DIM; j++)
        {
          if(i == end.x && j  == start.y)
          {
            minC.r = c[i*WINDOW_DIM + j].r;
            minC.i = c[i*WINDOW_DIM + j].i;
          }

          if(i == start.x && j  == end.y)
          {
            maxC.r = c[i*WINDOW_DIM + j].r;
            maxC.i = c[i*WINDOW_DIM + j].i;
          }
        }
      }
    }

    drawMandelbrot();                                                                               // Recompute the MB Set
    dispSelector = 0;                                                                               // Hide the square after mouse button release
    glutPostRedisplay();                                                                            // Redisplay the MB Set
  }
}

void motion(int x, int y)                                                                           // Function for mouse motion
{
  dx = abs(x - start.x);
  dy = abs(y - start.y);

  if(x > start.x && y > start.y)
  {
    if(dx > dy)
      dz = dy;
    if(dx < dy)
      dz = dx;
    end.x = start.x + dz;  
    end.y = start.y + dz;
  }

  if(x < start.x && y < start.y)
  {
    if(dx > dy)
      dz = dy;
    if(dx < dy)
      dz = dx;
    end.x = start.x - dz;  
    end.y = start.y - dz;
  }

  if(x > start.x && y < start.y)
  {
    if(dx > dy)
      dz = dy;
    if(dx < dy)
      dz = dx;
    end.x = start.x + dz;  
    end.y = start.y - dz;
  }

  if(x < start.x && y > start.y)
  {
    if(dx > dy)
      dz = dy;
    if(dx < dy)
      dz = dx;
    end.x = start.x - dz;  
    end.y = start.y + dz;
  }

  glutPostRedisplay();
}

/*************************************************************************************************************************
 *                                                 MAIN PROGRAM ENDS                                                     *
 *************************************************************************************************************************/