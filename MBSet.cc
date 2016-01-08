/* 
 * File:   MBSet.cu
 * 
 * Created on June 24, 2012
 * 
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 * Author:  Yash Shah
 */

#include <iostream>
#include <unistd.h>
#include <cuda_runtime_api.h>

#include <GL/freeglut.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <fstream>
#include <vector>

using namespace std;

//#include "Helper.cu"


#define WINDOW_DIM 512 
#define numThreads 32 


//******************************************************************************************************8
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
};

class RGB 
{
public:
  RGB() : r(0), g(0), b(0) {}

  RGB(double r0, double g0, double b0) : r(r0), g(g0), b(b0) {}

public:
  double r;
  double g;
  double b;
};


class Memory 
{
public:
  float minC_r, minC_i, maxC_r, maxC_i;

  Memory(float a, float b, float c, float d) : minC_r(a), minC_i(b), maxC_r(c), maxC_i(d) {}
};


struct Position 
{
  Position() : x(0), y(0) {}
  float x, y;                        
};
//*********************************************************************************************************8


Position start, end;   

Complex minC(-2.25, -1.25);     
Complex maxC(0.75, 1.25);

Complex* dev_minC;
Complex* dev_maxC;

Complex* dev_c;
int* iterCount;

const int maxIt = 2000;

Complex* c = new Complex[WINDOW_DIM * WINDOW_DIM];
int computation[WINDOW_DIM * WINDOW_DIM]; 

float dx, dy, dz;
bool dispSelector; 

vector<Memory> memBuffer;

RGB* colors = 0;

void init(void);
void cudaMB();

void display(void);
void displayMandelbrot(); 
void highlight();

void keyboard (unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);  
void motion(int x, int y); 



void InitializeColors()
{

  colors = new RGB[maxIt + 1];
  for (int i = 0; i < maxIt; ++i)
  {
    if(i < 8)
      colors[i] = RGB(0.8, 0.8, 0.8);
    else
      colors[i] = RGB(drand48(), drand48(), drand48());
  }
  colors[maxIt] = RGB();
}  


__global__ void computeMB(Complex* dev_minC, Complex* dev_maxC, int* iterCount, Complex* dev_c) 
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int i = id / WINDOW_DIM;
  int j = id % WINDOW_DIM; 
  
  double dX = dev_maxC->r - dev_minC->r;
  double dY = dev_maxC->i - dev_minC->i;
  double nX = (double) i / WINDOW_DIM;
  double nY = (double) j / WINDOW_DIM;
  
  dev_c[id].r = dev_minC->r + nX * dX;
  dev_c[id].i = dev_minC->i + nY * dY;

  Complex Z (0,0);
  Z.r = dev_c[id].r;
  Z.i = dev_c[id].i;
  iterCount[id] = 0;
      
  while(iterCount[id] < 2000 && Z.magnitude2() < 4.0)
  {
    iterCount[id]++;
    Z = (Z*Z) + dev_c[id];
  }
}


int main(int argc, char* argv[])
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);
  glutInitWindowPosition(700, 250);
  glutCreateWindow("Mandelbrot Set");

  InitializeColors();
  init(); 
  
  glutDisplayFunc(display);
  glutIdleFunc(display);
  glutKeyboardFunc (keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  glutMainLoop();
  return 0;
}
  
void init(void) 
{   
  
  cudaMB();

  glViewport(0, 0, WINDOW_DIM, WINDOW_DIM);                                            
  glMatrixMode(GL_PROJECTION); 
  glLoadIdentity();

  gluOrtho2D(0, WINDOW_DIM, WINDOW_DIM, 0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();  
}


void cudaMB() 
{

  cudaMalloc( (void**)&iterCount, WINDOW_DIM * WINDOW_DIM * sizeof(int));
  cudaMalloc( (void**)&dev_minC, sizeof(Complex));
  cudaMalloc( (void**)&dev_maxC, sizeof(Complex));
  cudaMalloc( (void**)&dev_c, WINDOW_DIM * WINDOW_DIM * sizeof(Complex));
  cudaMemcpy( dev_minC, &minC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy( dev_maxC, &maxC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy( iterCount, computation, WINDOW_DIM * WINDOW_DIM * sizeof(int), cudaMemcpyHostToDevice);  
  cudaMemcpy( dev_c, c, WINDOW_DIM * WINDOW_DIM * sizeof(Complex), cudaMemcpyHostToDevice);  
  
  computeMB<<< WINDOW_DIM * WINDOW_DIM / numThreads, numThreads >>>(dev_minC, dev_maxC, iterCount, dev_c);

  cudaMemcpy( computation, iterCount, WINDOW_DIM * WINDOW_DIM * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy( c, dev_c, WINDOW_DIM * WINDOW_DIM * sizeof(Complex), cudaMemcpyDeviceToHost);  

}



void display(void) 
{
  glClearColor(1.0, 1.0, 1.0, 1.0); 

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();

  displayMandelbrot();

  if(dispSelector)  highlight();

  glutSwapBuffers();
}

void displayMandelbrot()
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

void highlight()
{
  glColor3f(0, 0, 0);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glBegin(GL_POLYGON);
  glVertex2f(start.x, start.y);
  glVertex2f(end.x, start.y);
  glVertex2f(end.x, end.y);
  glVertex2f(start.x, end.y);
  glEnd(); 
}


void keyboard (unsigned char key, int x, int y) 
{
  if(key == 'q' || key == 'Q')
  { 
    exit(0);  
  }

  if(key == 'b' || key == 'B')
  {
    if(memBuffer.size() > 0)
    {
      Memory temp = memBuffer.back();
      memBuffer.pop_back();
      cout<< "Zoom vector = " << memBuffer.size()<<endl;
      minC.r = temp.minC_r;
      minC.i = temp.minC_i;
      maxC.r = temp.maxC_r;
      maxC.i = temp.maxC_i;
      cudaMB();   
      glutPostRedisplay(); 
    }
    else
      cout<< "Cannot zoom out" << endl;  
  }
}

void mouse(int button, int state, int x, int y) 
{
  if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) 
  {
    start.x = end.x = x;
    start.y = end.y = y;
    dispSelector = 1;
  }

  if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)                        
  {
    memBuffer.push_back(Memory(minC.r, minC.i, maxC.r, maxC.i)); 
    cout<<"Zoom Vector = "<<memBuffer.size()<<endl;


    if(x > start.x)
    {
      end.x = start.x + dz;

      if(y > start.y)
      {
        end.y = start.y + dz;
        for(int i = 0; i < WINDOW_DIM; i++){
          for(int j = 0; j < WINDOW_DIM; j++){
            if(i == start.x && j == start.y)
            {
              minC.r = c[i*WINDOW_DIM + j].r;
              minC.i = c[i*WINDOW_DIM + j].i;
            }
            else if(i == end.x && j == end.y)
            {
              maxC.r = c[i*WINDOW_DIM + j].r;
              maxC.i = c[i*WINDOW_DIM + j].i;
            }
          }
        }
      }
      else 
      {
        end.y = start.y - dz;
        for(int i = 0; i < WINDOW_DIM; i++){
          for(int j = 0; j < WINDOW_DIM; j++){
            if(i == start.x && j == end.y)
            {
              minC.r = c[i*WINDOW_DIM + j].r;
              minC.i = c[i*WINDOW_DIM + j].i;
            }
            else if(i == end.x && j == start.y)
            {
              maxC.r = c[i*WINDOW_DIM + j].r;
              maxC.i = c[i*WINDOW_DIM + j].i;
            }
          }
        }
      }
    }
    else
    {
      end.x = start.x - dz;

      if(y < start.y)
      {
        end.y = start.y - dz;
        for(int i = 0; i < WINDOW_DIM; i++){
          for(int j = 0; j < WINDOW_DIM; j++){
            if(i == end.x && j == end.y)
            {
              minC.r = c[i*WINDOW_DIM + j].r;
              minC.i = c[i*WINDOW_DIM + j].i;
            }
            else if(i == start.x && j == start.y)
            {
              maxC.r = c[i*WINDOW_DIM + j].r;
              maxC.i = c[i*WINDOW_DIM + j].i;
            }
          }
        }
      }
      else 
      {
        end.y = start.y + dz;
        for(int i = 0; i < WINDOW_DIM; i++){
          for(int j = 0; j < WINDOW_DIM; j++){
            if(i == end.x && j == start.y)
            {
              minC.r = c[i*WINDOW_DIM + j].r;
              minC.i = c[i*WINDOW_DIM + j].i;
            }
            else if(i == start.x && j == end.y)
            {
              maxC.r = c[i*WINDOW_DIM + j].r;
              maxC.i = c[i*WINDOW_DIM + j].i;
            }
          }
        }
      }
    }


    cudaMB();         
    dispSelector = 0;        
    glutPostRedisplay();      
  }
}

void motion(int x, int y)   
{
  dx = abs(x - start.x);
  dy = abs(y - start.y);


  if(x > start.x )
  {
    if( y > start.y)
    {
      if(dx > dy) dz = dy;
      else dz = dx;

      end.x = start.x + dz;  
      end.y = start.y + dz;
    }
    else
    {
      if(dx > dy) dz = dy;
      else  dz = dx;

      end.x = start.x + dz;  
      end.y = start.y - dz;
    }
  }
  else
  {
    if(y < start.y)
    {
      if(dx > dy) dz = dy;
      else  dz = dx;

      end.x = start.x - dz;  
      end.y = start.y - dz;
    }
    else
    {
      if(dx > dy) dz = dy;
      else  dz = dx;

      end.x = start.x - dz;  
      end.y = start.y + dz;
    }
  }


  glutPostRedisplay();
}

