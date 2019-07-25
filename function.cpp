#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdlib.h>	
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <time.h>


//#include <math.h>

using namespace cv;
using namespace std;

int*** channel_array(Mat image, int ch)
{
	int i, j, k;

	int*** channel = (int ***)malloc(ch * sizeof(int **));

	for(i=0;i<ch;i++)
	{
		*(channel+i) = (int **)malloc(image.rows*sizeof(int*));
		for(j=0;j<image.rows;j++)
		{
			*(*(channel+i)+j) = (int *)malloc(image.cols*sizeof(int));
		}
	}

	for(k=0;k<ch;k++)
	{
		for(i=0;i<image.rows;i++)
		{
			for(j=0;j<image.cols;j++)
				{
					channel[k][i][j] = image.at<cv::Vec3b>(i,j)[k];
				}
		}
	}

	return channel;
}


int*** pchannel_array(Mat image, int p, int ch, int ***channel)
{
	int i, j, k;	
	
	int p_c = image.cols+p*2;
	int p_r = image.rows+p*2;

	int*** pchannel = (int***)malloc(ch*sizeof(int**));

	for(i=0;i<ch;i++)
	{
		*(pchannel+i) = (int**)malloc(p_r*sizeof(int*));
		for(j=0;j<p_r;j++)
		{
			*(*(pchannel+i)+j) = (int *)malloc(p_c*sizeof(int));
		}
	}


	for(k=0;k<ch;k++)
	{
		for(i=0;i<p_r;i++)
		{
			for(j=0;j<p_c;j++)
			{
				if(i<p)			
				pchannel[k][i][j]=0;

				else if(j<p)
				pchannel[k][i][j]=0;
	
				else if(i>=image.rows+p)
				pchannel[k][i][j]=0;

				else if(j>=image.cols+p)
				pchannel[k][i][j]=0;
			
				else
				pchannel[k][i][j] = channel[k][i-p][j-p];
			}
		}
	}

	
	return pchannel;
}


double*** con_array(Mat image, int f, int p, int s, int ch, int ***pchannel, double **filter)
{
	clock_t begin, end;
	begin = clock();

	int i, j, k;	
	
	int c_c = ((image.cols-f+2*p)/s)+1;
	int c_r = ((image.rows-f+2*p)/s)+1;

	double*** con = (double ***)malloc(ch*sizeof(double **));

	for(i=0;i<ch;i++)
	{
		*(con+i) = (double **)malloc(c_r*sizeof(double*));
		for(j=0;j<c_r;j++)
		{
			*(*(con+i)+j) = (double *)malloc(c_c*sizeof(double));
		}
	}
	
	for(k=0;k<ch;k++)
	{
		for(i=0;i<c_r;i++)
		{
			for(j=0;j<c_c;j++)
			{
				con[k][i][j]=0;
			}
		}
	}

	for(k=0;k<ch;k++)
	{
		for(int b=0;b<c_r;b++)
		{
			for(int a=0;a<c_c;a++)
			{
				for(i=0;i<f;i++)
				{
					for(j=0;j<f;j++)
					{
						con[k][b][a] = con[k][b][a] + pchannel[k][i+s*b][j+s*a]*filter[i][j];
					}
				}
			}	
		}
	}
	
	for(k=0;k<ch;k++)
	{
		for(int b=0;b<c_r;b++)
		{
			for(int a=0;a<c_c;a++)
			{
				if(con[k][b][a]<0)
				{
					con[k][b][a] = 0;
				}
				
				else if(con[k][b][a]>255)
				{
					con[k][b][a] = 255;
				}
			}
		}
	}

	end = clock();
	double time = (double)(end - begin)/CLOCKS_PER_SEC;
	printf("convolution 걸린시간 : %lf sec\n", time);

	Mat new_image(c_r, c_c, image.type());

	for(k=0;k<ch;k++)
	{
		for(i=0;i<new_image.rows;i++)
		{
			for(j=0;j<new_image.cols;j++)
			{
				new_image.at<cv::Vec3b>(i,j)[k] = con[k][i][j];
			}
		}
	}
	
	imwrite("test1.jpg",new_image);
	waitKey(0);	

	
	return con;
}

int*** max_array(Mat image, int f, int s, int ch, int ***channel)
{
	int i, j, k;	

	int m_c = (image.cols-f)/s+1;
	int m_r = (image.rows-f)/s+1;

	int*** max_p = (int ***)malloc(ch*sizeof(int **));

	for(i=0;i<ch;i++)
	{
		*(max_p+i) = (int **)malloc(m_r*sizeof(int*));
		for(j=0;j<m_r;j++)
		{
			*(*(max_p+i)+j) = (int *)malloc(m_c*sizeof(int));
		}
	}


	for(k=0;k<ch;k++)
	{
		for(i=0;i<m_r;i++)
		{
			for(j=0;j<m_c;j++)
			{
				max_p[k][i][j]=0;
			}
		}
	}


	for(k=0;k<ch;k++)
	{
		for(int b=0;b<m_r;b++)
		{
			for(int a=0;a<m_c;a++)
			{
				for(i=0;i<f;i++)
				{
					for(j=0;j<f;j++)
					{
						if(channel[k][i+s*b][j+s*a]>max_p[k][b][a])
						{	
							max_p[k][b][a] = channel[k][i+s*b][j+s*a];
						}
					}
				}
			}	
		}
	}

	Mat new_image2(m_r, m_c, image.type());


	for(k=0;k<ch;k++)
	{
		for(i=0;i<new_image2.rows;i++)
		{
			for(j=0;j<new_image2.cols;j++)
			{
				new_image2.at<cv::Vec3b>(i,j)[k] = max_p[k][i][j];
			}
		}
	}
	

	imwrite("test2.jpg",new_image2);
	waitKey(0);

	return max_p;
}

/*
double activation(const double sigma, Mat image, int ch)
{
	int i, j, k;

	int*** activation = (int ***)malloc(ch * sizeof(int **));

	for(i=0;i<ch;i++)
	{
		*(activation+i) = (int **)malloc(image.rows*sizeof(int*));
		for(j=0;j<image.rows;j++)
		{
			*(*(channel+i)+j) = (int *)malloc(image.cols*sizeof(int));
		}
	}

	for(k=0;k<ch;k++)
	{
		for(i=0;i<image.rows;i++)
		{
			for(j=0;j<image.cols;j++)
				{
					
					activation[k][i][j] = image.at<cv::Vec3b>(i,j)[k];
					activation[k][i][j] = 1/(1+exp(-activation[k][i][j]));
				}
		}
	}
	
	return activation;
}
*/



int main()
{
	
	int ***channel, ***pchannel, ***max_p;
	double **filter, ***con;
	int i, j, k;
	int ch, p, s;
	int f, ff, f1;

	Mat image;
	image = imread("test.jpg", IMREAD_COLOR);
	

	printf("채널의 수를 입력하세요 : ");
	scanf("%d",&ch);
	
	printf("padding 크기 입력하시오 : ");	
	scanf("%d", &p);

	printf("stride 크기를 입력하시오 : ");	
	scanf("%d", &s);

	printf("filter 종류를 선택하시오 : ");	
	printf("1 : 3x3 blur   2 : 5x5 blur   3 : edge   4 : max \n");	
	scanf("%d", &f1);

	printf("filter 크기를 입력하시오 : ");	
	scanf("%d", &f);

	printf("filter 값을 입력하세요\n");
	
	filter = (double**)malloc(f*sizeof(double*));

	for(i=0;i<f;i++)
	{
		*(filter+i) = (double*)malloc(f*sizeof(double));
	}

	for(i=0;i<f;i++)
	{
		for(j=0;j<f;j++)
		{
			scanf(" %d", &ff);
			filter[i][j]=ff;
		}
	}

	if(f1==1)
	{	
		for(i=0;i<f;i++)
		{
			for(j=0;j<f;j++)
			{
				filter[i][j] = filter[i][j]/16;
			}	
		}
		channel = channel_array(image, ch);
		pchannel = pchannel_array(image, p, ch, channel);
		con = con_array(image, f, p, s, ch, pchannel, filter);
	}
	
	if(f1==2)
	{	
		for(i=0;i<f;i++)
		{
			for(j=0;j<f;j++)
			{
				filter[i][j] = filter[i][j]/256;
			}	
		}
		channel = channel_array(image, ch);
		pchannel = pchannel_array(image, p, ch, channel);
		con = con_array(image, f, p, s, ch, pchannel, filter);
	}


	if(f1==3)
	{
		for(i=0;i<f;i++)
		{
			for(j=0;j<f;j++)
			{
				filter[i][j] = filter[i][j];
			}	
		}
		channel = channel_array(image, ch);
		pchannel = pchannel_array(image, p, ch, channel);
		con = con_array(image, f, p, s, ch, pchannel, filter);
	}

	if(f1==4)
	{
		for(i=0;i<f;i++)
		{
			for(j=0;j<f;j++)
			{
				filter[i][j] = filter[i][j];
			}	
		}
		channel = channel_array(image, ch);
		max_p = max_array(image, f, s, ch, channel);
	}

	
	int c_r = ((image.rows-f+2*p)/s)+1;
	int c_c = ((image.cols-f+2*p)/s)+1;


	Mat dst(c_r, c_c, image.type());
	Mat kernel = (Mat_<int>(3,3) << -1,-1,-1,-1,8,-1,-1,-1,-1);
	

	cv::filter2D(image, dst, -1, kernel, Point(-1, -1), 0, BORDER_CONSTANT);

	Mat conv(c_r, c_c, image.type());

	for(k=0;k<ch;k++)
	{
		for(i=0;i<c_r;i++)
		{
			for(j=0;j<c_c;j++)
			{
				conv.at<cv::Vec3b>(i,j)[k] = con[k][i][j];
			}
		}
	}
	
	

	Mat nnew(c_r, c_c, image.type());

	for(k=0;k<ch;k++)
	{
		for(i=0;i<c_r;i++)
		{
			for(j=0;j<c_c;j++)
			{
				nnew.at<cv::Vec3b>(i,j)[k] = conv.at<cv::Vec3b>(i,j)[k] - dst.at<cv::Vec3b>(i,j)[k];
			}
		}
	}

	
	imwrite("test4444.jpg", nnew);
	
	waitKey(0);


/////////////////////free

	if(f1==1 && f1==2 && f1==3)
	{
		for(i=0;i<f;i++)
		{
			free(*(filter+i));
		}
		free(filter);

		for(i=0;i<ch;i++)
		{
			for(j=0;j<image.rows;j++)
			{
				free(*(*(channel+i)+j));
			}
		free(*(channel+i));
		}
		free(channel);


		for(i=0;i<ch;i++)
		{
			for(j=0;j<image.rows+p*2;j++)
			{
				free(*(*(pchannel+i)+j));
			}
			free(*(pchannel+i));
		}	
		free(pchannel);
	
		for(i=0;i<ch;i++)
		{
			for(j=0;j<((image.rows-f+2*p)/s)+1;j++)
			{
				free(*(*(con+i)+j));
			}
			free(*(con+i));
		}
		free(con);
	}

	if(f1==4)
	{
		for(i=0;i<f;i++)
		{
			free(*(filter+i));
		}
		free(filter);

		for(i=0;i<ch;i++)
		{
			for(j=0;j<image.rows;j++)
			{
				free(*(*(channel+i)+j));
			}
			free(*(channel+i));
		}
		free(channel);


		for(i=0;i<ch;i++)
		{
			for(j=0;j<(image.rows-f)/s+1;j++)
			{
				free(*(*(max_p+i)+j));
			}
			free(*(max_p+i));
		}	
		free(max_p);
	}







	return 0;
}
