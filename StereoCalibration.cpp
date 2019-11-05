/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
Oct. 3, 2008
Right to use this code in any way you want without warranty, support or any guarantee of it working.

BOOK: It would be nice if you cited it:
Learning OpenCV: Computer Vision with the OpenCV Library
by Gary Bradski and Adrian Kaehler
Published by O'Reilly Media, October 3, 2008

AVAILABLE AT:
http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
Or: http://oreilly.com/catalog/9780596516130/
ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

OPENCV WEBSITES:
Homepage:      http://opencv.org
Online docs:   http://docs.opencv.org
Q&A forum:     http://answers.opencv.org
GitHub:        https://github.com/opencv/opencv/
************************************************** */

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "shellCommand.h"

#define save 100 

using namespace cv;
using namespace std;

static int print_help()
{
	cout <<
		" Given a list of chessboard images, the number of corners (nx, ny)\n"
		" on the chessboards, and a flag: useCalibrated for \n"
		"   calibrated (0) or\n"
		"   uncalibrated \n"
		"     (1: use stereoCalibrate(), 2: compute fundamental\n"
		"         matrix separately) stereo. \n"
		" Calibrate the cameras and display the\n"
		" rectified results along with the computed disparity images.   \n" << endl;
	cout << "Usage:\n ./stereo_calib -w=<board_width default=9> -h=<board_height default=6> -s=<square_size default=1.0> <image list XML/YML file default=stereo_calib.xml>\n" << endl;
	return 0;
}


static void
StereoCalib(const vector<string>& imagelist, Size boardSize, float squareSize, bool displayCorners = true, bool useCalibrated = true, bool showRectified = true)
{
	if (imagelist.size() % 1 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	const int maxScale = 2;
	// ARRAY AND VECTOR STORAGE:
///*----------------------------------------------------------------------------------------
	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;
	Size imageSize;

	int i, j, k, nimages = (int)imagelist.size();

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		const string& filename = imagelist[i * 1];
		Mat img_o = imread(filename, 0);
		if (img_o.empty())
			break;
		if (imageSize == Size())
			imageSize = img_o.size();
		else if (img_o.size() != imageSize)
		{
			cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
			break;
		}
                cv::Rect rect_l(0,0,img_o.cols/2, img_o.rows);
                cv::Rect rect_r(img_o.cols/2,0, img_o.cols/2, img_o.rows);

                Mat mat_l, mat_r; 
                
                img_o(rect_l).copyTo(mat_l); 
                img_o(rect_r).copyTo(mat_r);

		for (k = 0; k < 2; k++)
		{
                        Mat img; 
                        if(k == 0)
                        {
                            mat_l.copyTo(img);
                        }
                        else 
                        {
                            mat_r.copyTo(img);                                    
                        }   
                        //imshow("mat",img);
                        //waitKey(0);                        
                           
			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];
			for (int scale = 1; scale < maxScale; scale++)
			{
				Mat timg;
				if (scale == 1)
                                {
					timg = img;
                                }
				else
                                {
					resize(img, timg, Size(), scale, scale, INTER_LINEAR);
                                }
				found = findChessboardCorners(timg, boardSize, corners,CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE); 
//					CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                                          
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
                                else
                                {
                                    std::cout<<"not find!"<<filename<<std::endl;
                                    remove(filename.c_str()); 
                                }
			}
			if (displayCorners)
			{
				//cout << filename << endl;
                                //std::cout<<"imagelist[i]" << imagelist[i];

				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, found);
				double sf = 4096. / MAX(img.rows, img.cols);
				resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR);

                                if(1)
                                {
                                    namedWindow("corners",2);
				    imshow("corners", cimg1);
				    char c = (char)waitKey(10);
				    if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
                                }

			}
			else
				putchar('.');
			if (!found)
				break;
			cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
					30, 0.01));
		}
		if (k == 2)
		{
			goodImageList.push_back(imagelist[i]);
			//goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";
	nimages = j;
	if (nimages < 2)
	{
		cout << "Error: too little pairs to run the calibration\n";
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (j = 0; j < boardSize.height; j++)
			for (k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}

	cout << "Running stereo calibration ...\n";

        imageSize.width = imageSize.width/2;

	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
	cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);

        string vin = "w"; 
        std::string common_path="../cameras_params/"+vin + "/";
        std::cout<< "parameters: " << common_path << std::endl; 

        std::string M1_path=common_path+"/_M1.xml";
        FileStorage storage(M1_path, FileStorage::READ); 
        storage["_M1"]>>cameraMatrix[0]; 
        storage.release(); 

        std::string M2_path=common_path+"/_M2.xml"; 
        storage.open(M2_path, FileStorage::READ);
        storage["_M2"]>>cameraMatrix[1];
        storage.release();

        std::string D1_path=common_path+"/_D1.xml";
        storage.open(D1_path, FileStorage::READ);
        storage["_D1"]>>distCoeffs[0];
        storage.release();

        std::string D2_path=common_path+"/_D2.xml";
        storage.open(D2_path, FileStorage::READ);
        storage["_D2"]>>distCoeffs[1];
        storage.release();

        LOG(INFO)<<"Load camera paramers end";
        std::cout<<"Load camera paramers end"<<std::endl;
        
	Mat R, T, E, F;

        
        std::cout<< "imageSize width : " << imageSize.width << std::endl; 
        std::cout<< "imageSize height: " << imageSize.height << std::endl; 

        std::cout<<"cameraMatrix[0]: " << cameraMatrix[0] << std::endl << "cameraMatrix[1]: " << cameraMatrix[1] <<std::endl <<  "distCoeffs[0]: " << distCoeffs[0] << std::endl << "distCoeffs[1]: "<< distCoeffs[1] << std::endl; 

	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F,
		CV_CALIB_USE_INTRINSIC_GUESS|CV_CALIB_RATIONAL_MODEL|CV_CALIB_FIX_FOCAL_LENGTH,//|CV_CALIB_FIX_INTRINSIC,//|CV_CALIB_FIX_PRINCIPAL_POINT,//|CV_CALIB_FIX_FOCAL_LENGTH,//|CV_CALIB_FIX_PRINCIPAL_POINT,//CV_CALIB_FIX_INTRINSIC,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,10000, 1e-5));
	cout << "done with RMS error=" << rms << endl;
        //return ; 
	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
        double errall = 0; 
	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];
		for (k = 0; k < 2; k++)
		{
			imgpt[k] = Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
					imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
                        errall += fabs(imagePoints[0][i][j].x*lines[1][j][0] + imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2])/sqrt(lines[1][j][0]*lines[1][j][0] + lines[1][j][1]*lines[1][j][1]) + fabs(imagePoints[1][i][j].x*lines[0][j][0] + imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2])/sqrt(lines[1][j][0]*lines[1][j][0] + lines[1][j][1]*lines[1][j][1]);
		}
		npoints += npt;
	}
        cout << "rms = " << sqrt(errall/(2*npoints)) << std::endl; 
	cout << "average epipolar err = " << err / npoints << endl;

	// save intrinsic parameters
	FileStorage fs("intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

#ifdef save
	fs.open("_M1.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_M1" << cameraMatrix[0]; 
		fs.release();
	}
	fs.open("_D1.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_D1" << distCoeffs[0];
		fs.release();
	}
	fs.open("_M2.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_M2" << cameraMatrix[1];
		fs.release();
	}
	fs.open("_D2.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_D2" << distCoeffs[1];
		fs.release();
	}
	fs.open("_R.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_R" << R;
		fs.release();
	}
	fs.open("_T.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_T" << T;
		fs.release();
	}
#endif

//----------------------------------------------------------------------------------*/
/*only rectify
        Mat cameraMatrix[2], distCoeffs[2];
	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;

//       int nimages = 100; 
//	imagePoints[0].resize(nimages);
//	imagePoints[1].resize(nimages);
//	objectPoints.resize(nimages);

        Size imageSize(4096,2160);

//	cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
//	cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);

        Mat R, T, E, F;

        
        FileStorage fs("_Ms.xml", FileStorage::READ);
        fs.release(); 
//-----------------------------------*/

        FileStorage fsRead("_Ms.xml", FileStorage::READ);
        fsRead.release(); 
        //read
	fsRead.open("_M1.xml", FileStorage::READ);
	if (fsRead.isOpened())
	{
		fsRead["_M1"] >> cameraMatrix[0]; 
                std::cout<<"M1: "<<cameraMatrix[0]<<std::endl; 
		fsRead.release();
	}
	fsRead.open("_D1.xml", FileStorage::READ);
	if (fsRead.isOpened())
	{
		fsRead["_D1"] >> distCoeffs[0];
                std::cout<<"D1: "<<distCoeffs[0]<<std::endl; 
		fsRead.release();
	}
	fsRead.open("_M2.xml", FileStorage::READ);
	if (fsRead.isOpened())
	{
		fsRead["_M2"] >> cameraMatrix[1];
                std::cout<<"M2: "<<cameraMatrix[1]<<std::endl;
		fsRead.release();
	}
	fsRead.open("_D2.xml", FileStorage::READ);
	if (fsRead.isOpened())
	{
		fsRead["_D2"] >> distCoeffs[1];
                std::cout<<"D2: "<<distCoeffs[1]<<std::endl;
		fsRead.release();
	}
	fsRead.open("_R.xml", FileStorage::READ);
	if (fsRead.isOpened())
	{
		fsRead["_R"] >> R;
		fsRead.release();
	}
	fsRead.open("_T.xml", FileStorage::READ);
	if (fsRead.isOpened())
	{
		fsRead["_T"] >> T;
                std::cout<<"T: "<<T<<std::endl;
		fsRead.release();
	}

	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 0, imageSize, &validRoi[0], &validRoi[1]);
        Mat mapx11, mapy11; 
        Mat mapx22, mapy22; 

        Size imageSizes(4096,2160);

        initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1,
                             imageSizes, CV_16SC2, mapx11, mapy11);
        initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2,
                             imageSizes, CV_16SC2, mapx22, mapy22);      


        string path= "/media/ibd01/Binocular/wuhan_calibration_20191029/标定场/0027-1-10-0001-20191029-155112/1-Runtime/14-Cam/test_combine/*.jpg";    
        string cmd = "ls " + path; 

        vector<string>  imagelists;

        IBD_SD::CShellCommand commandObj;
        bool ok = commandObj.getformatList(cmd, imagelists);


        string pathout = "/home/ibd01/DVISION/src/tools/calibration/data_rectify/"; 
        for(int i = 0; i < imagelists.size() ;i++)
        {
            //std::cout<<imagelists[i]<< std::endl; 
            int pos = imagelists[i].find_last_of("//");

            string name = imagelists[i].substr(pos + 1); 
            //std::cout<< "name: " << name << std::endl; 
 
            Mat matComb = imread(imagelists[i]); 
 
            cv::Rect rect_l(0,    0,    matComb.cols/2,     matComb.rows);
            cv::Rect rect_r(matComb.cols/2,    0,   matComb.cols/2, matComb.rows);

            Mat imageleft(matComb.rows, matComb.cols/2, CV_8UC3, Scalar::all(0));
            Mat imageleftnew(matComb.rows, matComb.cols/2, CV_8UC3, Scalar::all(0)); 
            Mat imageright(matComb.rows, matComb.cols/2, CV_8UC3, Scalar::all(0));
            Mat imagerightnew(matComb.rows, matComb.cols/2, CV_8UC3, Scalar::all(0)); 

            matComb(rect_l).copyTo(imageleft); 
            matComb(rect_r).copyTo(imageright);

            remap(imageleft, imageleftnew, mapx11, mapy11, INTER_LINEAR);
            remap(imageright, imagerightnew, mapx22, mapy22, INTER_LINEAR);

            Mat matOut(matComb.rows, matComb.cols, CV_8UC3, Scalar::all(0)); 

            imageleftnew.copyTo(matOut(rect_l)); 
            imagerightnew.copyTo(matOut(rect_r));
            for(int i = 0 ; i < matOut.rows; i = i + (matOut.rows/10) )
            {
                Point pt1(0, i); 
                Point pt2(2*matOut.cols - 1, i);
                line(matOut, pt1, pt2, Scalar(255,0,0), 2, 8);
            }
            string name_out = pathout + name; 
            std::cout<< name_out << std::endl; 
            imwrite(name_out, matOut); 
            namedWindow("show",2);
            imshow("show", matOut);
            waitKey(10);
             
        }

	fs.open("extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	fs.open("_Q.xml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs <<"_Q" << Q;
		fs.release();
	}

}


static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

int main(int argc, char** argv)
{
	Size boardSize;
	string imagelistfn;
	bool showRectified;
	cv::CommandLineParser parser(argc, argv, "{w|15|}{h|11|}{s|50|}{nr||}{help||}{@input|stereo_calib.xml|}");
	if (parser.has("help"))
		return print_help();
	showRectified = !parser.has("nr");
	//imagelistfn = samples::findFile(parser.get<string>("@input"));
	boardSize.width = parser.get<int>("w");
	boardSize.height = parser.get<int>("h");
	float squareSize = parser.get<float>("s");
	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}
	vector<string> imagelist;
        //bool getformatList(std::string &cmd, std::vector<std::string> &fileList);
        string path= "/home/ibd01/DVISION/src/tools/calibration/data/*.jpg";
        string cmd = "ls " + path; 

        IBD_SD::CShellCommand commandObj;
        bool ok = commandObj.getformatList(cmd, imagelist);
        //std::cout<< "imagelist.size: " << imagelist.size(); 
        for(int i = 0 ; i < imagelist.size() ; i++)
        {
            //std::cout<< imagelist[i] << std::endl; 
        }

	//bool ok = readStringList(imagelistfn, imagelist);

	if (!ok || imagelist.empty())
	{
		cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
		return print_help();
	}

	StereoCalib(imagelist, boardSize, squareSize, true, true, showRectified);
	return 0;
}
