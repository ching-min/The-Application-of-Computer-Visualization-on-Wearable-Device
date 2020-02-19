/****    Example of server using TCP protocol   */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <errno.h>
#include <arpa/inet.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>


#include <iostream>
#include <netdb.h>
#include <fstream>
#include <sstream>

#include "cv.h"
#include "cxcore.h"
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
//#include "highgui.h"

using namespace cv;
using namespace cv::face;
using namespace std;



#define MAXLINE 1536000

/*
void str_echo(int);
void err_dump(char*);
int readline(register int, register unsigned char*);
int written(register int, register char*, register int);
*/

int saveNum=0;
int processNum = 0;


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}


//資料訓練
Ptr<LBPHFaceRecognizer> model;
void training()
{
	string fn_csv = "database.csv";
	vector<Mat> images;
    vector<int> labels;
	try 
	{
		read_csv(fn_csv, images, labels);
	} 
	catch (cv::Exception& e) 
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if(images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(Error::StsError, error_message);
	}
    model = createLBPHFaceRecognizer();
    model->train(images, labels);
}


//回傳人臉編號
int returnNumber( CvRect *rec, IplImage *image )
{ 
	//裁切圖片
	cvSetImageROI( image, cvRect( rec->x, rec->y, rec->width, rec->height ) );
	
	//儲存圖片
	char saveName[64];
	sprintf( saveName, "result_%d.jpg", saveNum );
  	cvSaveImage(saveName,image,0);
	saveNum++;  

	//轉成Mat
	Mat mat = cvarrToMat(image);
	Mat greyMat;
	cvtColor(mat, greyMat, COLOR_BGR2GRAY);

	//取得人臉編號
	int predictedLabel = -1;
	predictedLabel = model->predict(greyMat);

	//還原圖片
	cvResetImageROI(image);  

	return predictedLabel;

}

//將名字寫在人臉旁邊
void putName( IplImage *image, int label, int x, int y )
{
	CvScalar color = CV_RGB( 255, 0, 0 );
	CvPoint pos = cvPoint( x, y );
	CvFont font = cvFont( 3, 1 );
	char *text;
	switch( label )
	{
		case 1:	text = "Teresa"; break;
		case 2: text = "Dog";  	 break;
		default: text = "UNKNOWN"; break;
	}
	cvPutText( image, text, pos, &font, color );
}


void err_dump(char *a)
{
int i;
        for (i=0; i<=20; i++)
            {  printf("%c", *a);
               a++;
            }
}

int eread(int fd, void *buf, size_t count){
	int n, ret = count;
	while( (n = read(fd, buf, count)) != count){
		if(n == 0){
			printf("read fd closed\n");
			return 0;
		}else if(n < 0){
			printf("ERROR read fd\n");
			return n;
		}
		buf += n;
		count -= n;
	}
	return ret;
}


int readline(register int fd, register char *ptr)
{ 
       int n, rc;
       char c,str[10];
       int i=0;

       while((rc = read(fd, &c, 1))>0)
       {
        if(c=='s')
	{
	  str[i] = '\0';
	  break;
	}
        else
        {
          str[i++]=c;
        }
       }

       int str_len=atoi(str);

       n = eread( fd, ptr, str_len );

      return n;
}


          
int written(register int fd, register char *ptr, register int nbytes)
{
int   nleft, nwritten;

      nleft = nbytes;
      while (nleft > 0) {
                nwritten = write(fd, ptr, nleft);
                if (nwritten <=0)
                    return(nwritten);
 
                 nleft -= nwritten;
                 ptr   += nwritten;
       }
       return(nbytes - nleft);

}


void str_echo(int sockfd)
{  
	int       n, len;
	unsigned char      line[MAXLINE];
	char* cont = "d";

	for( int i = 0; i < 5; i++ )
		int s = send( sockfd, cont, strlen( cont ), 0 ); 
   

	for ( ; ; ) {
	  memset( line, 0, MAXLINE );
 	  n = readline(sockfd, line);//////////////////////////////////////
          if  (n == 0)
	  {
	     printf("Q__________________________Q\n");
             return;
	  }
          else  if (n<0)
               err_dump("Str_echo: readline error");

	   unsigned char *tmp = line;
	   IplImage *image_detect;
	   CvMat cvmat = cvMat( 720, 1280, CV_8UC3, (void*)tmp );
	   image_detect = cvDecodeImage(&cvmat, 1);
	   

	   string cascade_name="haarcascade_frontalface_alt.xml";
	   //load cascade 
	   //detect face 
	   CvHaarClassifierCascade *classifier=(CvHaarClassifierCascade*)cvLoad(cascade_name.c_str(),0,0,0);
	   if(!classifier){//no face 
			printf("error to load classifier cascade!\n");
			return ;  
	   }
	   //create mem 
	   CvMemStorage *facesMemStorage=cvCreateMemStorage(0);

	   //initial mem
	   cvClearMemStorage(facesMemStorage);

	   if( image_detect == NULL )
	   {
  		cvReleaseImage( &image_detect );
   		cvReleaseMemStorage( &facesMemStorage );
   		cvReleaseHaarClassifierCascade( &classifier );  
		continue;
	   }

	   //detect face
	   CvSeq* face=cvHaarDetectObjects( image_detect, classifier,facesMemStorage,1.1,3,CV_HAAR_DO_CANNY_PRUNING,cvSize(50,50) );

	   if(face){//exist face
		int i=0;
		for(i=0;i<face->total;i++){
			CvPoint p1,p2;
			CvRect *rec=(CvRect*)cvGetSeqElem(face,i);
			
			p1.x=rec->x;
			p2.x=rec->x+rec->width;
			p1.y=rec->y;
			p2.y=rec->y+rec->height;
			cvRectangle( image_detect, p1, p2, CV_RGB(255,0,0),3,8,0 );
			
			putName( image_detect, returnNumber( rec, image_detect ), p1.x, p1.y );
		}
   }

   cvShowImage("OAO", image_detect);
   cvWaitKey(1);
   
   //Release Memory
   cvReleaseImage( &image_detect );
   cvReleaseMemStorage( &facesMemStorage );
   cvReleaseHaarClassifierCascade( &classifier );  

   //unblock client
   int s = send( sockfd, cont, strlen( cont ), 0 ); 
}

} 


int main(int argc, char *argv[])
{
	training();

	cvNamedWindow("OAO",1);
	IplImage* tmp = cvCreateImage( cvSize( 1280, 720 ), 8, 3);
	cvShowImage("OAO", tmp);
	cvWaitKey(1);
	
    int sockfd, newsockfd, clilen, childpid; 
    struct sockaddr_in   cli_addr, serv_addr;
	int port;
        //  pname = argv[0];
        /*** Open  a TCP socket (an Internet stream socket).   ****/
	if ( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
		err_dump("server: can't open stream socket");
	if(argc != 2){
		printf("usage: ./server <port>\n");
		exit(-1);
	}
	port = atoi(argv[1]);

	/**  Bind our local address so that the client can send to us.  **/

	bzero((char *) &serv_addr, sizeof(serv_addr));
	serv_addr.sin_family       =  AF_INET;
	serv_addr.sin_addr.s_addr  =  htonl(INADDR_ANY);
	serv_addr.sin_port         =  htons(port);

	if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) <0)
		err_dump("server: can't bind local address");

	listen(sockfd, 5);
	for (; ; ){
		/*
		*  Wait for a connection from a client process.
		*  This is an example of a concurrent server.
		*/
		clilen = sizeof(cli_addr);

		newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, (socklen_t*)&clilen);

		if (newsockfd < 0)
			err_dump("server: accept error");
		if( (childpid = fork())<0)
			err_dump("server: fork error");
		else  if (childpid == 0) {
			close(sockfd);
			str_echo(newsockfd);
			exit(0);
		}

	}
	close(newsockfd);
}


