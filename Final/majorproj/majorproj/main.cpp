#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <iostream>
#include <conio.h>
#include <math.h>
#include <fstream>
#include <string.h>
#include <stdio.h>

int NumClasses;
int K;

struct neuron
{
	//char classname[20];
	int classId;
	int points[585];
	double KnnDist;
	neuron * next;
	neuron * KnnNext;
} * KnnListHead;

struct classtype
{
	char classname[20];
	int classId;
	neuron *head;
	neuron *tail;
	int numSamples;
	classtype * next;
};

float R;
int rotate_array[9][2] = {{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1}};
int fp[585];
classtype * classlist;


void init(){
	classlist = NULL;
	NumClasses = 0;
	R = 101;
	K = 5;
}
void getpixels(int x, int y, int from, int to, int index, IplImage * image) {
	int i,j,p,q;
	int temp;
	for(i=from; i<=to; i++) {
		p=rotate_array[i][0];
		q=rotate_array[i][1];
		for(j=0; j<6; j++) {
			temp=(((unsigned char*)(image->imageData + (x+p)*image->widthStep))[y+q])-(((unsigned char*)(image->imageData + (x)*image->widthStep))[y]);
			fp[index]=temp;
			x+=p;
			y+=q;
			index++;
		}
	}
	
}

void extractor(char * file)
{
	//IplImage * img1 = cvLoadImage("D:\\faces\\test.jpg");
	IplImage * image1 = cvLoadImage(file);
	IplImage * img1 = cvCreateImage(cvSize(640,423),8,3);
	cvResize(image1,img1);
	//cvNamedWindow("image");
	//cvShowImage( "image", img1 );
	//cvWaitKey();
	IplImage* img4 = cvCreateImage(cvSize(640,423),8,3);
	cvResize(img1,img4);

	IplImage* img3 = cvCreateImage(cvGetSize(img4),8,3);
	cvCopyImage(img4,img3);
	cvCvtColor(img4,img4,CV_RGB2YCrCb);
	//cvShowImage( "image", img4 );
	//cvWaitKey();
	
	
	
	IplImage* img2 = cvCreateImage(cvGetSize(img4),8,1);
	cvInRangeS(img4,cvScalar(100,50,100),cvScalar(180,120,255),img2);
	
	//cvShowImage( "image", img2 );
	//cvWaitKey();
	
	cvErode(img2,img2,0,3);
	//cvShowImage( "image", img2 );
	//cvWaitKey();
	cvDilate(img2,img2,0,5);
	//cvShowImage( "image", img2 );
	//cvWaitKey();

	CvMemStorage * mem1 = cvCreateMemStorage();
	CvSeq* seq1 = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvPoint),mem1);
	int n1 = cvFindContours(img2,mem1,&seq1,88,CV_RETR_EXTERNAL);
		
	double area = 0;
	CvRect rect;
	CvBox2D box;
	cvZero(img2);
	for(CvSeq* contour = seq1;contour!=NULL;contour=contour->h_next) {
		if(area < cvContourArea(contour)) {
			area = cvContourArea(contour);
			rect = cvBoundingRect(contour);
			box = cvFitEllipse2(contour);

		}
		
	}
	CvRect rect1 = cvRect(box.center.x - box.size.width/2,box.center.y - box.size.height/2,box.size.width,box.size.height);
	if(rect1.x < 0)
		rect1.x=0;
	if(rect1.y < 0)
		rect1.y=0;
	if(rect1.x + rect1.width > img3->width)
		rect1.width=img3->width - rect.x;
	if(rect1.y + rect1.height > img3->height)
		rect1.height=img3->height - rect.y;
	//cvDrawRect(img3,cvPoint(rect.x,rect.y),cvPoint(rect.x + rect.width,rect.y + rect.height),cvScalar(0));
	//cvDrawRect(img3,cvPoint(rect1.x,rect1.y),cvPoint(rect1.x + rect1.width,rect1.y + rect1.height),cvScalar(0));

	
	CvRect rect2 =  CvRect(cv::Rect(rect) | cv::Rect(rect1));
	//cvShowImage( "image", img3 );
	//cvWaitKey();
	
	cvSetImageROI(img3, rect2);
	IplImage * face = cvCreateImage(cvSize(rect2.width,rect2.height), IPL_DEPTH_8U, img3->nChannels);
	
	cvCopy(img3, face);
	//cvShowImage( "image", face );
	//cvWaitKey();

	IplImage * greyface = cvCreateImage(cvGetSize(face), IPL_DEPTH_8U,1);
	cvCvtColor(face,greyface,CV_RGB2GRAY);
	//cvShowImage( "image", greyface );
	//cvWaitKey();
	cvSmooth(greyface,greyface);
	cvSmooth(greyface,greyface);
	//cvEqualizeHist(greyface,greyface);	
	IplImage * features = cvCreateImage(cvGetSize(face), IPL_DEPTH_8U,1);
	cvCopyImage(greyface,features);

	//cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	//cvReleaseImage(&img3);
	cvReleaseImage(&img4);


	IplImage * graph = cvCreateImage(cvSize(255,greyface->height), IPL_DEPTH_8U,1);
	cvSet(graph,cvScalar(255,255,255));
	int * val = new int[greyface->height];
	for( int i=0; i<greyface->height/2; i++ )
	 {
		val[i] = 0;
		for( int j=0; j<greyface->width; j++ )
         {   
            val[i] += ((unsigned char*)(greyface->imageData + i*greyface->widthStep))[j];   
        }
		val[i] = val[i]/greyface->width;
	
		cvDrawLine(graph,cvPoint(0,i),cvPoint(val[i],i),cvScalar(0));
    }
	//cvNamedWindow("graph",1);
	//cvShowImage("graph",graph);
	//cvWaitKey();

	IplImage * graph2 = cvCreateImage(cvSize(510,greyface->height), IPL_DEPTH_8U,1);
	cvSet(graph2,cvScalar(255,255,255));
	IplImage * graph3 = cvCreateImage(cvSize(510,greyface->height), IPL_DEPTH_8U,1);
	cvSet(graph3,cvScalar(255,255,255));
	int * diff = new int[greyface->height];
	
	for(int i=0;i<greyface->height/2-1;i++) {
		diff[i+1]=val[i+1]-val[i];
		//if((diff[i+1]<1 && diff[i+1]>0) || (diff[i+1]>-1 && diff[i+1]<0))
			//diff[i+1]=0;
		cvDrawLine(graph2,cvPoint(255,i+1),cvPoint(255+diff[i+1],i+1),cvScalar(0));
	}
	//cvNamedWindow("graph2",1);
	//cvShowImage("graph2",graph2);
	//cvWaitKey();

	int count = 0;
	int sign = 1;
	int eye=0;
	for(int i=1;i<greyface->height/2;i++) {
		if(diff[i]>0 && sign==0) {
			count++;
			if(count==2) {
				eye=i;
				break;
			}
			sign=1;
		}
		else if(diff[i]<0) {
			sign=0;
		}
	}

	cvReleaseImage(&graph);
	cvReleaseImage(&graph2);

	IplImage * graphv1 = cvCreateImage(cvSize(greyface->width,255), IPL_DEPTH_8U,1);
	cvSet(graphv1,cvScalar(255));
	int * valv = new int[greyface->width];
	for( int j=0; j<greyface->width; j++ )
	 {
		valv[j] = 0;
		for( int i=eye-0.025*greyface->height; i<eye+0.025*greyface->height; i++ )
         {   
            valv[j] += ((unsigned char*)(greyface->imageData + i*greyface->widthStep))[j];   
        }
		valv[j] = valv[j]/(0.05*greyface->height);
	
		cvDrawLine(graphv1,cvPoint(j,0),cvPoint(j,valv[j]),cvScalar(0));
	
    }

	//cvShowImage("graph",graphv1);
	//cvWaitKey();

	IplImage * graphv2 = cvCreateImage(cvSize(greyface->width,510), IPL_DEPTH_8U,1);
	cvSet(graphv2,cvScalar(255));
	int *diffv = new int[greyface->width];
	for(int i = 0; i<greyface->width-1;i++) {
		diffv[i+1] = valv[i+1]-valv[i];
	
		if((diffv[i+1]>-1 && diffv[i+1]<0) || (diffv[i+1]<1 && diffv[i+1]>0))
			diffv[i+1]=0;
	
		cvDrawLine(graphv2,cvPoint(i+1,255),cvPoint(i+1,255+diffv[i+1]),cvScalar(0));
	}
	
	//cvShowImage("graph2",graphv2);
	//cvWaitKey();

	int mid=valv[greyface->width/4];
	for(int i=greyface->width/5;i<greyface->width*4/5;i++)
	{
		if(valv[i]>valv[mid]) {
			mid=i;
		}
	}
	sign=1;
	count = 0;
	int peak = 0,max=0;
	int eyeh1=0,eyeh2=0;
	int i;
	for(i=mid;i>0;i--) {
		if(diffv[i]>0 && sign==0) {
			count++;
			if(count==2) {
				break;
			}
			max = 0;
			sign=1;
		}
		else if(diffv[i]<0) {
			sign=0;
			if(diffv[i]<max) {
				max = diffv[i];
				peak = i;
			}
		}
	}
	eyeh1=peak;
	
	sign=1;
	count=0;
	for(int i=mid;i<greyface->width;i++) {
		if(diffv[i]>0 && sign==0) {
			count++;
			if(count==2) {
				eyeh2=i;
				break;
			}
			sign=1;
		}
		else if(diffv[i]<0) {
			sign=0;
		}
	}

	cvDrawLine(features,cvPoint(0,eye),cvPoint(greyface->width,eye),cvScalar(255));
	cvDrawLine(features,cvPoint(eyeh1,0),cvPoint(eyeh1,greyface->height),cvScalar(255));
	cvDrawLine(features,cvPoint(eyeh2,0),cvPoint(eyeh2,greyface->height),cvScalar(255));
	cvDrawLine(features,cvPoint((eyeh1+eyeh2)/2,0),cvPoint((eyeh1+eyeh2)/2,greyface->height),cvScalar(255));
	//cvShowImage("image",features);
	//cvWaitKey();

	cvReleaseImage(&graphv1);
	cvReleaseImage(&graphv2);

	IplImage * graphn1 = cvCreateImage(cvSize(255,greyface->height), IPL_DEPTH_8U,1);
	cvSet(graphn1,cvScalar(255,255,255));
	int * valn = new int[greyface->height];
	for(int i=eye+0.05*greyface->height;i<greyface->height;i++) {
		valn[i]=0;
		for( int j=eyeh1; j<eyeh2; j++ )
			{   
            valn[i] += ((unsigned char*)(greyface->imageData + i*greyface->widthStep))[j];   
        }
		valn[i]=valn[i]/(eyeh2-eyeh1);
	
		cvDrawLine(graphn1,cvPoint(0,i),cvPoint(valn[i],i),cvScalar(0));
	}
	
	//cvShowImage("graph",graphn1);
	//cvWaitKey();

	IplImage * graphn2 = cvCreateImage(cvSize(510,greyface->height), IPL_DEPTH_8U,1);
	cvSet(graphn2,cvScalar(255,255,255));
	int * diffn = new int[greyface->height];
	for(int i = eye + 0.05*greyface->height; i<greyface->height;i++) {
		diffn[i+1] = valn[i+1]-valn[i];
	
		if(diffn[i+1]>-3 && diffn[i+1]<0)
			diffn[i+1]=0;
		cvDrawLine(graphn2,cvPoint(255,i+1),cvPoint(255+diffn[i+1],i+1),cvScalar(0));
	}
	//cvShowImage("graph2",graphn2);
	//cvWaitKey();
	
	
	sign = -1;
	int nose=0;
	for(int i= eye + (0.05*greyface->height)+3 ;i<greyface->height;i++) {
		if(diffn[i]>0 && sign==0) {
			
				nose=i;
				break;
			
			
		}
		else if(diffn[i]<0 && sign==1) {
			sign=0;
		}
		else if(diffn[i]>0) {
			sign=1;
		}
		
	}

	cvDrawLine(features,cvPoint(0,nose),cvPoint(greyface->width,nose),cvScalar(255));
	//cvShowImage("image",features);
	//cvWaitKey();

	sign = 1;
	int mouth=0;
	for(int i=nose+0.075*greyface->height;i<greyface->height;i++) {
		if(diffn[i]>0 && sign==0) {
			
				mouth=i;
				break;
			
			
		}
		else if(diffn[i]<0) {
			sign=0;
		}
		else if(diffn[i]>0) {
			sign=1;
		}
		
	}
	
	cvDrawLine(features,cvPoint(0,mouth),cvPoint(greyface->width,mouth),cvScalar(255));
	//cvShowImage("image",features);
	//cvWaitKey();
	
	cvReleaseImage(&graphn1);
	cvReleaseImage(&graphn2);

	IplImage * borders = cvCreateImage(cvGetSize(greyface),IPL_DEPTH_32F,1);
	IplImage * borders1 = cvCreateImage(cvGetSize(greyface),8,1);
	
	cvCornerHarris(greyface,borders,7,9);
	cvThreshold(borders,borders1,10,255,CV_THRESH_BINARY);
	cvDilate(borders1,borders1);
	
	//cvShowImage("image",borders1);
	//cvWaitKey();

	int mouthh1=0,mouthh2=0,pixel=0;
	//for(int i=(eyeh1+eyeh2)/2 + 30;i<greyface->width;i++) {
	for(int i=(eyeh2 + 5)>borders->width?borders->width:eyeh2 + 5;i>(eyeh1+eyeh2)/2;i--) {
		pixel = ((unsigned char*)(borders1->imageData + mouth*borders1->widthStep))[i];
		if(pixel!=0) {
			mouthh2=i;
			break;
		}
		
	}
	
	pixel=0;
	//for(int i=(eyeh1+eyeh2)/2 - 30;i>0;i--) {
	for(int i=(eyeh1 - 5)<0?0:eyeh1 - 5;i<(eyeh1+eyeh2)/2;i++) {
		pixel = ((unsigned char*)(borders1->imageData + mouth*borders1->widthStep))[i];
		if(pixel!=0) {
			mouthh1=i;
			break;
		}
	}
	
	cvDrawLine(features,cvPoint(mouthh1,0),cvPoint(mouthh1,greyface->height),cvScalar(200));
	cvDrawLine(features,cvPoint(mouthh2,0),cvPoint(mouthh2,greyface->height),cvScalar(200));
	cvDrawLine(features,cvPoint((mouthh1+mouthh2)/2,0),cvPoint((mouthh1+mouthh2)/2,greyface->height),cvScalar(200));
	//cvShowImage("image",features);
	//cvWaitKey();
	cout<<"----------\n";
	cout<<"width:"<<features->width<<"\n";
	cout<<"height:"<<features->height<<"\n";
	cout<<"eyes:"<<eye<<" "<<eyeh1<<" "<<eyeh2<<"\n";
	cout<<"nose:"<<nose<<"\n";
	cout<<"mouth:"<<mouth<<" "<<mouthh1<<" "<<mouthh2<<"\n";
	cout<<"----------\n";
	
	rect2.x/=2;
	rect2.y/=2;
	rect2.width/=2;
	rect2.height/=2;
	
	rect2.x-=7;
	rect2.width+=14;

	cvSetImageROI(img1, rect2);
	IplImage * pixelsource = cvCreateImage(cvSize(rect2.width,rect2.height), IPL_DEPTH_8U, img1->nChannels);
	
	cvCopy(img1, pixelsource);

	IplImage * original = cvCreateImage(cvGetSize(pixelsource),8,1);
	cvCvtColor(pixelsource,original,CV_RGB2GRAY);

	eyeh1/=2;
	eyeh2/=2;
	eye/=2;
	nose/=2;
	mouthh1/=2;
	mouthh2/=2;
	mouth/=2;

	eyeh1+=7;
	eyeh2+=7;
		
	mouthh1+=7;
	mouthh2+=7;
	

	/*float scale = 100/(float)(eyeh2-eyeh1);
	IplImage * scaled = cvCreateImage(cvSize((greyface->width)*scale, (greyface->height)*scale),8,1);
	cvResize(original,scaled,CV_INTER_LINEAR);
	//cvShowImage("image",scaled);
	//cvWaitKey();

	eyeh1*=scale;
	eyeh2*=scale;
	eye*=scale;
	nose*=scale;
	mouthh1*=scale;
	mouthh2*=scale;
	mouth*=scale;
	*/
	getpixels(eyeh1,eye,0,7,0,original);
	getpixels(eyeh2,eye,0,7,120,original);//getpixels(eyeh2,eye,0,7,88,original);
	getpixels((eyeh1+eyeh2)/2,eye,0,7,240,original);//getpixels((eyeh1+eyeh2)/2,eye,0,7,176,original);
	
	getpixels((mouthh1+mouthh2)/2,mouth,0,7,315,original);//getpixels((mouthh1+mouthh2)/2,mouth,0,7,231,original);
	getpixels(mouthh1,mouth,0,4,435,original);//getpixels(mouthh1,mouth,0,4,319,original);
	getpixels(mouthh2,mouth,4,8,510,original);//getpixels(mouthh2,mouth,4,8,374,original);
	/*5,585
	getpixels(eyeh1,eye,0,7,0,original);//getpixels(eyeh1,eye,0,7,0,original);
	getpixels(eyeh2,eye,0,7,40,original);//getpixels(eyeh2,eye,0,7,88,original);
	getpixels((eyeh1+eyeh2)/2,eye,0,7,80,original);//getpixels((eyeh1+eyeh2)/2,eye,0,7,176,original);
	
	getpixels((mouthh1+mouthh2)/2,mouth,0,7,105,original);//getpixels((mouthh1+mouthh2)/2,mouth,0,7,231,original);
	getpixels(mouthh1,mouth,0,4,145,original);//getpixels(mouthh1,mouth,0,4,319,original);
	getpixels(mouthh2,mouth,4,8,170,original);//getpixels(mouthh2,mouth,4,8,374,original);
	*/
	/*getpixels((mouthh1+mouthh2)/2,nose,2,6,264,original);
	getpixels((mouthh1+mouthh2)/2,mouth,0,7,319,original);
	getpixels(mouthh1,mouth,0,4,407,original);
	getpixels(mouthh2,mouth,4,8,462,original);
	*/
	


}



neuron * createNeuron(int Id, int point[585])
{
	neuron * ptr = new neuron;

	for(int i=0;i<585;i++) {
	ptr->points[i] = point[i];// + 50*Id;
	}
	
	ptr->classId = Id;
	ptr->next=NULL;
	ptr->KnnNext=NULL;
	ptr->KnnDist=0;
	return ptr;
}

classtype * addClass(char classname[20]) {
	classtype * ptr = new classtype;
	strcpy(ptr->classname,classname);
	ptr->head=NULL;
	ptr->tail=NULL;
	ptr->next=NULL;
	ptr->numSamples = 0;
	if(classlist == NULL) {
		classlist = ptr;
		classlist->classId=0;
		
	}
	else{
		classtype * temp = classlist;
		while(temp->next!=NULL) {
			temp=temp->next;
		}
		temp->next=ptr;
		ptr->classId=(temp->classId) + 1;
		 
	}
	NumClasses++;
	return ptr;
}

classtype * findClass(int Id) {

	classtype * temp = classlist;
	while(temp!=NULL && (temp->classId)!=Id) {
		temp=temp->next;
	}
	return temp;
}

double dist(neuron *neuron1, neuron *neuron2) {
	
	double dist=0;
	//cout<<"n1 :"<<neuron1->points[667]<<" "<<neuron1->points[668]<<" "<<neuron1->points[669]<<"\n";
	//cout<<"n2 :"<<neuron2->points[667]<<" "<<neuron2->points[668]<<" "<<neuron2->points[669]<<"\n";
	for(int i=0;i<585;i++) {
		dist+=pow((neuron1->points[i])-(neuron2->points[i]),2.0);
	}
	dist = sqrt(dist);
	return dist;
}

void addTrainedSample(neuron * sample) {
	classtype * sampleclass = findClass(sample->classId);
	

	if(sampleclass->head == NULL) {
		sampleclass->head = sample;
		sampleclass->tail = sample;
		sampleclass->head->next = NULL;

	}
	else if(sampleclass->head == sampleclass->tail) {
		sampleclass->head->next=sample;
		sampleclass->tail = sample;
	}
	else {
		
		sampleclass->tail->next=sample;
		sampleclass->tail=sample;
		
	}
	sample->next=NULL;
	sampleclass->numSamples++;
}
void addSample(neuron * sample) {
	
	//addTrainedSample(sample);
	
	classtype * sampleclass = findClass(sample->classId);
	if(sampleclass == NULL) {
		
		cout<<"Class not found!";
		return;
	}
	if(sampleclass->head == NULL) {
		sampleclass->head = sample;
		sampleclass->tail = sample;
	}
	else if(sampleclass->head->next == NULL) {
		sample->next=sampleclass->head;
		sampleclass->tail=sampleclass->head;
		sampleclass->head = sample;
		
	}
	else {
		neuron * ptr1 = sampleclass->head;
		neuron * ptr2 = sampleclass->head->next;
		while(ptr2!=NULL && dist(ptr1,sample)>dist(ptr2,sample)) {

			ptr1=ptr2;
			ptr2 = ptr2->next;
		}
		if(ptr2 == NULL) {
			sampleclass->tail = sample;
			ptr1->next = sample;
		}
		else {
			if(dist(ptr1,sample)>dist(ptr1,ptr2)){
				sample->next=ptr2->next;
				ptr2->next=sample;
			}
			else {
				ptr1->next = sample;
				sample->next = ptr2;
			}
		}

	}
	sampleclass->numSamples++;
	
}

double dot(int points1[585], int points2[585]) {

	double dotproduct = 0;
	for(int i=0;i<585;i++) {
		dotproduct += (points1[i] * points2[i]);
	}
	return dotproduct;

}
int * diff(int points1[585],int points2[585]) {
	
	int * difference = new int[585];
	for(int i = 0; i<585; i++) {
		difference[i]=points1[i] - points2[i];
	}
	return difference;
}

void printSum(neuron * n) {
	long double sum = 0;
	for(int i=0;i<100;i++) {
		sum = (sum + n->points[i]);
	}
	cout<<sum<<"\n";
	
}

float findDistance(neuron * input,neuron * sample1, neuron * sample2) {

	double dist12 = dist(sample1,sample2);
	cout<<"dist12 :"<<dist12<<"\n";
	
	cout<<"dist01 :"<<dist(input,sample1)<<"\n";
	cout<<"dist02 :"<<dist(input,sample2)<<"\n";
	printSum(input);
	printSum(sample1);
	printSum(sample2);
	float dotp = dot(diff(input->points,sample1->points),diff(sample2->points,sample1->points));
	//cout<<"dotp : "<<dotp<<"\n";
	float q = dotp/dist12;
	//cout<<"q :"<<q<<"\n";
	getch();
	if(q<=0) {
		return pow(dist(input,sample1),2);
	}
	else if(q>=dist12) {
		return pow(dist(input,sample2),2);
	}
	else {
		return pow(dist(input,sample1),2) - pow(q,2);
	}
}

classtype * identifyClass(neuron * input) {
	int found = 0;
	classtype * currentClass = classlist;
	neuron * sample1 = NULL;
	neuron * sample2 = NULL;
	while(currentClass !=NULL) {
		sample1 = currentClass->head;
		sample2 = sample1->next;
		
		while(sample2!=NULL) {
			
			float d = findDistance(input, sample1, sample2);
			cout<<"d: "<<sqrt(d)<<"\n\n";
			float s = ((d/pow(R,2)) - 1.0);
			if(s<=0) {
				return currentClass;
			}
			sample1=sample2;
			sample2=sample2->next;
		}
		currentClass = currentClass->next;
	}
	return NULL;
}

void KnnListInsert(neuron * sample)
{
	if(KnnListHead==NULL) {
		KnnListHead=sample;
		sample->KnnNext=NULL;
	}
	else
	{ neuron * ptr1=KnnListHead;
	  neuron * ptr2 = NULL;
	  neuron * save = NULL;
	while((ptr1 != NULL)&&(ptr1->KnnDist <= sample->KnnDist)){
			ptr2=ptr1;
			ptr1 = ptr1->KnnNext;
		}
	if(ptr2==NULL)
	{
		save=KnnListHead;
		KnnListHead=sample;
		sample->KnnNext=save;
	}
	else
	{
	ptr2->KnnNext = sample;
	sample->KnnNext=ptr1;
	}
	}
}

int KnnListScan(int k) {

	int * ClassCount=new int[NumClasses];
	for(int i=0; i< NumClasses;i++) {
		ClassCount[i]=0;
	}
	neuron * start=KnnListHead;
	int i = 0;
	while((i < k)&&(start != NULL)) {
		ClassCount[start->classId] += 1;
		start=start->KnnNext;
		i++;
	}
	int max=-1,index;
	for(int i=0;i<NumClasses;i++) {
		if(ClassCount[i]>max)
		{
			max=ClassCount[i];
			index=i;
		}
	}
	int flag=0;
	for(int i=0;i<NumClasses;i++)
	{
			if(ClassCount[i]==max) {
				flag +=1;

			}
	}
	if(flag>1)
	{
		index = KnnListScan(k+1);
	}
	return index;

}
classtype * KNN(neuron * input) {
	KnnListHead=NULL;
	classtype * currentClass = classlist;
	neuron * sample1 = NULL;
	while(currentClass !=NULL) {
		sample1 = currentClass->head;
		while(sample1!=NULL) {
			double d = dist(input, sample1);
			sample1->KnnDist=d;
			KnnListInsert(sample1);			
			sample1=sample1->next;
		}
		currentClass = currentClass->next;
	}
	int index = KnnListScan(K);
	classtype * KNNclass = classlist;
	while(KNNclass!=NULL) {
		if(KNNclass->classId == index)
			return KNNclass;
		KNNclass = KNNclass->next;
	}
}



void save(char filename[20]) {
	ofstream f1;
	f1.open(filename,ios::binary);
	f1.write((char*)&NumClasses,sizeof(NumClasses));
	classtype * temp = classlist;
	while(temp!=NULL) {
		f1.write((char*)temp,sizeof(classtype));
		cout<<"save class: "<<temp->classname<<" id : "<<temp->classId<<"\n";
		temp=temp->next;
	}
	temp = classlist;
	neuron * n;
	while(temp!=NULL) {
		n = temp->head;
		f1.write((char*)&(temp->numSamples),sizeof(temp->numSamples));
		while(n!=NULL) {
			f1.write((char*)n,sizeof(neuron));
			
			n=n->next;
		}
		temp=temp->next;
	}
	cout<<"Saved.\n";
	f1.close();
}

void load(char filename[20]) {
	ifstream f1;
	f1.open(filename,ios::binary);
	int n=0;
	f1.read((char*)&n,sizeof(n));
	cout<<"n :"<<n<<"\n";
	classtype * temp = new classtype;
	for(int i=0;i<n;i++) {
		f1.read((char*)temp,sizeof(classtype));
		addClass(temp->classname);
		
		
		
	}
	
	temp = classlist;
	neuron * nr;
	while(temp!=NULL) {
		f1.read((char*)&n,sizeof(n));
		
		for(int i=0;i<n;i++) {
			nr = new neuron;
			f1.read((char*)nr,sizeof(neuron));
			
			addTrainedSample(nr);
			
		}
		temp=temp->next;
	}
	cout<<"Loaded.\n";
	f1.close();
}

void clearfp() {
	for(int i=0;i<585;i++) {
		fp[i]=0;
	}

}

void test() {
	srand(time(NULL));
	neuron * n1;
	int x[585],x1[585],x2[585],x3[585],x4[585],y[585],y1[585],y2[585],y3[585],y4[585],test[585],r;
	for(int i=0;i<585;i++) {
		r = rand();
		x[i] = -255 + (r%511);
	}

		
	for(int i = 0;i<585;i++) { 
		r = rand();
		x1[i] = x[i] + (r%10);
		r = rand();
		x2[i] = x1[i] + (r%10);
		r = rand();
		x3[i] = x2[i] + (r%10);
		r = rand();
		x4[i] = x3[i] + (r%10);
		
	}

	n1 = createNeuron(0,x);
	addSample(n1);
	n1 = createNeuron(0,x1);
	addSample(n1);
	n1 = createNeuron(0,x2);
	addSample(n1);
	n1 = createNeuron(0,x3);
	addSample(n1);
	n1 = createNeuron(0,x4);
	addSample(n1);

	getch();
	srand(time(NULL));
	for(int i=0;i<585;i++) {
		r = rand();
		y[i] = -255 + (r%511);
	}

	
	
	for(int i = 0;i<585;i++) { 
		r = rand();
		y1[i] = y[i] + (r%10);
		r = rand();
		y2[i] = y1[i] + (r%10);
		r = rand();
		y3[i] = y2[i] + (r%10);
		r = rand();
		y4[i] = y3[i] + (r%10);
		r = rand();
		test[i] = x[i] + (r%10);
	}
	n1 = createNeuron(1,y);
	addSample(n1);
	n1 = createNeuron(1,y1);
	addSample(n1);
	n1 = createNeuron(1,y2);
	addSample(n1);
	n1 = createNeuron(1,y3);
	addSample(n1);
	n1 = createNeuron(1,y4);
	addSample(n1);

	neuron * ntest=createNeuron(0,test);
	classtype * c = KNN(ntest);

	if(c!=NULL){
		cout<<"\n\nClass :"<<c->classname;
	}
	else {
		cout<<"\n\nClass Not found!";
	}
	getch();
}

void main()
{
	
	init();
    //test();
	

	addClass("class A");
	addClass("class D");
	addClass("class E");
	
	neuron * n1;
	/*
	extractor("E:\\studies\\major\\project\\faces2\\working\\b15.JPG");
	n1 = createNeuron(0,fp);
	addSample(n1);
	clearfp();
	
	
	
	extractor("E:\\studies\\major\\project\\faces2\\working\\b16.JPG");
	n1 = createNeuron(0,fp);
	addSample(n1);
	clearfp();
	
	*/
	/*extractor("E:\\studies\\major\\project\\faces2\\working\\b13.JPG");
	n1 = createNeuron(0,fp);
	addSample(n1);
	clearfp();
	
	extractor("E:\\studies\\major\\project\\faces2\\working\\b14.JPG");
	n1 = createNeuron(0,fp);
	addSample(n1);
	clearfp();

	extractor("E:\\studies\\major\\project\\faces2\\working\\b16.JPG");
	n1 = createNeuron(0,fp);
	addSample(n1);
	clearfp();
	*/
	//------------------
	
	extractor("E:\\studies\\major\\project\\faces2\\working\\a2.JPG");
	n1 = createNeuron(0,fp);
	addSample(n1);
	clearfp();

	extractor("E:\\studies\\major\\project\\faces2\\working\\a3.JPG");
	n1 = createNeuron(0,fp);
	addSample(n1);
	clearfp();

	
	extractor("E:\\studies\\major\\project\\faces2\\working\\a5.JPG");
	n1 = createNeuron(0,fp);
	addSample(n1);
	clearfp();
	
	extractor("E:\\studies\\major\\project\\faces2\\working\\d2.JPG");
	n1 = createNeuron(1,fp);
	addSample(n1);
	clearfp();

	extractor("E:\\studies\\major\\project\\faces2\\working\\d3.JPG");
	n1 = createNeuron(1,fp);
	addSample(n1);
	clearfp();
	
	//----------------
	extractor("E:\\studies\\major\\project\\faces2\\working\\e7.JPG");
	n1 = createNeuron(2,fp);
	addSample(n1);
	clearfp();
	
	
	extractor("E:\\studies\\major\\project\\faces2\\working\\e8.JPG");
	n1 = createNeuron(2,fp);
	addSample(n1);
	clearfp();
	/*
	extractor("E:\\studies\\major\\project\\faces2\\working\\e7.JPG");
	n1 = createNeuron(2,fp);
	addSample(n1);
	clearfp();
	
	extractor("E:\\studies\\major\\project\\faces2\\working\\e8.JPG");
	n1 = createNeuron(2,fp);
	addSample(n1);
	clearfp();

	extractor("E:\\studies\\major\\project\\faces2\\working\\e9.JPG");
	n1 = createNeuron(2,fp);
	addSample(n1);
	clearfp();

	*/
		
	
	
	
	
	
	//cout<<NumClasses;
	save("test.txt");
	//getch();
	
	
	
	init();

	load("test.txt");
	//cout<<classlist->classname<<"\n"<<classlist->numSamples;
	//getch();
	
	
	extractor("E:\\studies\\major\\project\\faces2\\working\\d7.JPG");
	neuron * ntest = createNeuron(0,fp);
	
	//classtype * c = KNN(ntest);
	classtype * c = identifyClass(ntest);
	//cout<<"dist "<<dist(classlist->head,classlist->next->head)<<"\n";
	if(c!=NULL){
		cout<<"\n\nClass :"<<c->classname;
	}
	else {
		cout<<"\n\nClass Not found!";
	}
	getch();
	
	//extractor("E:\\studies\\major\\project\\major_Image_Database\\Approved\\06\\ID06_002.jpg");
	//getch();
	

	
}