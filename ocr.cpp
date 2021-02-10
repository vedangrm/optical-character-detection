#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/ml.hpp>
#include<iostream>
#include<sstream>

const int MIN_CONTOUR_AREA = 100;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

class ContourWithData {
public:
  
    std::vector<cv::Point> ptrContour;           // contour
    cv::Rect boundingRect;                      // bounding rect => contour{this will give a rectangular box encircling the characters}
    float Area;                              // area of contour

                                                
    bool IfContourIsValid() {                              
        if (Area < MIN_CONTOUR_AREA) return false;           
        return true;                                            
    }

    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {     
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   
    }

};


int main() {
    std::vector<ContourWithData> allContoursWithData;           // declaring empty vectors.
    std::vector<ContourWithData> validContoursWithData;         

                                                                

    cv::Mat matClassificationInts;      

    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);       

    if (fsClassifications.isOpened() == false) {                                                    
        std::cout << "error, unable to open training classifications file, exiting program\n\n";    
        return(0);                                                                                 
    }

    fsClassifications["classifications"] >> matClassificationInts;      
    fsClassifications.release();                                       

                                                                        

    cv::Mat matTrainingImagesAsFlattenedFloats;         

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          

    if (fsTrainingImages.isOpened() == false) {                                              
        std::cout << "error, unable to open training images file, exiting program\n\n";         
        return(0);                                                                              
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           
    fsTrainingImages.release();                                                 

                                                                              

    cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());       

                                                                                
                                                                                
    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    cv::Mat matTestingNumbers = cv::imread("Pictures/images.png");         


    if (matTestingNumbers.empty()) {                              
        std::cout << "error: image not read from file\n\n";        
        return(0);                                                
    }

    cv::Mat matGrayscale;           
    cv::Mat matBlurred;            
    cv::Mat matThresh;             
    cv::Mat matThreshCopy;          
    cv::cvtColor(matTestingNumbers, matGrayscale, cv::COLOR_BGR2GRAY);                                                                        
    cv::GaussianBlur(matGrayscale,matBlurred,cv::Size(5, 5),0);                                                      
    cv::adaptiveThreshold(matBlurred,matThresh,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY_INV,11,2);                                   

    matThreshCopy = matThresh.clone();           

    std::vector<std::vector<cv::Point> > ptrContours;        
    std::vector<cv::Vec4i> v4iHierarchy;                    

    cv::findContours(matThreshCopy,ptrContours,v4iHierarchy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);               

    for (int i = 0; i < ptrContours.size(); i++) {               
        ContourWithData contourWithData;                                                    
        contourWithData.ptrContour = ptrContours[i];                                          
        contourWithData.boundingRect = cv::boundingRect(contourWithData.ptrContour);         
        contourWithData.Area = cv::contourArea(contourWithData.ptrContour);               
        allContoursWithData.push_back(contourWithData);                                 
    }

    for (int i = 0; i < allContoursWithData.size(); i++) {                      
        if (allContoursWithData[i].IfContourIsValid()) {                  
            validContoursWithData.push_back(allContoursWithData[i]);            
        }
    }
    
    std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

    std::string strFinalString;         

    for (int i = 0; i < validContoursWithData.size(); i++) {            

                                                                        
        cv::rectangle(matTestingNumbers,                            
            validContoursWithData[i].boundingRect,        
            cv::Scalar(0, 255, 0),                        
            2);                                           

        cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect); 

        cv::Mat matROIResized;
        cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     

        cv::Mat matROIFloat;
        matROIResized.convertTo(matROIFloat, CV_32FC1);            

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

        cv::Mat matCurrentChar(0, 0, CV_32F);

        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

        strFinalString = strFinalString + char(int(fltCurrentChar));       
    }

    std::cout << "\n\n" << "numbers read = " << strFinalString << "\n\n";       

    cv::imshow("matTestingNumbers", matTestingNumbers);     

    cv::waitKey(0);                         

    return(0);
}
