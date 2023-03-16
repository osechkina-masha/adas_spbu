#ifndef ADAS_SPBU_CONSTSFOROPTICALFLOW_H
#define ADAS_SPBU_CONSTSFOROPTICALFLOW_H
double qualityLevel = 0.3;
double minDistance = 7;
int blockSize = 7;
bool useHarrisDetector = false;
double hassisK = 0.04;

int minPointsToTrack = 5;

int termCriteriaMaxCount = 10;
double termCriteriaEpsilon = 0.03;

int lkWindowWidth = 15;
int lkWindowHeight = 15;
int lkMaxDepth =2;

#endif //ADAS_SPBU_CONSTSFOROPTICALFLOW_H
