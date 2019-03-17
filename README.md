# k-NN Implemented in Go

## Description
k-Neareast Neighbor algorithm (k-NN) is a non-parametric method used for classification. In this case we used k-NN for predict the class of given test data using train data to find the best k for this case.

## Implemenation

### Distances
Calculation of the distance used in this program is using the **Euclidean Distance**. 
![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/795b967db2917cdde7c2da2d1ee327eb673276c0)
Euclidean distance used because it is the "ordinary" straight-line distance between two points.

### Validation
The validation used in this program is using **Cross Validation**. For this program the data propotion for validation set and test set is 25% and 75% respectively in random manner.

### K
Based on the observation of validation process using Cross Validation, we get the best K is 37 with accuracy of 71.5%.


## Result
Open `Prediksi_Tugas2AI_13-1174099.csv`

## Installation
```
 $ go get github.com/ehardi19/tetangga-terdekat
 $ cd $GOPATH/src/github.com/ehardi19/tetangga-terdekat
```

## Running the Program
```
$ go run main.go
```