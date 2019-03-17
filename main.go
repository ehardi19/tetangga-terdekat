package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"time"
)

// Data is a struct to store attributes of a row in csv
type Data struct {
	X1 float64 // atribut 1
	X2 float64 // atribut 2
	X3 float64 // atribut 3
	X4 float64 // atribut 4
	Y  string  // kelas
}

// StringToData to convert a row in csv in the form of Data type
func StringToData(row []string) Data {
	var dt Data
	var err error
	dt.X1, err = strconv.ParseFloat(row[0], 64)
	if err != nil {
		log.Fatal(err)
	}
	dt.X2, err = strconv.ParseFloat(row[1], 64)
	if err != nil {
		log.Fatal(err)
	}
	dt.X3, err = strconv.ParseFloat(row[2], 64)
	if err != nil {
		log.Fatal(err)
	}
	dt.X4, err = strconv.ParseFloat(row[3], 64)
	if err != nil {
		log.Fatal(err)
	}
	dt.Y = row[4]

	return dt
}

// DataSplit will split data train into 25% data validation and 75% data test in random manner
func DataSplit(data []Data) ([]Data, []Data) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	val := make([]Data, 1000)
	perm := r.Perm(1000)

	// Assign the random data into data validation based on permutation
	for i, randIdx := range perm {
		val[i] = data[randIdx]
	}

	// Get the rest for data test
	for _, i := range perm {
		data[i] = data[len(data)-1]
		data = data[:len(data)-1]
	}

	return val, data
}

// EuclideanDistance to get distances between two datas
func EuclideanDistance(a, b Data) float64 {
	ret := 0.00
	ret += math.Pow(a.X1-b.X1, 2)
	ret += math.Pow(a.X2-b.X2, 2)
	ret += math.Pow(a.X3-b.X3, 2)
	ret += math.Pow(a.X4-b.X4, 2)

	return math.Sqrt(ret)
}

// Point holds the distance between two datas
type Point struct {
	Label    string
	Distance float64
}

// GetNeighbors to get the k-nearest neighbors
func GetNeighbors(a Data, train []Data, k int) []Point {
	var distances []Point

	// Calculate all distance between objects
	for _, b := range train {
		var dist Point
		dist.Label = b.Y
		dist.Distance = EuclideanDistance(a, b)
		distances = append(distances, dist)
	}

	// Sort the array using comparator
	sort.Slice(distances, func(x, y int) bool {
		if distances[x].Distance == distances[y].Distance {
			return distances[x].Label < distances[y].Label
		}
		return distances[x].Distance < distances[y].Distance
	})

	// Store the k neareast data to neighbors
	neighbors := distances[:k]

	return neighbors
}

// GetResponse to get the prediction based on the number of occurence for every classes
func GetResponse(neighbors []Point) string {
	classVotes := map[string]int{
		"0": 0,
		"1": 0,
	}

	// Get the number of occurence for every classes
	for _, x := range neighbors {
		if x.Label == "0" {
			classVotes["0"]++
		} else if x.Label == "1" {
			classVotes["1"]++
		}
	}

	// Compare the occurences of every classes
	var res string
	if classVotes["0"] > classVotes["1"] {
		res = "0"
	} else {
		res = "1"
	}

	return res
}

// GetAccuracy hat sums the total correct predictions and returns the accuracy as a percentage of correct
func GetAccuracy(val []Data, prediction []string) float64 {
	correct := 0

	// Get the sums of total correct
	for i := 0; i < len(val); i++ {
		if val[i].Y == prediction[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(val))
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	// READ THE DATA OF GIVEN DATA TRAIN
	csvFile, _ := os.Open("./data/DataTrain_Tugas_2_AI.csv.csv")
	reader := csv.NewReader(bufio.NewReader(csvFile))
	defer csvFile.Close()

	// Store the data into array of Data
	var dataTrain []Data
	reader.Read()
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		dt := StringToData(row)
		dataTrain = append(dataTrain, dt)
	}

	// READ THE DATA OF GIVEN DATA TEST
	csvFile, _ = os.Open("./data/DataTest_Tugas_2_AI.csv.csv")
	reader = csv.NewReader(bufio.NewReader(csvFile))
	defer csvFile.Close()

	// Store the data in dataTrain array
	var dataTest []Data
	reader.Read()
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		dt := StringToData(row)
		dataTest = append(dataTest, dt)
	}

	// Split data train into data validation and data test
	// Please note data test used for validation is not same as data test used for prediction
	val, test := DataSplit(dataTrain)

	// Print the length of data validation and data test
	fmt.Println("data validation:", len(val), "data test:", len(test))

	// Initialize the best k and its accuracy
	bestK := 1
	bestAcc := 0.00

	// Try to get the best k from 1 to 100
	for i := 1; i <= 100; i++ {
		prediction := []string{}

		// Perform the validation from data test to data validation
		for j := 0; j < len(val); j++ {
			neighbors := GetNeighbors(val[j], test, i)
			result := GetResponse(neighbors)
			prediction = append(prediction, result)
		}
		acc := GetAccuracy(val, prediction)

		// Print the current k and its accuracy
		fmt.Println("k:", i, "acc:", acc)

		if acc > bestAcc {
			bestAcc = acc
			bestK = i
		}
	}

	// Print the result
	fmt.Println("Best k:", bestK, "acc: ", bestAcc)

	// Use the best k based on our observation
	k := bestK

	// Get prediction for given data test
	for i, dt := range dataTest {
		neighbors := GetNeighbors(dt, dataTrain, k)
		dataTest[i].Y = GetResponse(neighbors)
	}

	// Make file out prediction of the data test
	outFile, _ := os.Create("Prediksi_Tugas2AI_1301174099.csv")
	defer outFile.Close()

	writer := csv.NewWriter(outFile)
	defer writer.Flush()

	// Write the class value for the data
	for _, r := range dataTest {
		csvData := []string{
			fmt.Sprintf("%s", r.Y),
		}

		if err := writer.Write(csvData); err != nil {
			log.Fatalln("ERROR WRITING RECORD TO CSV:", err)
		}
	}
}
