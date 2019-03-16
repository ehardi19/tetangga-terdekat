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

// Data to store attributes of the data
type Data struct {
	X1 float64 // atribut 1
	X2 float64 // atribut 2
	X3 float64 // atribut 3
	X4 float64 // atribut 4
	Y  string  // kelas
}

// DataSplit will split data train into 25% validation data and 75% train data in random manner
func DataSplit(data []Data) ([]Data, []Data) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	val := make([]Data, 1000)
	perm := r.Perm(1000)
	for i, randI := range perm {
		val[i] = data[randI]
	}
	for _, i := range perm {
		data[i] = data[len(data)-1]
		data = data[:len(data)-1]
	}
	return val, data
}

// StringtoData to convert row into Data type form
func StringtoData(row []string) Data {
	var data Data
	var err error
	data.X1, err = strconv.ParseFloat(row[0], 64)
	if err != nil {
		log.Fatal(err)
	}
	data.X2, err = strconv.ParseFloat(row[1], 64)
	if err != nil {
		log.Fatal(err)
	}
	data.X3, err = strconv.ParseFloat(row[2], 64)
	if err != nil {
		log.Fatal(err)
	}
	data.X4, err = strconv.ParseFloat(row[3], 64)
	if err != nil {
		log.Fatal(err)
	}
	data.Y = row[4]

	return data
}

// EuclideanDistance to calculate the distance between two data feature
func EuclideanDistance(a, b Data) float64 {
	return math.Sqrt(math.Pow(a.X1-b.X1, 2) + math.Pow(a.X2-b.X2, 2) + math.Pow(a.X3-b.X3, 2) + math.Pow(a.X4-b.X4, 2))
}

// Inference that holds the distances between two data
type Inference struct {
	Label    string
	Distance float64
}

// getResponse is a function to determine the inference of target data based on a train data by getting K-closest distance
func getResponse(a Data, train []Data, k int) string {
	var distances []Inference
	inf := map[string]int{
		"0": 0,
		"1": 0,
	}

	// Calculate all distance between objects
	for _, b := range train {
		var distance Inference
		distance.Label = b.Y
		distance.Distance = EuclideanDistance(a, b)
		distances = append(distances, distance)
	}

	// Sort the map
	sort.Slice(distances, func(x, y int) bool {
		if distances[x].Distance == distances[y].Distance {
			return distances[x].Label < distances[y].Label
		}
		return distances[x].Distance < distances[y].Distance
	})

	// Get k nearest data a.k.a neighbors
	for i := 0; i < k; i++ {
		if distances[i].Label == "0" {
			inf["0"]++
		} else if distances[i].Label == "1" {
			inf["1"]++
		}
	}

	// Compare the occurence for every 'kelas'
	max := inf["0"]
	key := "0"
	if max < inf["1"] {
		max = inf["1"]
		key = "1"
	}

	// Return the most inference showed up in the first K data.
	return key
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Read the data of given data train
	inputFile, _ := os.Open("./data/DataTrain_Tugas_2_AI.csv.csv")
	reader := csv.NewReader(bufio.NewReader(inputFile))
	defer inputFile.Close()

	// Store the data in array of Data type form
	var datas []Data
	reader.Read()
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		data := StringtoData(line)
		datas = append(datas, data)
	}

	// Split the data into validation and test data
	validation, test := DataSplit(datas)

	fmt.Println(len(validation), len(test))

	// Initialize the best k and accuracy
	best := 1
	accuracy := 0.000

	// Number of try is 100
	for i := 1; i <= 100; i++ {
		correct := 0

		for v := 0; v < len(validation); v++ {
			a := validation[v]
			predict := getResponse(a, test, i)
			if predict == a.Y {
				correct++
			}
		}

		acc := float64(correct) / float64(len(validation))
		fmt.Println("k:", i, "acc:", acc)

		if acc > accuracy {
			accuracy = acc
			best = i
		}
	}

	// Print the best k and its accuracy
	fmt.Println("Best K")
	fmt.Println("k:", best, "acc:", accuracy)

	// Read the data train again for predict the data test
	inputFile, _ = os.Open("./data/DataTrain_Tugas_2_AI.csv.csv")
	reader = csv.NewReader(bufio.NewReader(inputFile))
	defer inputFile.Close()

	// Store the data in dataTrain array
	var dataTrain []Data
	reader.Read()
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		data := StringtoData(line)
		dataTrain = append(dataTrain, data)
	}

	inputFile, _ = os.Open("./data/DataTest_Tugas_2_AI.csv.csv")
	reader = csv.NewReader(bufio.NewReader(inputFile))
	defer inputFile.Close()

	// Store the data in dataTest array
	var dataTest []Data
	reader.Read()
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		data := StringtoData(line)
		dataTest = append(dataTest, data)
	}

	// Use the best k based on our observation
	k := best
	for index, a := range dataTest {
		dataTest[index].Y = getResponse(a, dataTrain, k)
	}

	// Make file for the predictions of the data test
	outputFile, _ := os.Create("Prediksi_Tugas2AI_13-1174099.csv")
	defer outputFile.Close()

	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

	head := []string{
		"atribut 1",
		"atribut 2",
		"atribut 3",
		"atribut 4",
		"kelas",
	}
	if err := writer.Write(head); err != nil {
		log.Fatalln("error writing record to csv:", err)
	}

	// Write the 'kelas' value for the data
	for _, t := range dataTest {
		csvData := []string{
			fmt.Sprintf("%f", t.X1),
			fmt.Sprintf("%f", t.X2),
			fmt.Sprintf("%f", t.X3),
			fmt.Sprintf("%f", t.X4),
			fmt.Sprintf("%s", t.Y),
		}
		if err := writer.Write(csvData); err != nil {
			log.Fatalln("error writing record to csv:", err)
		}
	}
}
