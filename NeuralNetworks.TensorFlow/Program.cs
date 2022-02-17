using Keras.Layers;
using Keras.Models;
using Numpy;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using CsvHelper;
using Keras;
using Keras.Callbacks;
using Python.Runtime;

namespace NeuralNetworks.TensorFlow
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            Console.WriteLine("Hello test!");

            using var reader = new StreamReader("C:\\backtest-data\\SPY.csv");
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            
            var records = csv.GetRecords<OHLC>().ToList();

            var xData = records.Select(bar => new []
            {
                bar.Open, bar.High, bar.Low, bar.Close
            }).ToArray();

            var rnd = new Random();
            var yData = records.Select(bar => rnd.Next(0, 2)).ToArray();

            //Load train data
            NDarray x = np.array(CreateRectangularArray(xData)) / 49;
            NDarray y = np.array(yData);


            //Build sequential model
            var model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_shape: new Shape(4)));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dense(1, activation: "linear"));

            //Compile and train
            model.Compile(optimizer: "sgd", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            model.Fit(x, y, batch_size: 2, epochs: 3, verbose: 1);

            //Save model and weights
            string json = model.ToJson();
            File.WriteAllText("model.json", json);
            model.SaveWeight("model.h5");

            //Load model and weight
            var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
            loaded_model.LoadWeight("model.h5");


            NDarray xTest = np.array(CreateRectangularArray(xData.Take(500).ToArray())) / 49;
            NDarray yTest = np.array(yData.Take(500).ToArray());

            loaded_model.Compile(optimizer: "sgd", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });

            var predictions = model.Predict(xTest, 2, verbose: 1);
            var loss = model.TestOnBatch(xTest, yTest);


            Console.WriteLine("Predictions: " + predictions);

            Console.WriteLine("Loss: " + string.Join(", ", loss));

            Console.WriteLine("Done.");
            Console.ReadKey();

        }

        static T[,] CreateRectangularArray<T>(T[][] arrays)
        {
            // TODO: Validation and special-casing for arrays.Count == 0
            int minorLength = arrays[0].Length;
            T[,] ret = new T[arrays.Length, minorLength];
            for (int i = 0; i < arrays.Length; i++)
            {
                var array = arrays[i];
                if (array.Length != minorLength)
                {
                    throw new ArgumentException
                        ("All arrays must be the same length");
                }
                for (int j = 0; j < minorLength; j++)
                {
                    ret[i, j] = array[j];
                }
            }
            return ret;
        }
    }

    public class OHLC
    {
        public float Open { get; set; }
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }
    }
}
