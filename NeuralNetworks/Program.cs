using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.SupervisedLearning.Progress;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworks
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=========== Neural network ===========");

            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(28, 28),
                NetworkLayers.Convolutional((5, 5), 20, ActivationType.Identity),
                NetworkLayers.Pooling(ActivationType.LeakyReLU),
                NetworkLayers.FullyConnected(100, ActivationType.LeCunTanh),
                NetworkLayers.Softmax(10));

            var rnd = new Random();
            IEnumerable<(float[] x, float[] y)> data = Enumerable.Range(1, 10000)
                .Select(x => (
                    Enumerable.Range(1, 28).Select(y => (float)rnd.Next(999)).ToArray(), 
                    Enumerable.Range(1, 28).Select(y => (float)rnd.Next(999)).ToArray()
                    ));

            ITrainingDataset dataset = DatasetLoader.Training(data, 100);
            ITestDataset testdata = DatasetLoader.Test(data);

            // Train the network
            TrainingSessionResult result = await NetworkManager.TrainNetworkAsync(network,
                dataset,
                TrainingAlgorithms.AdaDelta(),
                60, 0.5f,
                TrackBatchProgress,
                testDataset: testdata);

            Printf($"Stop reason: {result.StopReason}, elapsed time: {result.TrainingTime}");

            Console.ReadKey();
        }
        
        // Prints an output message
        private static void Printf(string text)
        {
            Console.ForegroundColor = ConsoleColor.DarkRed;
            Console.Write(">> ");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"{text}\n");
        }

        // Training monitor
        private static void TrackBatchProgress(BatchProgress progress)
        {
            Console.SetCursorPosition(0, Console.CursorTop);
            int n = (int)(progress.Percentage * 32 / 100); // 32 is the number of progress '=' characters to display
            char[] c = new char[32];
            for (int i = 0; i < 32; i++) c[i] = i <= n ? '=' : ' ';
            Console.Write($"[{new string(c)}] ");
        }
    }
}
