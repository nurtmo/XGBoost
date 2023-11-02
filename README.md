# XGBoost
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using XGBoost;

namespace XGBoostExample
{
    class Program
    {
        static void Main(string[] args)
        {
            // Загрузка данных из CSV файла
            string dataPath = "путь_к_CSV_файлу";
            var data = LoadData(dataPath);

            // Разделение данных на признаки и целевую переменную
            var features = data.Select(d => d.features).ToArray();
            var labels = data.Select(d => d.label).ToArray();

            // Разделение данных на обучающую и тестовую выборки
            double testSize = 0.2;
            int numTestSamples = Convert.ToInt32(data.Length * testSize);
            var xTrain = features.Take(data.Length - numTestSamples).ToArray();
            var yTrain = labels.Take(data.Length - numTestSamples).ToArray();
            var xTest = features.Skip(data.Length - numTestSamples).ToArray();
            var yTest = labels.Skip(data.Length - numTestSamples).ToArray();

            // Создание и обучение модели XGBoost
            var model = new XgbClassifier();
            model.MaxDepth = 3;
            model.LearningRate = 0.1;
            model.NumRound = 100;
            model.Objective = ObjectiveType.BinaryLogistic;

            model.Fit(xTrain, yTrain);

            // Предсказание на тестовой выборке
            var predictions = model.Predict(xTest);

            // Вычисление точности модели
            var accuracy = CalculateAccuracy(predictions, yTest);
            Console.WriteLine("Точность модели XGBoost: " + accuracy);
        }

        static (float[] features, float label)[] LoadData(string filePath)
        {
            var data = new List<(float[] features, float label)>();

            using (var reader = new StreamReader(filePath, Encoding.Default))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');

                    var features = values.Take(values.Length -1).Select(float.Parse).ToArray();
                    var label = float.Parse(values.Last());

                    data.Add((features, label));
                }
            }

            return data.ToArray();
        }

        static float CalculateAccuracy(float[] predictions, float[] labels)
        {
            int correctCount = 0;

            for (int i = 0; i < predictions.Length; i++)
            {
                if ((predictions[i] > 0.5 && labels[i] == 1) || (predictions[i] <= 0.5 && labels[i] == 0))
                {
                    correctCount++;
                }
            }

            return (float)correctCount / predictions.Length;
        }
    }
}
