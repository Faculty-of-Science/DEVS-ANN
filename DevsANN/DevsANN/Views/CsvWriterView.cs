using DevsANN.Data;
using DevsANN.Models;
using DEVSsharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace DevsANN.Views
{
    class CsvWriterView : Atomic
    {
        readonly string inputFilePath;
        readonly string outputFilePath;

        int testDataLength;
        int numberOfHints;
        StreamReader sr;
        StreamWriter sw;

        public List<InputPort> estimatedValuePorts;
        public InputPort showResultsInputPort;
        public List<OutputPort> startBackPropOutputPorts;
        public OutputPort getNextDataOutputPort; // javi readeru da procita sledeci podatak
        public static bool testPhase;

        public List<double> outputValues; // izlazne vrednosti neurona
        private int outputValuesCounter;

        Queue<PortValue> msgBuff;
        public CsvWriterView(string name, TimeUnit tu, string inputFilePath, string outputFilePath) : base(name, tu)
        {
            this.inputFilePath = inputFilePath;
            this.outputFilePath = outputFilePath;

            //correctFlowerTypePort = AddIP("correctFlowerType");
            getNextDataOutputPort = AddOP("getNextDataOutputPort");
            showResultsInputPort = AddIP("showResultsInputPort");

            startBackPropOutputPorts = new List<OutputPort>();
            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
                startBackPropOutputPorts.Add(AddOP("startBackPropOutputPorts" + i));

            estimatedValuePorts = new List<InputPort>();
            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
                estimatedValuePorts.Add(AddIP("estimatedValuePort" + i + ""));

            init();
        }

        public override bool delta_x(PortValue x)
        {
            if (x.port == showResultsInputPort)
            {
                double result = (double)x.value * 100.0;
                sw.WriteLine(TimeCurrent + "," +result.ToString() +"%");
                sw.Flush();
                return false;
            }

            for (int i = 0; i < estimatedValuePorts.Count; i++)
                if (x.port == estimatedValuePorts[i]) // stigla je nova vrednost za upis u csv
                {
                    outputValues[i] = (double)x.value;
                    ++outputValuesCounter;

                    if (outputValuesCounter == estimatedValuePorts.Count) // imamo izlaz svih neurona u izlaznom sloju
                    {
                        //if (!testPhase)
                        //{
                            sw.WriteLine(TimeCurrent + "," + outputValues[0] + "," + outputValues[1] + "," + outputValues[2] + "," + ErrorCalculatorModel.neuronWithCorrectOutput.ToString());
                            sw.Flush();
                            outputValuesCounter = 0;
                        //}
                        outputValuesCounter = 0;
                        return true;
                    }
                    return false;
                }
            return false;
        }

        public override void delta_y(ref PortValue y)
        {
            if (msgBuff.Count > 0)
                y = msgBuff.Dequeue();
            return;
        }

        public override void init()
        {
            testPhase = false;
            numberOfHints = 0;
            testDataLength = 0;
            outputValuesCounter = 0;
            msgBuff = new Queue<PortValue>();
            outputValues = new List<double>();
            for (int i = 0; i < estimatedValuePorts.Count; i++)
                outputValues.Add(0.0);

            if (sr != null)
                sr.Close();
            sr = new StreamReader(inputFilePath);

            if (sw != null)
                sw.Close();
            sw = new StreamWriter(outputFilePath);

            sw.WriteLine("Time [s], Setosa, Versicolor, Virginica, Correct type");
            sw.Flush();
        }
        public override double tau()
        {
            if (msgBuff.Count > 0)
                return 0.0;

            return double.PositiveInfinity;
        }
        //public override string Get_s()
        //{
        //    return "Accuracy: " +((double)numberOfHints / (double)testDataLength * 100.0).ToString();
        //}
    }
}
