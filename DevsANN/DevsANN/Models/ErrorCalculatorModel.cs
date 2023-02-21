using DevsANN.Data;
using DevsANN.Views;
using DEVSsharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DevsANN.Models
{
    class ErrorCalculatorModel : Atomic
    {
        public static int neuronWithCorrectOutput; // neuron koji treba da da tacan izlaz
        public int outputValuesCounter; // broj neurona izlaznog sloja koji su poslali svoje izlaze.
        public List<double> outputValues; // izlazi neurona izlaznog sloja
        int testDataLength;
        int numberOfHits;

        // ----------- NOVO ---------------
        List<List<double>> errorDerivativeValues;
        int dataInsideBatch = 0;
        int readBatches = 0;
        // --------------------------------

        public List<InputPort> neuronOutputInputsPorts; // izlazi neurona izlaznog sloja
        public InputPort desiredOutputInputPort;
        public InputPort calculateAccuracyInputPort;

        public List<OutputPort> errorDerivativeOutputPorts; // dC / dActivation
        public OutputPort accuracyOutputPort; // javlja csvWriteru tacnost mreze za upis
        public OutputPort fetchNextDataOutputPort; // javlja csvReaderu da procita sledeci podatak

        private Queue<PortValue> msgBuff;

        public ErrorCalculatorModel(string name, TimeUnit tu) : base(name, tu)
        {
            desiredOutputInputPort = AddIP(name +"_desiredOutputInputPort");
            calculateAccuracyInputPort = AddIP(name + "_calculateAccuracyInputPort");
            accuracyOutputPort = AddOP(name + "_accuracyOutputPort");
            fetchNextDataOutputPort = AddOP(name + "_fetchNextDataOutputPort");

            neuronOutputInputsPorts = new List<InputPort>();
            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
                neuronOutputInputsPorts.Add(AddIP(name + "_outputFromNeuron" + i.ToString()));

            errorDerivativeOutputPorts = new List<OutputPort>();
            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
                errorDerivativeOutputPorts.Add(AddOP(name + "_errDer" + i.ToString()));

            init();
        }

        public override bool delta_x(PortValue x)
        {
            if (x.port == desiredOutputInputPort) // READER javlja koji je izlaz za trenutni trening primer
            {
                neuronWithCorrectOutput = (int)x.value;
                return false;
            }
            else if (x.port == calculateAccuracyInputPort)
            {
                msgBuff.Enqueue(new PortValue(accuracyOutputPort, (double)numberOfHits / testDataLength));
                return true;
            }

            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
                if (x.port == neuronOutputInputsPorts[i])
                {
                    if (!CsvWriterView.testPhase)
                    {
                        if (dataInsideBatch == Config.BATCH_SIZE)
                            for (int j = 0; j < Config.OUTPUT_LAYER_NEURONS_COUNT; j++)
                                errorDerivativeValues[j].Clear();

                        double errorDerivative = 2.0 * ((double)x.value - (i == neuronWithCorrectOutput ? 1.0 : 0.0));
                        ++outputValuesCounter;
                        errorDerivativeValues[i].Add(errorDerivative); // dodaj novu vrednost delta greske odgovarajucem 

                        if (outputValuesCounter == Config.OUTPUT_LAYER_NEURONS_COUNT)
                        {
                            outputValuesCounter = 0;
                            ++dataInsideBatch; // broj trening primera koje su obradili svi neuroni izlaznog sloja
                            // stigla su sva tri odgovora, proveri da slucajno nije kraj jednog batch-a

                            bool temp = (readBatches == CsvReaderView.passesThorughEpoch - 1) && (dataInsideBatch == Config.TRAINING_FILE_SIZE % Config.BATCH_SIZE);
                            //if (deltaWeightsCounter == Config.BATCH_SIZE || temp)

                            if (dataInsideBatch == Config.BATCH_SIZE || temp) // kraj jednog batcha, prosledi BP
                            {
                                ++readBatches;

                                if (readBatches == CsvReaderView.passesThorughEpoch)
                                    readBatches = 0;

                                dataInsideBatch = 0; // broj procitanih podataka unutar batch-a
                                for (int j = 0; j < Config.OUTPUT_LAYER_NEURONS_COUNT; j++)
                                    msgBuff.Enqueue(new PortValue(errorDerivativeOutputPorts[j], errorDerivativeValues[j]));
                            }
                            else // ako nije kraj batcha, ucitaj sledeci podatak iz batcha
                                msgBuff.Enqueue(new PortValue(fetchNextDataOutputPort, null));

                            return true;
                        }
                        else
                            return false;
                    }
                    else
                    {
                        ++outputValuesCounter;
                        outputValues[i] = (double)x.value;
                        if (outputValuesCounter == Config.OUTPUT_LAYER_NEURONS_COUNT)
                        {
                            outputValuesCounter = 0;

                            double maxValue = outputValues.Max();
                            int index = outputValues.IndexOf(maxValue);

                            ++testDataLength;
                            if (index == neuronWithCorrectOutput)
                                ++numberOfHits;
                            msgBuff.Enqueue(new PortValue(fetchNextDataOutputPort, null));
                            return true;
                        }
                        else
                            return false;
                    }
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
            msgBuff = new Queue<PortValue>();
            outputValuesCounter = 0;
            outputValues = new List<double>();
            errorDerivativeValues = new List<List<double>>();
            for (int i = 0; i < Config.OUTPUT_LAYER_NEURONS_COUNT; i++)
            {
                outputValues.Add(0.0);
                errorDerivativeValues.Add(new List<double>());
            }

            testDataLength = 0;
            numberOfHits = 0;
        }

        public override double tau()
        {
            if (msgBuff.Count > 0)
                return 0.0;

            return double.PositiveInfinity;
        }

        public override string Get_s()
        {
            return "Accuracy: " +((double)numberOfHits / (double)testDataLength * 100.0).ToString();
        }
    }
}
