using DevsANN.Data;
using DevsANN.Models;
using DevsANN.Simulators;
using DEVSsharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace DevsANN.Views
{
    class CsvReaderView : Atomic
    {
        private StreamReader sr;
        private List<FlowerModel> data;
        private int currentIndex;
        private readonly string trainFilePath;
        private readonly string testFilePath;

        readonly int neuronsToWaitForNextRead;
        int neuronsToWaitCounter;

        int currentEpoch; // u kojoj epohi se trenutno nalazimo
        public static int passesThorughEpoch; // broj prolazaka po epohi => epochs / batch_size (+1)
        int currentPassageThroughEpoch;

        public InputPort loadTestDataInputPort; // ucitaj podatke za treniranje mreze
        public InputPort loadTrainingDataInputPort;
        public InputPort fetchNextTestExampleInputPort; // zapocni ciklus treniranja za dati podatak
        public InputPort fetchNextTrainingExampleInputPort;
        public List<OutputPort> neuronsDataOutputPort; // koristi se za slanje podataka neuronima u ulaznom sloju.

        public OutputPort startTestPhaseOutputPort;
        public OutputPort showResultsOutputPort;
        public OutputPort desiredResultOutputPort;
        public OutputPort loadTrainingDataOutputPort;

        private Queue<PortValue> msgBuff;
        public CsvReaderView(string name, TimeUnit tu, string trainFilePath, string testFilePath, int neuronsToWaitForNextRead) : base(name, tu)
        {
            this.trainFilePath = trainFilePath;
            this.testFilePath = testFilePath;
            this.neuronsToWaitForNextRead = neuronsToWaitForNextRead;

            loadTestDataInputPort = AddIP(name +"_loadTestDataInputPort");
            loadTrainingDataInputPort = AddIP(name +"_loadTrainingDataInputPort");
            loadTrainingDataOutputPort = AddOP(name + "_loadTrainingDataOutputPort");
            fetchNextTrainingExampleInputPort = AddIP(name +"_fetchNextTrainingExampleInputPort");
            fetchNextTestExampleInputPort = AddIP(name +"_fetchNextTestExampleInputPort");
            startTestPhaseOutputPort = AddOP("startTestPhaseOutputPort");
            showResultsOutputPort = AddOP("showResultsOutputPort");
            desiredResultOutputPort = AddOP(name + "_desiredResultOutputPort");

            neuronsDataOutputPort = new List<OutputPort>();
            for (int i = 0; i < Config.INPUT_LAYER_NEURONS_COUNT; i++)
                neuronsDataOutputPort.Add(AddOP("neuronsDataOutputPort" + i + ""));

            init();
        }
        public override bool delta_x(PortValue x)
        {
            if (x.port == loadTrainingDataInputPort)
            {
                LoadData(trainFilePath, false);
                currentIndex = 0;

                msgBuff.Enqueue(new PortValue(desiredResultOutputPort, data[currentIndex].FlowerType));
                for (int i = 0; i < neuronsDataOutputPort.Count; i++)
                    msgBuff.Enqueue(new PortValue(neuronsDataOutputPort[i], data[currentIndex].GetDataByNeuronID(i))); // salje podatke neuronima u ulaznom sloju
                
                ++currentIndex;
                return true;
            }
            else if (x.port == loadTestDataInputPort)
            {
                LoadData(testFilePath, true);
                currentIndex = 0;
                neuronsToWaitCounter = 0;
                CsvWriterView.testPhase = true;

                msgBuff.Enqueue(new PortValue(desiredResultOutputPort, data[currentIndex].FlowerType));
                for (int i = 0; i < neuronsDataOutputPort.Count; i++)
                    msgBuff.Enqueue(new PortValue(neuronsDataOutputPort[i], data[currentIndex].GetDataByNeuronID(i)));

                ++currentIndex;
                return true;
            }
            else if (x.port == fetchNextTrainingExampleInputPort)
            {
                // ----------- sacekaj odgovor veza izmedju ulaznog i prvog skrivenog sloja ---------
                ++neuronsToWaitCounter;
                if (neuronsToWaitCounter < neuronsToWaitForNextRead)
                    return false;
                else
                    neuronsToWaitCounter = 0;
                // ------------------------------------------------------------------------------
                // ------------------------------------------------------------------------------

                if (currentIndex < data.Count)
                {
                    msgBuff.Enqueue(new PortValue(desiredResultOutputPort, data[currentIndex].FlowerType));
                    for (int i = 0; i < neuronsDataOutputPort.Count; i++)
                        msgBuff.Enqueue(new PortValue(neuronsDataOutputPort[i], data[currentIndex].GetDataByNeuronID(i)));

                    currentIndex++;
                }
                else // procitani su svi podaci iz batch_size-a
                {
                    ++currentPassageThroughEpoch; // uvecavamo broj odradjenih batchSize po epohi

                    if (currentPassageThroughEpoch < passesThorughEpoch) // ucitaj sledeci batch_size za ovu epohu
                        msgBuff.Enqueue(new PortValue(loadTrainingDataOutputPort, null));
                }

                if (currentPassageThroughEpoch == passesThorughEpoch) // odradili smo jednu epohu
                {
                    ++currentEpoch;
                    sr.Close();

                    if (currentEpoch == Config.EPOCHS) // odradjene su sve epohe
                        msgBuff.Enqueue(new PortValue(startTestPhaseOutputPort, null));
                    else
                    {
                        sr = new StreamReader(Config.TRAIN_FILE_PATH);
                        sr.ReadLine();
                        currentPassageThroughEpoch = 0; // resetujemo broj odradjenih batchsize po epohi jer prelazimo na novu epohu
                        msgBuff.Enqueue(new PortValue(loadTrainingDataOutputPort, null));
                    }
                }

                return true;
            }
            else if (x.port == fetchNextTestExampleInputPort)
            {
                if (currentIndex < data.Count)
                {
                    msgBuff.Enqueue(new PortValue(desiredResultOutputPort, data[currentIndex].FlowerType));
                    for (int i = 0; i < neuronsDataOutputPort.Count; i++)
                        msgBuff.Enqueue(new PortValue(neuronsDataOutputPort[i], data[currentIndex].GetDataByNeuronID(i)));

                    ++currentIndex;
                    return true;
                }
                else
                {
                    msgBuff.Enqueue(new PortValue(showResultsOutputPort, null)); // konacan rezultat
                    return true;
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
            currentEpoch = 0;
            passesThorughEpoch = Config.TRAINING_FILE_SIZE % Config.BATCH_SIZE == 0 ? Config.TRAINING_FILE_SIZE / Config.BATCH_SIZE : Config.TRAINING_FILE_SIZE / Config.BATCH_SIZE + 1;
            currentPassageThroughEpoch = 0;

            neuronsToWaitCounter = 0;

            if (sr != null)
                sr.Close();

            sr = new StreamReader(Config.TRAIN_FILE_PATH);
            sr.ReadLine(); // procitaj zaglavlje
            data = new List<FlowerModel>();

            currentIndex = 0;
            msgBuff = new Queue<PortValue>();
        }

        public override double tau()
        {
            if (msgBuff.Count > 0)
                return 0.0;

            return double.PositiveInfinity;
        }
        public void LoadData(string filePath, bool testPhase)
        {
            if (testPhase)
            {
                sr.Close();
                sr = new StreamReader(Config.TEST_FILE_PATH);
                sr.ReadLine();
            }

            data.Clear();
            string line;
            string[] parts;
            int readCount = 0; // broj ucitanih trening podataka

            while ((testPhase || readCount < Config.BATCH_SIZE) && (line = sr.ReadLine()) != null) // ako je testna faza, onda nije bitan batch_size
            {
                ++readCount;
                parts = line.Split(',');
                data.Add(new FlowerModel
                {
                    SepalLength = double.Parse(parts[0]),
                    SepalWidth = double.Parse(parts[1]),
                    PetalLength = double.Parse(parts[2]),
                    PetalWidth = double.Parse(parts[3]),
                    FlowerType = int.Parse(parts[4])
                });
            }
            //data = data.OrderBy(x => NeuralNetworkSimulator.rand.Next()).ToList();
            //sr.Close();
        }

        public override string Get_s()
        {
            return "EPOCH: " + currentEpoch + " / " + Config.EPOCHS + " *** BATCH: " + currentPassageThroughEpoch + " / " + passesThorughEpoch + " *** DATA: " + currentIndex + " / " + (data != null? data.Count : 0) + "";
        }

    }
}
