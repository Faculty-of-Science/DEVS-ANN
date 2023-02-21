using DevsANN.Data;
using DEVSsharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DevsANN.Models
{
    class NeuronModel : Atomic
    {
        readonly double eps_z;
        readonly double tau_delay;
        readonly int outputsCount; // broj izlaza iz ovog neurona
        int waitCounter; // broji jos koliko ulaza treba da stigne kako bi prosledio izlaz dalje
        double previousSignal; // poslednji ulaz u neuron Z_j
        double lastOutput; // poslednji izlaz (ulaz se propusti kroz aktivacionu f-ju) A_j

        public List<InputPort> inputPorts; // slanje podataka na ulaz u neuron
        public OutputPort outputPort; // izlaz iz neurona
        public OutputPort deltaErrorOutputPort; // za koliko se menja greska prilikom promene ulaza ovog neurona (dC / d_ulaz)
        public InputPort startBackPropInputPort; // koristi se samo za neurone u izlaznom sloju
        public InputPort deltaErrorInputPort; // dobijamo za koliko se menja greska prilikom promene ulaza neurona u L + 1 (SLEDECEM) sloju.

        private List<double> deltaInputErrors; // smestaju se (dC / d_ulaz) neurona L + 1 (SLEDECEG) sloja
        private List<double> inputValues; // vrednosti signala
        public SortedList<double, double> taus;

        // ----------- NOVO ---------------
        List<double> errorDerivativesValues;
        List<double> totalErrorDerivativesValues;
        List<double> previousSignalsList;
        int deltaErrorListCounter = 0;
        // --------------------------------

        Queue<PortValue> msgBuff;
        public NeuronModel(string name, TimeUnit tu, int inputsCount, double eps_z, double tau_delay, int outputsCount) : base(name, tu) //wait_for_inputs visak?
        {
            this.eps_z = eps_z;
            this.outputsCount = outputsCount;
            this.tau_delay = tau_delay;

            deltaErrorOutputPort = AddOP(name +"deltaErrorOutputPort");
            deltaErrorInputPort = AddIP(name +"deltaErrorInputPort");
            startBackPropInputPort = AddIP(name +"startBackPropInputPort");
            outputPort = AddOP(name + "outputPort");
            inputPorts = new List<InputPort>();
            for (int i = 0; i < inputsCount; i++)
                inputPorts.Add(AddIP(name + "inputPort" + i + ""));

            init();
        }

        public override bool delta_x(PortValue x)
        {
            for (int i = 0; i < inputPorts.Count; i++)
                if (x.port == inputPorts[i])
                {
                    inputValues[i] = (double)x.value;
                    ++waitCounter;
                    double sum = inputValues.Sum();

                    if (previousSignal == -1.0 || Math.Abs(previousSignal - sum) > eps_z)
                        previousSignal = sum;

                    if (waitCounter == inputPorts.Count) // ako je izlazni neuron moze da izracuna gresku
                    {
                        waitCounter = 0;
                        double triggerTime = TimeCurrent + tau_delay;

                        previousSignalsList.Add(previousSignal);

                        if (taus.ContainsKey(triggerTime))
                            taus[triggerTime] = previousSignal;
                        else
                            taus.Add(triggerTime, previousSignal);

                        return true; // pozovi tau
                    }
                    return false;
                }

            if (x.port == startBackPropInputPort) // za neurone izlaznog sloja
            {
                errorDerivativesValues = (List<double>)x.value; // lista dc/dout

                for (int i = 0; i < errorDerivativesValues.Count; i++)
                    errorDerivativesValues[i] *= Config.primActivationFunction(previousSignalsList[i]);

                msgBuff.Enqueue(new PortValue(deltaErrorOutputPort, errorDerivativesValues));

                return true;
            }
            else if (x.port == deltaErrorInputPort) // ovaj port gadja ConnectionModel
            {
                //string[] parts = x.value.ToString().Split('_');

                errorDerivativesValues = (List<double>)x.value; // na poslednjem mestu je tezina veze
                double connectionWeight = errorDerivativesValues[errorDerivativesValues.Count - 1];
                //errorDerivativesValues.RemoveAt(errorDerivativesValues.Count - 1); // stavi u lokalnu velicinu da ne meris 2 puta. // lista od jednog neurona

                if (deltaErrorListCounter == 0)
                {
                    totalErrorDerivativesValues.Clear();
                    for (int i = 0; i < errorDerivativesValues.Count - 1; i++)
                        totalErrorDerivativesValues.Add(0.0);
                }

                for (int i = 0; i < errorDerivativesValues.Count - 1; i++) // izracunali smo za jednu listu od odredjenog neurona u L + 1 sloju.
                    totalErrorDerivativesValues[i] += (connectionWeight * errorDerivativesValues[i] * Config.primActivationFunction(previousSignalsList[i]));
                deltaErrorListCounter++;

                if (deltaErrorListCounter == outputsCount) // stigle sve promene dC / d_ul neurona L + 1 sloja
                {
                    msgBuff.Enqueue(new PortValue(deltaErrorOutputPort, totalErrorDerivativesValues));
                    deltaErrorListCounter = 0;
                    return true;
                }
                return false;
            }

            return false;
        }
        public override void delta_y(ref PortValue y)
        {
            if (msgBuff.Count > 0)
            {
                y = msgBuff.Dequeue();
                return;
            }

            lastOutput = Config.activationFunction(taus.Values[0]); // koristi se opet
            y.Set(outputPort, lastOutput);
            taus.RemoveAt(0);
            return;
        }
        public override void init()
        {
            previousSignal = -1.0;
            waitCounter = 0;

            deltaInputErrors = new List<double>();
            msgBuff = new Queue<PortValue>();
            inputValues = new List<double>();
            for (int i = 0; i < inputPorts.Count; i++)
                inputValues.Add(0.0);

            taus = new SortedList<double, double>();
            totalErrorDerivativesValues = new List<double>();

            previousSignalsList = new List<double>();
        }
        public override double tau()
        {
            if (taus.Count > 0)
                return taus.Keys[0] - TimeCurrent;

            if (msgBuff.Count > 0)
                return 0.0;

            return double.PositiveInfinity;
        }
    }
}
