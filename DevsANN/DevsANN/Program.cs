using DevsANN.Data;
using DevsANN.Models;
using DevsANN.Simulators;
using DevsANN.Views;
using DEVSsharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace DevsANN
{
    class Program
    {
        static void Main(string[] args)
        {
            SRTEngine engine = new SRTEngine(new NeuralNetworkSimulator("ann"), 5.0, input);
            engine.RunConsoleMenu();

            //double a1 = Config.ActivationFunction(6.4);
            //double a2 = Config.ActivationFunction(2.8);
            //double a3 = Config.ActivationFunction(5.6);
            //double a4 = Config.ActivationFunction(2.2);

            //double ul_skriveni = a1 * -0.85 + a2 * -0.56 + a3 * 0.88 + a4 * 0.07;

            //double iz_skriveni = Config.ActivationFunction(ul_skriveni);
            //Console.WriteLine(iz_skriveni);

            //double o1 = Config.ActivationFunction(iz_skriveni * 0.5);
            //double o2 = Config.ActivationFunction(iz_skriveni * -0.27);
            //double o3 = Config.ActivationFunction(iz_skriveni * 0.81);

            //Console.WriteLine(2.0 * o1 * Config.PrimActivationFunction(iz_skriveni * 0.5));

            //Console.WriteLine();
            //Console.WriteLine(o1);
            //Console.WriteLine(o2);
            //Console.WriteLine(o3);
        }

        private static PortValue input(Devs model)
        {
            NeuralNetworkSimulator nn = (NeuralNetworkSimulator)model;

            if (nn != null)
            {
                if (Console.ReadLine().Trim().Equals("train"))
                    return new PortValue(nn.trainInputPort, null);
                else // testiranje
                    return new PortValue(nn.testInputPort, null);
            }

            return new PortValue(null, null);
        }
    }
}
