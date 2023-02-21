using System;
using System.Collections.Generic;
using System.Text;

namespace DevsANN.Data
{
    class ActivationFunctions
    {
        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        public static double SigmoidDerivative(double x)
        {
            double temp = Sigmoid(x);
            return  temp * (1.0 - temp);
        }
    }
}
