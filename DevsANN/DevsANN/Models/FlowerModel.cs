using DevsANN.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace DevsANN.Models
{
    class FlowerModel
    {
        public double SepalLength { get; set; }
        public double SepalWidth { get; set; }
        public double PetalLength { get; set; }
        public double PetalWidth { get; set; }
        public int FlowerType { get; set; }

        public double GetDataByNeuronID(int index)
        {
            switch (index)
            {
                case 0:
                    return SepalLength; 
                case 1:
                    return SepalWidth;
                case 2:
                    return PetalLength;
                case 3:
                    return PetalWidth;
                default:
                    return 0.0;
            }
        }

        //public List<double> GetDesiredOutputs()
        //{
        //    if (FlowerType.ToLower().Equals(Flowers.setosa.ToString()))
        //        return new List<double> { 1.0, 0.0, 0.0 };
        //    else if (FlowerType.ToLower().Equals(Flowers.versicolor.ToString()))
        //        return new List<double> { 0.0, 1.0, 0.0 };
        //    else
        //        return new List<double> { 0.0, 0.0, 1.0 };
        //}
    }
}
