using System;
using System.Collections.Generic;
using System.Text;

namespace DevsANN.Data
{
    internal delegate double ActivationFunction(double x);
    internal delegate double ActivationFunctionDerivative(double x);

    internal delegate double LossFunction(double y_true, double y_out);
    internal delegate double LossFunctionDerivative(double y_true, double y_out);
    class Config
    {
        public static int INPUT_LAYER_NEURONS_COUNT = 4;
        public static int HIDDEN_LAYER_NEURONS_COUNT = 2;
        public static int OUTPUT_LAYER_NEURONS_COUNT = 3;
        public static int NUMBER_OF_HIDDEN_LAYERS = 1;

        public static int EPOCHS = 1;
        public static int BATCH_SIZE = 32;
        public static int TRAINING_FILE_SIZE = 120;
        public static double LEARNING_RATE = 1.0;

        public static string TRAIN_FILE_PATH = "../../../Data/train.csv";
        public static string TEST_FILE_PATH = "../../../Data/test.csv";
        public static string OUTPUT_FILE_PATH = "../../../Data/output.csv";

        public static double TAU_DELAY = 0.0;
        public static double EPS_Z = 0.0;

        public static ActivationFunction activationFunction = ActivationFunctions.Sigmoid;
        public static ActivationFunctionDerivative primActivationFunction = ActivationFunctions.SigmoidDerivative;
    }
}
