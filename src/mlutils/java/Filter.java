package mlutils.java;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.factory.Nd4j;

public abstract class Filter {
    abstract INDArray out(INDArray input);

    public static class None extends Filter {
        public INDArray out(INDArray input) {
            return input;
        }
    }

    public abstract class Classifier extends Filter {
        public static class Threshold extends Filter {
            public double threshold = 0.5;

            public Threshold() {}
            public Threshold(double threshold) {
                this.threshold = threshold;
            }

            public INDArray out(INDArray input) {
                return input.gte(this.threshold).castTo(DataType.FLOAT);
            }
        }

        public static class Argmax extends Filter {
            public INDArray out(INDArray input) {
                return Nd4j.exec(new OneHot(input.argMax(1), input.columns(), 1, 1.0, 0.0))[0].castTo(DataType.FLOAT);
            }
        }

        public static class Closest extends Filter {
            public INDArray out(INDArray input) {
                return Nd4j.math().round(input);
            }
        }
    }
}